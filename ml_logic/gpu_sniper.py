"""GPU VM sniper for Compute Engine.

This is a grisounet-native rewrite inspired by the public
https://github.com/Stefgug/gpu-sniper project. The upstream repository does
not ship a license file, so this implementation keeps only the high-level idea
and is adapted to the local grisounet configuration model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZONE_FILTERS = (
    "europe-west4",  # Netherlands
    "europe-west2",  # London
    "europe-west3",  # Germany
)
DEFAULT_TARGETS_JSON = json.dumps(
    [
        {
            "gpu_type": "nvidia-l4",
            "machine_type": "g2-standard-4",
            "attach_accelerator": False,
            "accelerator_count": 1,
        },
        {
            "gpu_type": "nvidia-tesla-t4",
            "machine_type": "n1-standard-4",
            "attach_accelerator": True,
            "accelerator_count": 1,
        },
    ]
)


class GcloudError(RuntimeError):
    """Raised when gcloud returns a non-zero exit code."""


@dataclass(frozen=True)
class GpuTarget:
    """GPU target and matching machine configuration."""

    gpu_type: str
    machine_type: str
    attach_accelerator: bool = True
    accelerator_count: int = 1

    @property
    def short_name(self) -> str:
        value = self.gpu_type.lower()
        value = value.removeprefix("nvidia-tesla-")
        value = value.removeprefix("nvidia-")
        return value

    @property
    def standard_quota_metric(self) -> str:
        upper_name = self.gpu_type.upper()
        upper_name = upper_name.replace("NVIDIA-TESLA-", "NVIDIA_")
        upper_name = upper_name.replace("NVIDIA-", "NVIDIA_")
        upper_name = upper_name.replace("-", "_")
        return f"{upper_name}_GPUS"

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "GpuTarget":
        gpu_type = str(payload["gpu_type"]).strip()
        machine_type = str(payload["machine_type"]).strip()
        attach_accelerator = bool(payload.get("attach_accelerator", True))
        accelerator_count = int(payload.get("accelerator_count", 1))
        if not gpu_type or not machine_type:
            raise ValueError("Each GPU target needs gpu_type and machine_type.")
        if accelerator_count < 1:
            raise ValueError("accelerator_count must be >= 1.")
        return cls(
            gpu_type=gpu_type,
            machine_type=machine_type,
            attach_accelerator=attach_accelerator,
            accelerator_count=accelerator_count,
        )


@dataclass(frozen=True)
class SniperConfig:
    """Runtime configuration for the GPU sniper."""

    project_id: str
    instance_name_base: str
    zone_filters: tuple[str, ...]
    max_retries: int
    retry_delay: int
    max_workers: int
    image_family: str
    image_project: str
    boot_disk_size_gb: int
    targets: tuple[GpuTarget, ...]
    labels: tuple[str, ...]
    provisioning_model: str
    startup_script: str | None
    dry_run: bool = False


def _ensure_dotenv_loaded() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _coerce_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(entry.strip() for entry in value.split(",") if entry.strip())


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "gpu"


def parse_targets_json(raw_targets: str | None) -> tuple[GpuTarget, ...]:
    payload = json.loads(raw_targets or DEFAULT_TARGETS_JSON)
    if not isinstance(payload, list) or not payload:
        raise ValueError("GPU target configuration must be a non-empty JSON list.")
    return tuple(GpuTarget.from_mapping(item) for item in payload)


def _default_zone_filters() -> tuple[str, ...]:
    filters = _coerce_csv(os.getenv("GPU_SNIPER_ZONE_FILTERS"))
    if filters:
        return filters
    return DEFAULT_ZONE_FILTERS


def _default_project_id() -> str:
    return (
        os.getenv("GCP_COMPUTE_PROJECT")
        or os.getenv("GCP_PROJECT")
        or "grisounet"
    )


def build_parser() -> argparse.ArgumentParser:
    _ensure_dotenv_loaded()

    parser = argparse.ArgumentParser(
        description="Continuously try to create a GPU VM until capacity is available."
    )
    parser.add_argument(
        "--project",
        default=_default_project_id(),
        help="Compute Engine project to use.",
    )
    parser.add_argument(
        "--instance-name-base",
        default=os.getenv("GPU_SNIPER_INSTANCE_NAME_BASE", "grisou-gpu"),
        help="Base name used when creating the VM.",
    )
    parser.add_argument(
        "--region-filter",
        action="append",
        default=[],
        help="Substring filter for zones to probe. Repeat to add more regions/zones.",
    )
    parser.add_argument(
        "--gpu",
        action="append",
        default=[],
        help="Optional GPU selection. Repeat or use full types like nvidia-l4.",
    )
    parser.add_argument(
        "--targets-json",
        default=os.getenv("GPU_SNIPER_TARGETS_JSON"),
        help="JSON list of GPU targets. Defaults to built-in L4/T4 settings.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("GPU_SNIPER_MAX_RETRIES", "-1")),
        help="-1 retries forever; 0 does not attempt; positive values limit waves.",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=int(os.getenv("GPU_SNIPER_RETRY_DELAY", "120")),
        help="Seconds to sleep between waves.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("GPU_SNIPER_MAX_WORKERS", "6")),
        help="Parallel instance creation attempts per wave.",
    )
    parser.add_argument(
        "--image-family",
        default=os.getenv("GPU_SNIPER_IMAGE_FAMILY") or os.getenv("IMAGE_FAMILY") or "ubuntu-2404-lts-amd64",
        help="Boot disk image family.",
    )
    parser.add_argument(
        "--image-project",
        default=os.getenv("GPU_SNIPER_IMAGE_PROJECT") or os.getenv("IMAGE_PROJECT") or "ubuntu-os-cloud",
        help="Project hosting the boot image family.",
    )
    parser.add_argument(
        "--boot-disk-size-gb",
        type=int,
        default=int(os.getenv("GPU_SNIPER_BOOT_DISK_SIZE_GB", "200")),
        help="Boot disk size in GB.",
    )
    parser.add_argument(
        "--labels",
        default=os.getenv("GPU_SNIPER_LABELS", "source=grisounet,role=gpu-sniper"),
        help="Comma-separated labels to attach to created instances.",
    )
    parser.add_argument(
        "--provisioning-model",
        choices=("STANDARD", "SPOT"),
        default=os.getenv("GPU_SNIPER_PROVISIONING_MODEL", "STANDARD").upper(),
        help="Provisioning model for new instances.",
    )
    parser.add_argument(
        "--startup-script",
        default=os.getenv("GPU_SNIPER_STARTUP_SCRIPT"),
        help="Optional path passed via --metadata-from-file startup-script=...",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover zones and print the gcloud commands without creating anything.",
    )
    return parser


def _select_targets(
    all_targets: Sequence[GpuTarget],
    raw_selections: Sequence[str],
) -> tuple[GpuTarget, ...]:
    if not raw_selections:
        return tuple(all_targets)

    selections = {value.strip().lower() for value in raw_selections if value.strip()}
    chosen = [
        target
        for target in all_targets
        if target.gpu_type.lower() in selections or target.short_name in selections
    ]
    if not chosen:
        raise ValueError(
            "No GPU target matched the requested selection(s): "
            + ", ".join(sorted(selections))
        )
    return tuple(chosen)


def build_config_from_args(args: argparse.Namespace) -> SniperConfig:
    parsed_targets = parse_targets_json(args.targets_json)
    targets = _select_targets(parsed_targets, args.gpu)

    if not args.project.strip():
        raise ValueError("A Compute Engine project is required.")
    if args.max_workers < 1:
        raise ValueError("max-workers must be >= 1.")
    if args.retry_delay < 0:
        raise ValueError("retry-delay must be >= 0.")
    if args.boot_disk_size_gb < 1:
        raise ValueError("boot-disk-size-gb must be >= 1.")

    startup_script = args.startup_script
    if startup_script:
        path = Path(startup_script).expanduser()
        if not path.is_file():
            raise ValueError(f"Startup script not found: {path}")
        startup_script = str(path)

    labels = _coerce_csv(args.labels)
    if not labels:
        labels = ("source=grisounet", "role=gpu-sniper")

    return SniperConfig(
        project_id=args.project.strip(),
        instance_name_base=_normalize_name(args.instance_name_base),
        zone_filters=tuple(
            dict.fromkeys(
                value.strip()
                for value in (args.region_filter or list(_default_zone_filters()))
                if value.strip()
            )
        ),
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        max_workers=args.max_workers,
        image_family=args.image_family.strip(),
        image_project=args.image_project.strip(),
        boot_disk_size_gb=args.boot_disk_size_gb,
        targets=targets,
        labels=labels,
        provisioning_model=args.provisioning_model,
        startup_script=startup_script,
        dry_run=args.dry_run,
    )


def _gcloud_env(project_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CLOUDSDK_CORE_PROJECT"] = project_id
    env.setdefault("CLOUDSDK_CORE_DISABLE_PROMPTS", "1")
    return env


def run_gcloud(
    command: Sequence[str],
    *,
    project_id: str,
    expect_json: bool = False,
    tolerate_errors: bool = False,
) -> object:
    cmd = list(command)
    if expect_json and "--format=json" not in cmd:
        cmd.append("--format=json")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_gcloud_env(project_id),
        check=False,
    )
    if result.returncode != 0:
        if tolerate_errors:
            return None
        stderr = result.stderr.strip() or result.stdout.strip() or "gcloud command failed"
        raise GcloudError(stderr)

    if expect_json:
        stdout = result.stdout.strip()
        return json.loads(stdout) if stdout else {}
    return result


def build_instance_name(base_name: str, target: GpuTarget, zone: str) -> str:
    gpu_slug = _normalize_name(target.short_name)
    zone_slug = _normalize_name(zone)
    suffix = f"-{gpu_slug}-{zone_slug}"
    max_base_length = max(1, 63 - len(suffix))
    trimmed_base = _normalize_name(base_name)[:max_base_length].rstrip("-")
    return f"{trimmed_base or 'gpu'}{suffix}"


def build_create_command(
    config: SniperConfig,
    target: GpuTarget,
    zone: str,
) -> list[str]:
    instance_name = build_instance_name(config.instance_name_base, target, zone)
    labels = list(config.labels) + [f"gpu={_normalize_name(target.short_name)}"]
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        instance_name,
        f"--project={config.project_id}",
        f"--zone={zone}",
        f"--machine-type={target.machine_type}",
        "--maintenance-policy=TERMINATE",
        f"--provisioning-model={config.provisioning_model}",
        f"--image-family={config.image_family}",
        f"--image-project={config.image_project}",
        f"--boot-disk-size={config.boot_disk_size_gb}GB",
        "--scopes=cloud-platform",
        f"--labels={','.join(labels)}",
        "--quiet",
    ]
    if target.attach_accelerator:
        command.append(
            f"--accelerator=type={target.gpu_type},count={target.accelerator_count}"
        )
    if config.startup_script:
        command.append(
            f"--metadata-from-file=startup-script={config.startup_script}"
        )
    return command


def zone_to_region(zone: str) -> str:
    if zone.count("-") < 2:
        return zone
    return zone.rsplit("-", 1)[0]


def quota_metric_for_target(target: GpuTarget, provisioning_model: str) -> str:
    prefix = "PREEMPTIBLE_" if provisioning_model == "SPOT" else ""
    return f"{prefix}{target.standard_quota_metric}"


class GpuSniper:
    """Coordinates discovery, quota checks, and repeated instance creation."""

    def __init__(self, config: SniperConfig) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self.print_lock = threading.Lock()

    def log(self, message: str) -> None:
        with self.print_lock:
            print(message)

    def warn_if_project_mismatch(self) -> None:
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
        except FileNotFoundError as exc:
            self.log(f"[WARN] gcloud is not available in PATH: {exc}")
            return

        current = result.stdout.strip()
        if current in {"", "(unset)"}:
            self.log(
                f"[INFO] No active gcloud project set. Commands will target {self.config.project_id}."
            )
        elif current != self.config.project_id:
            self.log(
                f"[INFO] Active gcloud project is {current}; this run will override to {self.config.project_id}."
            )

    def discover_zones_by_target(self) -> dict[GpuTarget, tuple[str, ...]]:
        accelerator_types = run_gcloud(
            ["gcloud", "compute", "accelerator-types", "list", f"--project={self.config.project_id}"],
            project_id=self.config.project_id,
            expect_json=True,
        )
        assert isinstance(accelerator_types, list)

        zones_by_gpu_type: dict[str, set[str]] = {
            target.gpu_type: set() for target in self.config.targets
        }
        filters = tuple(filter_value.lower() for filter_value in self.config.zone_filters)

        for item in accelerator_types:
            if not isinstance(item, dict):
                continue
            gpu_type = item.get("name")
            zone = str(item.get("zone", "")).split("/")[-1]
            if gpu_type not in zones_by_gpu_type or not zone:
                continue
            if filters and not any(filter_value in zone.lower() for filter_value in filters):
                continue
            zones_by_gpu_type[str(gpu_type)].add(zone)

        return {
            target: tuple(sorted(zones_by_gpu_type[target.gpu_type]))
            for target in self.config.targets
        }

    def discover_region_quotas(
        self,
        zones_by_target: dict[GpuTarget, tuple[str, ...]],
    ) -> dict[str, dict[str, dict[str, float]]]:
        regions = {
            zone_to_region(zone)
            for zones in zones_by_target.values()
            for zone in zones
        }
        quotas_by_region: dict[str, dict[str, dict[str, float]]] = {}

        for region in sorted(regions):
            payload = run_gcloud(
                ["gcloud", "compute", "regions", "describe", region, f"--project={self.config.project_id}"],
                project_id=self.config.project_id,
                expect_json=True,
                tolerate_errors=True,
            )
            if not isinstance(payload, dict):
                continue
            quotas = payload.get("quotas", [])
            if not isinstance(quotas, list):
                continue

            metric_index: dict[str, dict[str, float]] = {}
            for quota in quotas:
                if not isinstance(quota, dict):
                    continue
                metric = quota.get("metric")
                if not metric:
                    continue
                metric_index[str(metric)] = {
                    "limit": float(quota.get("limit", 0) or 0),
                    "usage": float(quota.get("usage", 0) or 0),
                }
            quotas_by_region[region] = metric_index

        return quotas_by_region

    def log_quota_summary(
        self,
        zones_by_target: dict[GpuTarget, tuple[str, ...]],
        quotas_by_region: dict[str, dict[str, dict[str, float]]],
    ) -> None:
        self.log("[Init] Regional GPU quota summary:")
        for target in self.config.targets:
            zones = zones_by_target.get(target, ())
            regions = tuple(dict.fromkeys(zone_to_region(zone) for zone in zones))
            if not regions:
                self.log(f"  {target.gpu_type}: no matching zones discovered.")
                continue

            metric_name = quota_metric_for_target(
                target, self.config.provisioning_model
            )
            lines = []
            for region in regions:
                quota = quotas_by_region.get(region, {}).get(metric_name)
                if quota:
                    lines.append(
                        f"{region}={int(quota['usage'])}/{int(quota['limit'])}"
                    )
                else:
                    lines.append(f"{region}=unknown")

            self.log(f"  {target.gpu_type}: {', '.join(lines)}")

    def build_tasks(
        self,
        zones_by_target: dict[GpuTarget, tuple[str, ...]],
    ) -> list[tuple[str, GpuTarget]]:
        tasks: list[tuple[str, GpuTarget]] = []
        for target, zones in zones_by_target.items():
            self.log(f"[Init] {target.gpu_type}: {len(zones)} zone(s) detected.")
            tasks.extend((zone, target) for zone in zones)
        return tasks

    def log_dry_run(self, tasks: Sequence[tuple[str, GpuTarget]]) -> int:
        self.log("[Dry Run] No instances will be created.")
        for zone, target in tasks:
            command = build_create_command(self.config, target, zone)
            self.log("  " + " ".join(command))
        return 0

    def instance_exists(self, instance_name: str, zone: str) -> bool:
        payload = run_gcloud(
            [
                "gcloud",
                "compute",
                "instances",
                "describe",
                instance_name,
                f"--project={self.config.project_id}",
                f"--zone={zone}",
            ],
            project_id=self.config.project_id,
            expect_json=True,
            tolerate_errors=True,
        )
        return isinstance(payload, dict) and payload.get("name") == instance_name

    def create_vm(self, zone: str, target: GpuTarget) -> bool:
        if self.stop_event.is_set():
            return False

        instance_name = build_instance_name(self.config.instance_name_base, target, zone)
        command = build_create_command(self.config, target, zone)
        self.log(f"[Start] Attempting {target.gpu_type} in {zone} as {instance_name}...")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=_gcloud_env(self.config.project_id),
        )

        if self.stop_event.is_set():
            return False

        if result.returncode == 0:
            self.stop_event.set()
            self.log(f"[SUCCESS] GPU {target.gpu_type} available in {zone}.")
            self.log(
                f"[SSH] gcloud compute ssh {instance_name} --project={self.config.project_id} --zone={zone}"
            )
            return True

        stderr = (result.stderr or "").strip()
        lowered = stderr.lower()

        if "already exists" in lowered:
            if self.instance_exists(instance_name, zone):
                self.stop_event.set()
                self.log(f"[SUCCESS] Instance {instance_name} already exists in {zone}.")
                self.log(
                    f"[SSH] gcloud compute ssh {instance_name} --project={self.config.project_id} --zone={zone}"
                )
                return True

            last_line = stderr.splitlines()[-1] if stderr else "Unexpected error."
            self.log(
                f"[Fail] {zone} ({target.gpu_type}): create reported an existing resource, "
                f"but instance {instance_name} was not found. {last_line}"
            )
            return False
        if any(
            marker in lowered
            for marker in (
                "does not have enough resources available",
                "resource_availability",
                "currently unavailable",
                "not available in zone",
                "zone_resource_pool_exhausted",
                "stockout",
                "sold out",
            )
        ):
            self.log(f"[Fail] {zone} ({target.gpu_type}): capacity exhausted.")
        elif "quota" in lowered:
            self.log(f"[Fail] {zone} ({target.gpu_type}): quota error.")
        elif "permission" in lowered or "required 'compute.instances.create'" in lowered:
            self.log(f"[Fail] {zone} ({target.gpu_type}): permission denied.")
        else:
            last_line = stderr.splitlines()[-1] if stderr else "Unexpected error."
            self.log(f"[Fail] {zone} ({target.gpu_type}): {last_line}")
        return False

    def run(self) -> int:
        self.warn_if_project_mismatch()
        self.log("[Init] Discovering eligible zones...")

        try:
            zones_by_target = self.discover_zones_by_target()
        except FileNotFoundError as exc:
            self.log(f"[Error] gcloud not found: {exc}")
            return 1
        except GcloudError as exc:
            self.log(f"[Error] Failed to discover accelerator types: {exc}")
            return 1

        tasks = self.build_tasks(zones_by_target)
        if not tasks:
            self.log("[Error] No matching GPU zones found for the requested filters.")
            return 1

        quotas_by_region = self.discover_region_quotas(zones_by_target)
        self.log_quota_summary(zones_by_target, quotas_by_region)
        self.log(f"[Init] Total combinations to test: {len(tasks)}")

        if self.config.dry_run:
            return self.log_dry_run(tasks)

        if self.config.max_retries == 0:
            self.log("[Stop] max-retries=0, nothing to do.")
            return 0

        attempt_count = 0
        while not self.stop_event.is_set():
            attempt_count += 1
            wave_tasks = list(tasks)
            random.shuffle(wave_tasks)
            self.log(f"[Wave {attempt_count}] Running with {self.config.max_workers} workers...")

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [
                    executor.submit(self.create_vm, zone, target)
                    for zone, target in wave_tasks
                ]

                for future in as_completed(futures):
                    try:
                        success = future.result()
                    except FileNotFoundError as exc:
                        self.log(f"[Error] gcloud not found: {exc}")
                        self.stop_event.set()
                        executor.shutdown(wait=False, cancel_futures=True)
                        return 1
                    except Exception as exc:  # pragma: no cover - defensive path
                        self.log(f"[Error] Worker crashed: {exc}")
                        continue

                    if success:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

            if self.stop_event.is_set():
                break

            if self.config.max_retries != -1 and attempt_count >= self.config.max_retries:
                self.log("[Stop] Retry limit reached without a successful VM creation.")
                break

            self.log(f"[Sleep] Waiting {self.config.retry_delay}s before the next wave...")
            time.sleep(self.config.retry_delay)

        return 0 if self.stop_event.is_set() else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = build_config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    sniper = GpuSniper(config)
    return sniper.run()


if __name__ == "__main__":
    sys.exit(main())
