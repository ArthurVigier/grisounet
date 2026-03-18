from pathlib import Path
import sys

from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_logic import gpu_sniper as gpu_sniper_module
from ml_logic.gpu_sniper import DEFAULT_ZONE_FILTERS
from ml_logic.gpu_sniper import DEFAULT_TARGETS_JSON
from ml_logic.gpu_sniper import GpuSniper
from ml_logic.gpu_sniper import GpuTarget
from ml_logic.gpu_sniper import SniperConfig
from ml_logic.gpu_sniper import build_config_from_args
from ml_logic.gpu_sniper import build_parser
from ml_logic.gpu_sniper import build_create_command
from ml_logic.gpu_sniper import build_instance_name
from ml_logic.gpu_sniper import parse_targets_json


def make_config(target: GpuTarget) -> SniperConfig:
    return SniperConfig(
        project_id="grisounet",
        instance_name_base="grisou-gpu",
        zone_filters=("europe-west1",),
        max_retries=-1,
        retry_delay=120,
        max_workers=4,
        image_family="ubuntu-2404-lts-amd64",
        image_project="ubuntu-os-cloud",
        boot_disk_size_gb=200,
        targets=(target,),
        labels=("source=grisounet",),
        provisioning_model="STANDARD",
        startup_script=None,
        dry_run=False,
    )


def test_parse_targets_json_uses_defaults():
    targets = parse_targets_json(DEFAULT_TARGETS_JSON)

    assert [target.gpu_type for target in targets] == [
        "nvidia-l4",
        "nvidia-tesla-t4",
    ]
    assert targets[0].attach_accelerator is False
    assert targets[1].attach_accelerator is True


def test_build_create_command_omits_accelerator_for_g2_l4():
    target = GpuTarget(
        gpu_type="nvidia-l4",
        machine_type="g2-standard-4",
        attach_accelerator=False,
    )
    config = make_config(target)

    command = build_create_command(config, target, "europe-west1-b")

    assert "--machine-type=g2-standard-4" in command
    assert not any(part.startswith("--accelerator=") for part in command)


def test_build_create_command_includes_accelerator_for_t4():
    target = GpuTarget(
        gpu_type="nvidia-tesla-t4",
        machine_type="n1-standard-4",
        attach_accelerator=True,
        accelerator_count=1,
    )
    config = make_config(target)

    command = build_create_command(config, target, "europe-west1-b")

    assert "--machine-type=n1-standard-4" in command
    assert "--accelerator=type=nvidia-tesla-t4,count=1" in command


def test_build_instance_name_keeps_gcloud_length_limit():
    target = GpuTarget(gpu_type="nvidia-tesla-t4", machine_type="n1-standard-4")

    name = build_instance_name("grisou" * 20, target, "europe-west1-b")

    assert len(name) <= 63
    assert name.endswith("-t4-europe-west1-b")


def test_default_zone_filters_prefer_netherlands_london_and_germany(monkeypatch):
    monkeypatch.delenv("GPU_SNIPER_ZONE_FILTERS", raising=False)

    parser = build_parser()
    args = parser.parse_args([])
    config = build_config_from_args(args)

    assert config.zone_filters == DEFAULT_ZONE_FILTERS


def test_create_vm_does_not_report_success_when_instance_is_missing(monkeypatch):
    target = GpuTarget(
        gpu_type="nvidia-tesla-t4",
        machine_type="n1-standard-4",
        attach_accelerator=True,
    )
    sniper = GpuSniper(make_config(target))

    monkeypatch.setattr(
        gpu_sniper_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stderr="ERROR: The resource already exists.",
            stdout="",
        ),
    )
    monkeypatch.setattr(GpuSniper, "instance_exists", lambda self, name, zone: False)

    assert sniper.create_vm("europe-west1-c", target) is False
    assert not sniper.stop_event.is_set()


def test_create_vm_reports_success_only_when_existing_instance_is_real(monkeypatch):
    target = GpuTarget(
        gpu_type="nvidia-tesla-t4",
        machine_type="n1-standard-4",
        attach_accelerator=True,
    )
    sniper = GpuSniper(make_config(target))

    monkeypatch.setattr(
        gpu_sniper_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stderr="ERROR: The resource already exists.",
            stdout="",
        ),
    )
    monkeypatch.setattr(GpuSniper, "instance_exists", lambda self, name, zone: True)

    assert sniper.create_vm("europe-west1-c", target) is True
    assert sniper.stop_event.is_set()
