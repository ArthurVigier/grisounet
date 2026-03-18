"""CLI entrypoint for the Compute Engine GPU sniper."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_logic.gpu_sniper import main


if __name__ == "__main__":
    raise SystemExit(main())
