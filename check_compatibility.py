"""Check that the included checkpoints match the included runtime model."""

from __future__ import annotations

from pathlib import Path

import torch

from plate_model import load_gru_checkpoint


def main() -> None:
    root = Path(__file__).resolve().parent
    device = torch.device("cpu")
    for checkpoint in sorted((root / "models").glob("*.pth")):
        try:
            _, report = load_gru_checkpoint(str(checkpoint), device)
            print(f"[OK] {checkpoint.name}: {report}")
        except Exception as exc:
            print(f"[MISMATCH] {checkpoint.name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
