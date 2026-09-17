"""TensorBoard plus a plain-text log, written by rank 0 only."""

from __future__ import annotations

import json
import time
from pathlib import Path


class Tracker:
    def __init__(self, output_dir: str | Path, enabled: bool) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.writer = None
        self._log = None
        if not enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(self.output_dir / "tb"))
        self._log = open(self.output_dir / "log.txt", "a", buffering=1)

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def scalars(self, values: dict[str, float], step: int) -> None:
        for tag, value in values.items():
            self.scalar(tag, value, step)

    def text(self, message: str) -> None:
        if not self.enabled:
            return
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        self._log.write(line + "\n")

    def dump_json(self, name: str, payload: dict) -> None:
        if self.enabled:
            (self.output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self._log is not None:
            self._log.close()
