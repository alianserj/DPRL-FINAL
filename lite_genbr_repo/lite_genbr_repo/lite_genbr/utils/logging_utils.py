# lite_genbr/utils/logging_utils.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

def log(msg: str) -> None:
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

@dataclass
class StageTimer:
    name: str
    start: Optional[float] = None

    def __enter__(self):
        log(f"▶ START: {self.name}")
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = (time.perf_counter() - (self.start or time.perf_counter()))
        if exc is None:
            log(f"✔ DONE:  {self.name}  (took {dt:.2f}s)")
        else:
            log(f"✖ FAIL:  {self.name}  (after {dt:.2f}s)  err={exc}")
        return False
