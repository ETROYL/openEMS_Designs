"""CPU capacity detection and validation for optimization resource settings."""
from __future__ import annotations

import os
from pathlib import Path


RESERVED_LOGICAL_CPUS = 1


def detect_cpu_resources() -> dict[str, int | str]:
    """Return CPU capacity visible to this process, respecting Linux affinity."""
    logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    logical = max(1, logical)
    physical_ids: set[tuple[str, str]] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            blocks = cpuinfo.read_text(encoding="utf-8", errors="replace").strip().split("\n\n")
            allowed = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            for block in blocks:
                fields = {}
                for line in block.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        fields[key.strip()] = value.strip()
                processor = fields.get("processor")
                if allowed is not None and processor is not None and processor.isdigit():
                    if int(processor) not in allowed:
                        continue
                package = fields.get("physical id", "0")
                core = fields.get("core id")
                if core is not None:
                    physical_ids.add((package, core))
        except OSError:
            pass
    physical = len(physical_ids) if physical_ids else logical
    physical = max(1, min(physical, logical))
    return {"physical_cores": physical, "logical_threads": logical}


def validate_allocation(workers: int, threads_per_worker: int) -> None:
    """Reject allocations exceeding physical cores or logical CPUs after OS reserve."""
    resources = detect_cpu_resources()
    physical = int(resources["physical_cores"])
    logical = int(resources["logical_threads"])
    if workers < 1:
        raise ValueError("Workers must be at least 1.")
    if threads_per_worker < 1:
        raise ValueError("Threads per worker must be at least 1.")
    if workers > physical:
        raise ValueError(
            f"Workers ({workers}) exceed the available physical CPU cores ({physical})."
        )
    if threads_per_worker > logical:
        raise ValueError(
            f"Threads per worker ({threads_per_worker}) exceed the available logical CPUs ({logical})."
        )
    total = workers * threads_per_worker
    usable_logical = max(0, logical - RESERVED_LOGICAL_CPUS)
    if total > usable_logical:
        raise ValueError(
            f"Total allocation ({workers} × {threads_per_worker} = {total}) exceeds "
            f"the usable logical CPU budget ({usable_logical} of {logical}; "
            f"{RESERVED_LOGICAL_CPUS} reserved for the operating system and background tasks)."
        )
