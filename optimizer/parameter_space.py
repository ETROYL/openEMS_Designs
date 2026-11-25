"""Exact, start-anchored discrete parameter spaces for YAML antenna designs."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from itertools import product
from math import prod
from typing import Any, Iterable, Mapping, Sequence

class ParameterSpaceError(ValueError):
    """Raised when an optimization variable cannot form a valid grid."""

def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ParameterSpaceError(f"{label} must be a finite number; got {value!r}") from exc
    if not result.is_finite():
        raise ParameterSpaceError(f"{label} must be finite; got {value!r}")
    return result

def _is_integral(value: Decimal) -> bool:
    return value == value.to_integral_value()

@dataclass(frozen=True)
class ParameterDefinition:
    """One optimized YAML variable represented by an integer lattice."""
    name: str
    value_type: str
    start: Decimal
    step: Decimal
    count: int

    @property
    def levels(self) -> int:
        return self.count + 1

    @property
    def last_value(self) -> Decimal:
        return self.start + self.count * self.step

    @property
    def cardinality(self) -> int:
        return self.levels

    def value_at(self, index: int) -> int | float:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ParameterSpaceError(f"Index for {self.name!r} must be an integer")
        if index < 0 or index >= self.levels:
            raise ParameterSpaceError(f"Index {index} for {self.name!r} is outside [0, {self.count}]")
        value = self.start + index * self.step
        return int(value) if self.value_type == "int" else float(value)

    def index_of(self, value: Any) -> int:
        physical = _decimal(value, self.name)
        offset = (physical - self.start) / self.step
        if not _is_integral(offset):
            raise ParameterSpaceError(f"{self.name}={value!r} is not on the {self.step} grid anchored at {self.start}")
        index = int(offset)
        if index < 0 or index >= self.levels:
            raise ParameterSpaceError(f"{self.name}={value!r} is outside the grid [{self.start}, {self.last_value}]")
        return index

class DiscreteParameterSpace:
    """Convert YAML optimization variables to and from an integer-grid space.

    Grid points are start + k * step for integer k >= 0. If stop is not
    exactly reachable, the final legal point is the largest grid point below
    stop. A YAML default is a CAD-preview default, not an optimization bound;
    it may intentionally lie outside [start, stop].
    """
    def __init__(self, variables: Mapping[str, Any]):
        if not isinstance(variables, Mapping):
            raise ParameterSpaceError("'variables' must be a mapping")
        self._raw_variables = dict(variables)
        self.parameters: list[ParameterDefinition] = []
        self.fixed_values: dict[str, Any] = {}
        for name, options in self._raw_variables.items():
            if not isinstance(options, Mapping):
                self.fixed_values[name] = options
                continue
            if options.get("optimize", True) is False:
                self.fixed_values[name] = options.get("default", options.get("start"))
                continue
            if "start" not in options or "stop" not in options:
                raise ParameterSpaceError(f"Optimized variable {name!r} must define both start and stop")
            value_type = str(options.get("type", "float")).lower()
            if value_type not in {"float", "int"}:
                raise ParameterSpaceError(f"Variable {name!r} has unsupported type {value_type!r}; use float or int")
            start = _decimal(options["start"], f"{name}.start")
            stop = _decimal(options["stop"], f"{name}.stop")
            if stop < start:
                raise ParameterSpaceError(f"{name}.stop must be >= {name}.start")
            if value_type == "int":
                step = _decimal(options.get("step", 1), f"{name}.step")
                if not all(_is_integral(v) for v in (start, stop, step)):
                    raise ParameterSpaceError(f"Integer variable {name!r} requires integral start, stop and step")
            else:
                if "step" not in options:
                    raise ParameterSpaceError(f"Optimized float variable {name!r} must define a positive step for discrete optimization")
                step = _decimal(options["step"], f"{name}.step")
            if step <= 0:
                raise ParameterSpaceError(f"{name}.step must be greater than zero")
            count = int(((stop - start) / step).to_integral_value(rounding=ROUND_FLOOR))
            if count == 0:
                self.fixed_values[name] = int(start) if value_type == "int" else float(start)
            else:
                self.parameters.append(ParameterDefinition(name, value_type, start, step, count))
        self._by_name = {p.name: p for p in self.parameters}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DiscreteParameterSpace":
        if not isinstance(config, Mapping):
            raise ParameterSpaceError("Configuration must be a mapping")
        return cls(config.get("variables", {}))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def dimension(self) -> int:
        return len(self.parameters)

    @property
    def cardinality(self) -> int:
        return prod(p.cardinality for p in self.parameters) if self.parameters else 1

    def levels(self) -> tuple[int, ...]:
        return tuple(p.levels for p in self.parameters)

    def to_values(self, indices: Sequence[int], include_fixed: bool = True) -> dict[str, Any]:
        if len(indices) != self.dimension:
            raise ParameterSpaceError(f"Expected {self.dimension} grid indices, received {len(indices)}")
        values = dict(self.fixed_values) if include_fixed else {}
        for parameter, index in zip(self.parameters, indices):
            values[parameter.name] = parameter.value_at(index)
        return values

    def to_indices(self, values: Mapping[str, Any]) -> tuple[int, ...]:
        missing = [name for name in self.names if name not in values]
        if missing:
            raise ParameterSpaceError(f"Missing optimized parameter values: {missing}")
        return tuple(p.index_of(values[p.name]) for p in self.parameters)

    def iter_indices(self) -> Iterable[tuple[int, ...]]:
        """Iterate over the complete grid; only use when cardinality is manageable."""
        return product(*(range(p.levels) for p in self.parameters))

    def describe(self) -> list[dict[str, Any]]:
        """Return JSON-friendly parameter metadata for logs and dashboards."""
        return [{
            "name": p.name, "type": p.value_type,
            "start": int(p.start) if p.value_type == "int" else float(p.start),
            "step": int(p.step) if p.value_type == "int" else float(p.step),
            "stop_requested": float(self._raw_variables[p.name]["stop"]),
            "last_grid_value": p.value_at(p.count), "levels": p.levels,
        } for p in self.parameters]
