# infer_stage_weights.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


Number = Union[int, float]


@dataclass(frozen=True)
class StageWeights:
    """Return object for inferred stage weights."""
    stage_to_weight: Dict[str, float]
    ordered_stages: List[str]
    unit: str
    scale01: bool
    origin: str


def _natural_sort_key(s: str) -> Tuple:
    """
    Natural-ish sort key that keeps numeric chunks ordered numerically.
    Example: ["E6.5", "E10", "E8"] -> E6.5, E8, E10
    """
    parts = re.split(r"(\d+(?:\.\d+)?)", str(s))
    key: List[Tuple[int, Union[str, float]]] = []
    for p in parts:
        if p == "":
            continue
        try:
            key.append((1, float(p)))
        except ValueError:
            key.append((0, p.lower()))
    return tuple(key)


def _parse_embryo_day(stage: str) -> Optional[float]:
    """
    Parse embryo day from common labels:
      - "E6.5", "e8", "E10.0"
      - ranges: "E8.0-E8.5", "E8.0 to E8.5"  -> mean(8.0, 8.5)
      - bare numeric: "6.5" -> 6.5
    Returns None if cannot parse.
    """
    if stage is None:
        return None
    x = str(stage).strip().lower()

    # Capture E-values anywhere in the string, including ranges.
    # Examples matched: "e6", "e6.5", "e 8.0", "E8.0-E8.5"
    e_nums = re.findall(r"e\s*(\d+(?:\.\d+)?)", x)
    if e_nums:
        vals = [float(v) for v in e_nums]
        return sum(vals) / len(vals)

    # Fallback: if the whole stage looks numeric already
    if re.fullmatch(r"\d+(?:\.\d+)?", x):
        return float(x)

    return None


def infer_stage_weights(
    stages: Sequence[str],
    unit: str = "hours",
    scale01: bool = False,
    origin: str = "min",
    explicit_order: Optional[Sequence[str]] = None,
) -> StageWeights:
    """
    Infer a numeric weight w_s for each unique stage label.

    Parameters
    ----------
    stages:
        Iterable/sequence of stage labels from adata.obs['stage'] (strings).
    unit:
        One of {"hours", "days", "rank"}.
        - "hours": interpret stage as embryo day (E6.5 etc.) and convert to hours.
        - "days": interpret as embryo day and keep in days.
        - "rank": ignore numeric times; assign 0..m in chronological order of labels.
    scale01:
        If True, min-max scale weights to [0, 1] after origin/unit handling.
    origin:
        One of {"min", "absolute"}.
        - "min": subtract the minimum (earliest) value so the first stage is 0.
        - "absolute": keep absolute parsed values (e.g., E6.5 -> 6.5 days or 156 hours).
        (Note: for rank mode, origin is irrelevant.)
    explicit_order:
        Optional list giving the desired chronological order of stage labels.
        If provided, this order is used and parsing/sorting is skipped.

    Returns
    -------
    StageWeights:
        stage_to_weight: dict mapping stage label -> inferred w_s
        ordered_stages: list of stages in chronological order used for assignment
        unit, scale01, origin: echoes of settings

    Notes
    -----
    Recommended for developmental datasets:
      unit="hours", origin="min"  -> hours since earliest stage
    For categorical stages without parseable times:
      unit="rank"
    """
    unit = unit.lower()
    if unit not in {"hours", "days", "rank"}:
        raise ValueError("unit must be one of {'hours', 'days', 'rank'}")
    origin = origin.lower()
    if origin not in {"min", "absolute"}:
        raise ValueError("origin must be one of {'min', 'absolute'}")

    # Collect unique non-null labels
    uniq = []
    seen = set()
    for s in stages:
        if s is None:
            continue
        ss = str(s)
        if ss.strip() == "":
            continue
        if ss not in seen:
            uniq.append(ss)
            seen.add(ss)

    if len(uniq) < 2:
        raise ValueError("Need >= 2 unique non-empty stages to infer weights.")

    # Determine stage order
    if explicit_order is not None:
        ordered = [str(x) for x in explicit_order]
        missing = [x for x in uniq if x not in set(ordered)]
        if missing:
            raise ValueError(
                f"explicit_order is missing stage labels present in data: {missing}"
            )
    else:
        ordered = sorted(uniq, key=_natural_sort_key)

    # Try time-based weights if requested
    parsed_days: Dict[str, Optional[float]] = {s: _parse_embryo_day(s) for s in ordered}
    any_missing = any(parsed_days[s] is None for s in ordered)

    if unit == "rank" or any_missing:
        # Rank-based assignment: 0,1,2,... following `ordered`
        weights = {s: float(i) for i, s in enumerate(ordered)}
        used_unit = "rank" if unit == "rank" else "rank(fallback)"
    else:
        vals = [float(parsed_days[s]) for s in ordered]  # type: ignore[arg-type]
        if origin == "min":
            v0 = min(vals)
            vals = [v - v0 for v in vals]

        if unit == "hours":
            vals = [v * 24.0 for v in vals]
        # unit == "days" uses vals as-is

        weights = {s: float(v) for s, v in zip(ordered, vals)}
        used_unit = unit

    # Optional scaling to [0, 1]
    if scale01:
        ws = list(weights.values())
        wmin, wmax = min(ws), max(ws)
        if wmax == wmin:
            weights = {s: 0.0 for s in ordered}
        else:
            weights = {s: (weights[s] - wmin) / (wmax - wmin) for s in ordered}

    return StageWeights(
        stage_to_weight=weights,
        ordered_stages=ordered,
        unit=used_unit,
        scale01=scale01,
        origin=origin,
    )


if __name__ == "__main__":
    # Minimal example
    demo = ["E6.0", "E6.5", "E8.0-E8.5", "E10", "E6.0", None]
    sw = infer_stage_weights(demo, unit="hours", origin="min", scale01=False)
    print("Ordered:", sw.ordered_stages)
    print("Unit:", sw.unit)
    print("Weights:")
    for k in sw.ordered_stages:
        print(f"  {k:12s} -> {sw.stage_to_weight[k]:.2f}")
