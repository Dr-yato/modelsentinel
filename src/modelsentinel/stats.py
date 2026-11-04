"""Fisher's exact test, chi-square, and Benjamini-Hochberg.

The primary test for a trigger cell is a one-sided Fisher exact test on the
2x2 table of insecure vs not, baseline vs triggered. Chi-square with Yates
correction is reported when every expected count is at least 5. q-values use
Benjamini-Hochberg across every cell in the run.
"""

from __future__ import annotations

import math


def log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_greater(baseline_insecure: int, baseline_n: int, triggered_insecure: int, triggered_n: int) -> float:
    """One-sided p-value for the triggered arm having at least this many insecure runs."""
    if baseline_n <= 0 or triggered_n <= 0:
        return 1.0
    if not (0 <= baseline_insecure <= baseline_n and 0 <= triggered_insecure <= triggered_n):
        return 1.0
    row_base = baseline_n
    row_trig = triggered_n
    col_insecure = baseline_insecure + triggered_insecure
    total = row_base + row_trig
    k_min = max(0, col_insecure - row_base)
    k_max = min(row_trig, col_insecure)
    logs = []
    for k in range(k_min, k_max + 1):
        logs.append(log_choose(row_trig, k) + log_choose(row_base, col_insecure - k) - log_choose(total, col_insecure))
    if not logs:
        return 1.0
    peak = max(logs)
    denom = sum(math.exp(item - peak) for item in logs)
    numer = sum(math.exp(item - peak) for k, item in zip(range(k_min, k_max + 1), logs) if k >= triggered_insecure)
    if denom == 0:
        return 1.0
    return max(0.0, min(1.0, numer / denom))


def chi_square(baseline_insecure: int, baseline_n: int, triggered_insecure: int, triggered_n: int) -> float | None:
    a = baseline_insecure
    b = baseline_n - baseline_insecure
    c = triggered_insecure
    d = triggered_n - triggered_insecure
    total = a + b + c + d
    if total == 0:
        return None
    cells = [a, b, c, d]
    rows = [a + b, c + d]
    cols = [a + c, b + d]
    expected = []
    for index, obs in enumerate(cells):
        row = rows[0] if index < 2 else rows[1]
        col = cols[index % 2]
        expected.append(row * col / total)
    if any(value < 5 for value in expected):
        return None
    stat = 0.0
    for obs, exp in zip(cells, expected):
        stat += (abs(obs - exp) - 0.5) ** 2 / exp
    # Survival function of chi-square with 1 degree of freedom.
    return math.erfc(math.sqrt(stat / 2.0))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda index: p_values[index])
    q_values = [1.0] * m
    running = 1.0
    for rank_from_end, index in enumerate(reversed(order)):
        rank = m - rank_from_end
        adjusted = p_values[index] * m / rank
        running = min(running, adjusted)
        q_values[index] = min(1.0, running)
    return q_values


def risk_difference(baseline_insecure: int, baseline_n: int, triggered_insecure: int, triggered_n: int) -> float:
    base = baseline_insecure / baseline_n if baseline_n else 0.0
    trig = triggered_insecure / triggered_n if triggered_n else 0.0
    return trig - base


def normal_sf(z: float) -> float:
    """Upper-tail probability of a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def cmh_greater(strata: list[tuple[int, int, int, int]]) -> float:
    """One-sided Cochran-Mantel-Haenszel test across tasks.

    Each stratum is ``(baseline_insecure, baseline_n, triggered_insecure,
    triggered_n)`` for one task. The question is whether the triggered arm
    is more insecure *within* tasks, pooled over tasks, so one easy task
    cannot dominate the way it would in a naive pooled 2x2. Uses the
    continuity-corrected normal approximation; strata with no variation
    contribute nothing.
    """
    observed = 0.0
    expected = 0.0
    variance = 0.0
    for base_x, base_n, trig_x, trig_n in strata:
        total = base_n + trig_n
        events = base_x + trig_x
        if total < 2 or events == 0 or events == total:
            continue
        observed += trig_x
        expected += trig_n * events / total
        variance += base_n * trig_n * events * (total - events) / (total * total * (total - 1))
    if variance <= 0:
        return 1.0
    z = (observed - expected - 0.5) / math.sqrt(variance)
    return max(0.0, min(1.0, normal_sf(z)))


def mantel_haenszel_rd(strata: list[tuple[int, int, int, int]]) -> float:
    """Mantel-Haenszel weighted risk difference (triggered minus baseline)."""
    numer = 0.0
    denom = 0.0
    for base_x, base_n, trig_x, trig_n in strata:
        if base_n <= 0 or trig_n <= 0:
            continue
        weight = base_n * trig_n / (base_n + trig_n)
        numer += weight * (trig_x / trig_n - base_x / base_n)
        denom += weight
    return numer / denom if denom else 0.0
