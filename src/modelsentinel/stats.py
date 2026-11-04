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
