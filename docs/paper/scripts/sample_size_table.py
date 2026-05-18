"""Compute planning sample sizes for fixed-threshold FPR estimates.

For target FPR p and relative half-width r, the paper uses the 95% rule

    n >= 4 / (r^2 * p).

This script prints the rounded values used in the paper's sample-size table.
"""

from __future__ import annotations

import math


def planning_n(p: float, r: float) -> int:
    """Return the normal-approximation planning sample size."""
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if not 0 < r < 1:
        raise ValueError(f"r must be in (0, 1), got {r}")
    return math.ceil(4.0 / ((r**2) * p))


def format_n(n: int) -> str:
    """Format large counts compactly for the paper table."""
    if n < 100:
        return str(n)
    if n < 1_000:
        return f"{round(n, -1):,.0f}"
    if n < 10_000:
        return f"{round(n / 100) * 100:,}"
    if n < 1_000_000:
        return f"{round(n / 1000) * 1000:,}"
    millions = n / 1_000_000
    if millions < 10:
        return f"{millions:.1f}M"
    if millions < 100:
        return f"{millions:.2g}M"
    return f"{round(millions):.0f}M"


def main() -> None:
    fprs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    relative_widths = [0.50, 0.25, 0.10]

    print("# Sample sizes for FPR estimation (95% planning rule)\n")
    header = "| Target FPR | " + " | ".join(f"r={int(r * 100)}%" for r in relative_widths) + " |"
    print(header)
    print("|" + "---|" * (len(relative_widths) + 1))

    for p in fprs:
        cells = [format_n(planning_n(p, r)) for r in relative_widths]
        exponent = int(round(math.log10(p)))
        print(f"| $10^{{{exponent}}}$ | " + " | ".join(cells) + " |")

    print("\nLaTeX rows:\n")
    for p in fprs:
        exponent = int(round(math.log10(p)))
        cells = [format_n(planning_n(p, r)).replace(",", "{,}") for r in relative_widths]
        print(f"    $10^{{{exponent}}}$ & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
