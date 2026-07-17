"""Shared timing helper for ``@pytest.mark.perf`` gates.

CI perf gates run on shared runners where OS-scheduling noise is strictly
**additive** -- a preemption only ever makes a call look *slower*, never faster.
The robust statistic for a micro-benchmark regression gate is therefore the
**best of N runs** (the minimum): it estimates the true compute cost, is immune
to transient runner load, and still rises under a genuine regression (which
inflates every run, including the fastest). Median/mean of a sub-millisecond
call, by contrast, get dragged over the gate by a single stray preemption --
the #383 flake was iiwa14 ``max_solutions=1`` (0.23 ms compute) whose *median*
of 20 measured 1.00 ms under runner load and tripped a ``< 1 ms`` gate.

Every absolute perf gate routes through :func:`best_call_ms` so the whole suite
shares one noise-robust definition of "how long does this call take". Existing
thresholds keep all their headroom (``min <= median <= mean`` always), and now
read as "best-case cost must stay under X" -- exactly what a regression gate
wants.
"""

from __future__ import annotations

import time
from collections.abc import Callable


def best_call_ms(fn: Callable[[], object], *, warmup: int = 3, runs: int = 20) -> float:
    """Best-of-``runs`` wall-clock time for ``fn``, in milliseconds.

    ``warmup`` untimed calls first prime import / cache / branch-predictor
    state; the minimum over ``runs`` timed calls is the noise-floor estimate of
    true compute cost. Deterministic single-input benchmarks (the perf gates'
    usage) are unimodal, so the minimum is the genuine floor -- a real compute
    regression shifts the whole distribution, minimum included, while runner
    load only ever inflates the slower samples the minimum discards.

    :param fn: zero-argument callable to time (wrap the call in a lambda).
    :param warmup: untimed priming calls before measurement.
    :param runs: timed calls; the fastest is returned.
    :returns: best observed call time in milliseconds.
    """
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0
