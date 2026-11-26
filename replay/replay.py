"""Reproducible decision replay with an explicit data provenance field."""
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
SLUG = CONFIG["slug"]
FIELDS = ("period", "group", "scenario", "baseline", "candidate", "actual")


def dispatch(prices, forecast, goal=6, limit=2, efficiency=.88):
    """Compare earliest charging and rolling replanning with integer charge units."""
    def plan(signals, required):
        states = {0: (0.0, [])}
        for price in signals:
            next_states = {}
            for charge, (cost, actions) in states.items():
                for step in range(min(limit, required - charge) + 1):
                    state = charge + step
                    proposal = (cost + step * price / efficiency + .035 * step, actions + [step])
                    if state not in next_states or proposal[0] < next_states[state][0]:
                        next_states[state] = proposal
            states = next_states
        return states[required][1]

    remaining, actual_cost, baseline_cost = goal, 0., 0.
    for hour, price in enumerate(prices):
        if remaining:
            # Current price is observed. Future values are taken from the forecast.
            step = plan([price, *forecast[hour + 1:]], remaining)[0]
            remaining -= step
            actual_cost += step * price / efficiency + .035 * step
        baseline_step = min(limit, max(0, goal - hour * limit))
        baseline_cost += baseline_step * price / efficiency + .035 * baseline_step
    if remaining:
        raise ValueError("Dispatch did not meet the departure target")
    return baseline_cost, actual_cost


def fixture():
    rng = random.Random(26)
    for i in range(84):
        group = CONFIG["groups"][i % 3]
        scenario = ("Typical", CONFIG["stress"], "Feed degraded")[(i // 3) % 3]
        degraded = scenario == "Feed degraded"
        signal = rng.uniform(0, 1)
        if SLUG == "voltflex-uk":
            prices = [rng.uniform(.04, .42) for _ in range(8)]
            if scenario == CONFIG["stress"]:
                prices[4] = -.06
            forecast = [p + rng.uniform(-.11, .11) for p in prices]
            if degraded:
                forecast = [.21] * 8
            baseline, candidate = dispatch(prices, forecast)
            actual = 0.0
        elif SLUG == "gridguard-uk":
            # Compare rankings against a hidden, simulated benefit for matched budget units.
            baseline = 300 + 200 * signal
            actual = baseline + 150 * signal + rng.gauss(0, 65)
            candidate = baseline if degraded else baseline + 155 * signal
        elif SLUG == "railpulse-uk":
            baseline = 3 + 20 * signal
            actual = max(0, baseline * .72 + rng.gauss(0, 2.8) + (5 if scenario == CONFIG["stress"] else 0))
            candidate = baseline if degraded else max(0, .72 * baseline + (4 if scenario == CONFIG["stress"] else 0))
        elif SLUG == "aquasentinel-uk":
            event_chance = min(.96, .11 + .58 * signal + (.22 if scenario == CONFIG["stress"] else 0))
            actual = float(rng.random() < event_chance)
            baseline = .7 if signal > .67 else .25
            candidate = baseline if degraded else event_chance
        elif SLUG == "harvestshield-uk":
            baseline = 7.1 - 2.0 * signal
            canopy = rng.uniform(-.9, .9)
            actual = baseline + canopy + rng.gauss(0, .32)
            candidate = baseline if degraded else baseline + .85 * canopy
        else:
            baseline = 20 + 24 * signal
            cold = 8 if scenario == CONFIG["stress"] else 0
            actual = min(100, baseline + cold + rng.gauss(0, 4.5))
            candidate = baseline if degraded else min(100, baseline + cold)
        yield (i + 1, group, scenario, baseline, candidate, actual)


def rows_from_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        source = csv.DictReader(file)
        if set(source.fieldnames or ()) != set(FIELDS):
            raise ValueError("Expected CSV columns " + ", ".join(FIELDS))
        for line, row in enumerate(source, 2):
            try:
                number = int(row["period"])
                values = [float(row[key]) for key in FIELDS[3:]]
                if number < 1 or not row["group"].strip() or not row["scenario"].strip() or not all(map(math.isfinite, values)):
                    raise ValueError
                if SLUG == "aquasentinel-uk" and (
                    values[2] not in (0, 1) or any(not 0 <= value <= 1 for value in values[:2])
                ):
                    raise ValueError
                yield (number, row["group"], row["scenario"], *values)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid replay record on CSV line {line}") from error


def report(rows, provenance):
    cases = []
    seen = set()
    for period, group, scenario, baseline, candidate, actual in rows:
        key = (period, group)
        if key in seen:
            raise ValueError(f"Duplicate replay case {period} for {group}")
        seen.add(key)
        if SLUG == "voltflex-uk":
            first, second = baseline, candidate
        elif SLUG == "aquasentinel-uk":
            first, second = (actual - baseline) ** 2, (actual - candidate) ** 2
        else:
            first, second = abs(actual - baseline), abs(actual - candidate)
        cases.append(dict(period=period, group=group, scenario=scenario, baseline=round(first, 5),
                          candidate=round(second, 5)))
    if not cases:
        raise ValueError("The replay contains no records")
    cases.sort(key=lambda case: (case["period"], case["group"]))
    return {"config": CONFIG, "provenance": provenance, "cases": cases,
            "baseline_mean": statistics.mean(case["baseline"] for case in cases),
            "candidate_mean": statistics.mean(case["candidate"] for case in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="CSV with precomputed, out-of-sample scores or predictions")
    parser.add_argument("--source", help="Provenance for external input")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.input and not (options.source or "").strip():
        parser.error("--source is required with external data")
    try:
        rows = rows_from_csv(options.input) if options.input else fixture()
        result = report(rows, options.source or "Deterministic synthetic scenarios")
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Replay failed: {error}\n")


if __name__ == "__main__":
    main()
