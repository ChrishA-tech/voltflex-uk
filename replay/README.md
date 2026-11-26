# VoltFlex UK decision replay

[Open the interactive portfolio dashboard](https://chrisha-tech.github.io/replay/?project=voltflex-uk).

This charging replay compares two policies with the same departure target and import cap. It includes conversion losses and a wear charge. Prices and forecast errors are simulated. The actor–critic and savings from the source project are outside this benchmark.

## Reproduce

From the repository root, use Python 3.10+:

```sh
python replay/replay.py --output replay/example.json
python -m unittest discover -s replay -v
```

The committed HTML is a portable snapshot of the bundled report. Open `replay/index.html` directly. The dashboard supports group and scenario filters, case-level scores and CSV export.

## Error analysis on the bundled scenarios

Scores are in **£ per replay**. Lower is better. Candidate regressions count cases where its score is higher.

| Scenario | Cases | Baseline mean | Candidate mean | Candidate regressions |
| --- | ---: | ---: | ---: | ---: |
| Typical | 30 | 1.671 | 1.047 | 0 |
| Negative prices | 27 | 1.696 | 0.607 | 0 |
| Feed degraded | 27 | 1.813 | 1.111 | 1 |

The largest regression was 0.060 at case 80 (London, Feed degraded). The scenarios test software behaviour. They cannot establish real-world accuracy or savings.

## Replay an external archive

Supply a CSV with columns `period,group,scenario,baseline,candidate,actual`. Sort and align predictions and outcomes upstream. Each period/group pair must be unique. Supply genuinely out-of-sample predictions computed from information available at the decision time. The CLI validates input and scores cases. It does not train or backtest the source model for you.

For cost replay, put the realised policy costs in baseline and candidate, and a numeric placeholder in actual. For spill replay, use probabilities in [0,1] and binary outcomes. Other projects use predictions and observed numeric outcomes in the units above.

```sh
python replay/replay.py --input archive.csv --source "Archive name and holdout period" --output report.json
```
