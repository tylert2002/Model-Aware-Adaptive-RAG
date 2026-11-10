# HotpotQA 500 A/B/C and Retrospective Oracle Analysis

All values below are derived offline from the validated paired A/B/C records. The Oracle uses observed
ground-truth F1 and is an analytical upper bound, not a deployable router.

## Gemma 12B

- Best fixed strategy: C (mean F1 46.28%)
- Retrospective Oracle mean F1: 59.37%
- Oracle Gap: 13.09 percentage points
- Oracle selections: A 267 (53.4%), B 129 (25.8%), C 104 (20.8%)
- Highest-F1 ties: 294 (58.8%)
- Oracle strictly exceeds the globally best fixed strategy: 111 (22.2%)

## Gemma 31B

- Best fixed strategy: C (mean F1 56.44%)
- Retrospective Oracle mean F1: 67.11%
- Oracle Gap: 10.66 percentage points
- Oracle selections: A 286 (57.2%), B 121 (24.2%), C 93 (18.6%)
- Highest-F1 ties: 328 (65.6%)
- Oracle strictly exceeds the globally best fixed strategy: 88 (17.6%)

