# HotpotQA 500 Offline A/B/C Analysis

The two canonical result files contain exactly 1,500 complete, unique question/strategy records each.
Both models use the same 500 manifest question IDs. Historical incomplete attempts remain isolated
outside the canonical JSONL files and are not included in any aggregate.

## Gemma 12B

| Strategy | Mean F1 | Mean EM | Retrieval calls | API requests | Tokens | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.280700 | 0.202000 | 0.000000 | 1.000000 | 1542.488 | 2.028940 |
| B | 0.373320 | 0.288000 | 1.000000 | 1.000000 | 3148.700 | 2.147609 |
| C | 0.462800 | 0.348000 | 9.590000 | 10.590000 | 32559.476 | 17.958399 |

Optimal-set counts (percent): A 61 (12.2%), B 41 (8.2%), C 104 (20.8%), AB 9 (1.8%), AC 15 (3.0%), BC 88 (17.6%), ABC 182 (36.4%)
All-failed subset of ABC: 118 (23.6%)

## Gemma 31B

| Strategy | Mean F1 | Mean EM | Retrieval calls | API requests | Tokens | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.363300 | 0.250000 | 0.000000 | 1.000000 | 1536.638 | 6.200123 |
| B | 0.397620 | 0.310000 | 1.000000 | 1.000000 | 3153.736 | 6.673711 |
| C | 0.564420 | 0.440000 | 7.442000 | 8.442000 | 26133.198 | 49.146515 |

Optimal-set counts (percent): A 65 (13.0%), B 14 (2.8%), C 93 (18.6%), AB 9 (1.8%), AC 25 (5.0%), BC 107 (21.4%), ABC 187 (37.4%)
All-failed subset of ABC: 101 (20.2%)

