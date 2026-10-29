# LearnFlow release: difficulty-signal fusion analyses

Code and aggregate results for the manuscript *Difficulty-Scale Fusion: Representation Sensitivity and
Criterion-Bounded Evaluation* (Zexiao Weng and MinPo Jung, Youngsan University). Exported from commit
`f08f67398eaa5767aa0231bf03d9bcca23c32beb` of the authors' working repository.

## Contents

| Manuscript section | Script | Result in this repository |
|---|---|---|
| Sections 3-4, synthetic validation | `results/code/m1_instability.py`; re-aggregation in `scripts/research/audit_review_claims.py` | `results/code/m1_results.json`; `results/review_20261009/claim_audit.json` (synthetic parts) |
| Section 5.3, assist09 contribution drift; Section 7, experienced difficulty | `results/code/m1_real_data.py` | `results/code/real_data_results.json` |
| Section 5.4, fixed-bank selection sensitivity | `scripts/research/m1_selection_sensitivity.py` | `results/venue_review_20261009/m1_selection_sensitivity.json` |
| Section 6.1, DBE-KT22 teacher-label criterion | `results/code/optimized_experiments.py`, `results/code/dbe_signals.py`, `scripts/research/recheck_o4_cv.py`, `results/m1/run_a1a2_fit.py` | `results/code/o4_fusion_results.json`; `results/review_20261009/o4_cv_recheck.json` (fold level); `results/m1/m1_criteria_results.json` (aggregates) |
| Section 6.2, Junyi response-disjoint IRT criterion | `results/m1/run_a1a2_fit.py`, `results/m1/verify_m1_criteria.py` | `results/m1/m1_criteria_results.json` |
| Sections 6.3 and 6.5, fitted-criterion and numerical stopping sensitivity | `scripts/research/irt_sensitivity.py`, `scripts/research/irt_convergence_audit.py` | `results/completion_20261009/irt_convergence/` |

`supplementary_data/` holds the four files cited as Supplementary Data 1-4 in the manuscript.

## Data

No raw data are included. Obtain each dataset from its provider and place it under `data/` as the scripts
expect (see `DATA.md`). Results marked with a `_public_release` field were redacted: item-level fields were
removed, and the note records the removed fields and the SHA256 of the full file. Running the named script on
the raw data regenerates the full file.

## Environment

Python 3.11 or later; `pip install -r requirements-research.txt`. Scripts are run from the repository root.

## Running the original scripts

`results/code/m1_real_data.py`, `results/code/optimized_experiments.py` and `results/m1/run_a1a2_fit.py` are kept
byte-identical to the versions that produced the stored results. They write their outputs in place, and
`optimized_experiments.py` still names its original absolute root. To rerun them without overwriting the stored
results, use the wrapper, which redirects outputs to a separate folder and substitutes the repository root:

```
python scripts/research/rebuild_source_evidence.py m1_criteria      # Section 6.1-6.2 criteria
python scripts/research/rebuild_source_evidence.py m1_windows       # Section 7 and assist09 drift
python scripts/research/rebuild_source_evidence.py m1_o1            # earlier-specification drift
python scripts/research/rebuild_source_evidence.py m1_exploratory   # DBE-KT22 fusion and XES3G5M check
```

`scripts/research/irt_convergence_audit.py` reads a local response array produced by
`scripts/research/irt_sensitivity.py` from the raw Junyi data; that array is not distributed.

## License

Code: MIT (see `LICENSE`). The datasets keep their own terms; see `DATA.md`.
