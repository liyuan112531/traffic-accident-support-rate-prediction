# Data leakage audit

A scan of every data-bearing file in the release for material from which a real case
could be identified or reconstructed.

## Scope

File types scanned: `.csv`, `.xlsx`, `.xls`, `.json`, `.txt`, `.docx`, `.doc`,
`.parquet`, `.pkl`, `.pickle`, `.joblib`.

These are the file types that can carry tabular or textual case material. The release's
remaining files — the code, the configuration, the documentation, the CI workflow
(`.github/workflows/tests.yml`) and the audit records themselves — are non-data files
and are covered by `public_release_audit.md` and `path_and_secret_audit.md`. None of
them holds a data value, and the CI workflow holds nothing but shell commands and pinned
setup steps.

Patterns searched, in both Chinese and English:

| Category | Patterns |
|---|---|
| Party identity | 原告, 被告, 上诉人, 被上诉人, 当事人, plaintiff, defendant, appellant |
| Direct identifiers | 身份证号, 身份证号码, 住址, 家庭住址, 联系电话, 手机号, identity number, ID card, address, phone |
| Court material | 判决书号, 案号, 法院名称, 承办法官, 审判员, 书记员, judgment number, docket |
| Judgment text | any long-form prose field, any field named for text content, any `text`, `content`, `body`, `full_text`, `ocr` column |
| Case-level amounts | any column of per-case claim or award values, any column of derived rates |
| Case-level predictions | any column of per-case or per-case-item predicted values |
| Attribution values | any column of per-unit attribution values |
| Case identifiers | any identifier resembling a study identifier rather than `SYNTH-####` |

## Result

**Pass.** No finding in any released file.

## Inventory of data-bearing files

The release contains these data-bearing files, all of which were scanned:

| File | Rows / size | Contains | Why it is safe |
|---|---|---|---|
| `data/schema/data_dictionary.csv` | 131 rows, one per documented column | column name, type, description, role, encoding, and four boolean usage flags | Describes structure. No data value, no identifier, no amount, no textual material. The `description` field is a fixed phrase written for the dictionary, not extracted from a record. Regenerable byte-for-byte from `scripts/make_data_dictionary.py`. |
| `data/synthetic/synthetic_cases.csv` | 600 cases, 39 columns | synthetic `case_id`, synthetic context columns, 16 claim/award pairs, two totals | Generated from parametric distributions in `scripts/make_synthetic_data.py` with a recorded seed. Not derived from any real record. |
| `examples/synthetic_demo/expected_output/unit_level_metrics.csv` | one row per variant | aggregate metrics | Aggregate over synthetic rows only. Six numbers per variant, no per-case value. |
| `examples/synthetic_demo/expected_output/aggregated_metrics.csv` | one row per variant | aggregate case-level metrics | Same. |
| `examples/synthetic_demo/expected_output/item_level_metrics.csv` | one row per (variant, item) | per-item aggregate metrics | Aggregate over synthetic rows only. |
| `examples/synthetic_demo/expected_output/objective_alignment.csv` | one row per item | unit counts and weight ratios | Structural quantities. Unit counts refer to synthetic rows. |
| `examples/synthetic_demo/expected_output/run_diagnostics.json` | one object | run settings, structure check, weight invariants, conservation identity, concentration, inference results, verification block | Settings and aggregate diagnostics. No per-case value, no identifier. Holds the meta feature **names** in one place, which are column names, not values. |

No `.xlsx`, `.xls`, `.docx`, `.doc`, `.parquet`, `.pkl`, `.pickle` or `.joblib` file is
present.

`requirements.txt` is the only other file the scan classifies as potentially
data-bearing, and it is classified that way by extension alone: it holds pinned version
strings, package names and explanatory comments, and no data value.

## Why the synthetic dataset cannot be traced back

Three independent reasons, each of which would be sufficient on its own.

**It was never derived from the real data.** The generator samples from distributions
declared inside the script. No real record was read, perturbed, resampled or
summarised in producing it. There is therefore no record pair whose nearest-neighbour
distance could be computed.

**It is not an anonymised copy.** The distinction matters. A perturbed copy retains the
joint distribution of the real records, including their dependence structure, so a
sufficiently determined attacker can often match records across releases. A parametric
draw retains nothing of the original.

**It announces itself.** Case identifiers are `SYNTH-0001` onwards, the region column
holds the literal `SYNTHETIC-REGION`, the graded severity column is named
`injury_severity_synthetic`, and the file lives under `data/synthetic/`.
`tests/test_synthetic_pipeline.py` asserts the naming, so a synthetic row cannot be
mistaken for a real one even by an automated consumer.

## Structural checks on the synthetic dataset

Beyond the pattern scan, the following were checked because they are the ways a
synthetic-looking file can still carry real information.

| Check | Result |
|---|---|
| Any column named for text content | None |
| Any column whose values are long free text | None; the longest field is a fixed category label |
| Any value resembling a real identifier format | None |
| Any exactly-duplicated row with a real record | Not applicable; no real record is present for comparison, and the generator was not seeded from one |
| Award never exceeding its claim | Asserted by `tests/test_synthetic_pipeline.py` |
| Any claim or award that is negative or non-finite | None; asserted on load |
| Any case with a zero claim total | None; asserted to produce an excluded case rather than a zeroed one |

## Files excluded from the release

The following were considered and deliberately excluded. Each is a file type that
would carry case-level material.

| Material | Why excluded |
|---|---|
| The case-level dataset (table of claim and award amounts per case) | Case-level amounts for identifiable individuals; the core restricted artefact |
| The judgment corpus | Contains personal information and is not publicly accessible from the source |
| The frozen meta feature matrix | One row per real case-item unit with real values |
| The frozen meta feature table used to recover the column order | Its **column names** informed `src/features/meta_features.py`; its **values** are real and it is not distributed. `source_mapping.md` records this. |
| Real out-of-fold predictions | Per-case predicted rates |
| `test_predictions.csv`, `case_predictions.csv`, `unit_predictions.csv` from the study runs | Per-case predicted rates |
| `full_results.csv`, `case_level_results.csv` from the study runs | Per-case results, sometimes with identifiers retained |
| Real SHAP value tables | Per-unit attribution values, computed on real feature rows |
| Real feature matrices | Real request-side values per case |
| Experiment logs | Contain absolute paths, run identifiers and sometimes sampled rows |
| Local screenshots | May show file paths, numeric results, or case content |
| Fitted model artefacts | Weights derived from real data; also large binaries |
| Notebooks with committed outputs | Cell outputs can contain rendered tables of real values |

`.gitignore` blocks each of these categories by file type or directory name, so the
exclusion survives a careless `git add .`.

## A note on scope

This audit establishes that no released file contains material from which a real case
could be identified. It does not establish that the study dataset itself is safe to
release, and it does not evaluate whether the data was lawfully held. Those are
questions for the authors and, if a release were ever contemplated, for the source of
the judgments. The position taken here is the straightforward one: the data is not
needed to inspect or reuse the method, so it is not distributed.
