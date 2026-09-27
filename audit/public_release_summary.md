# Public release summary

The single-page account of what this release contains, what it deliberately omits, and
what a reader should and should not conclude from it. The four companion audits
(`public_release_audit.md`, `data_leakage_audit.md`, `path_and_secret_audit.md`,
`source_mapping.md`) give the evidence; this file gives the account.

## 1. Public files

Every file below is intended content. There is no generated artefact, no cache, no
local configuration and no scratch file in the tree.

| Path | What it is |
|---|---|
| `README.md` | Method, layout, installation, per-stage commands, data availability, authors, funding, licence |
| `LICENSE` | The MIT License, with the four copyright holders |
| `CITATION.cff` | Citation metadata, cff-version 1.2.0, authors in manuscript order, the repository URL, and `license: MIT` |
| `CHANGELOG.md` | Version history and the open items |
| `.gitignore` | Exclusion rules, written so a careless `git add .` cannot commit restricted material |
| `requirements.txt` | Pinned runtime dependencies |
| `environment.yml` | Conda environment with the same pins |
| `pyproject.toml` | Package metadata, the MIT classifier and project URLs, dependency groups, pytest configuration |
| `.github/workflows/tests.yml` | Continuous integration: the fast tests, the end-to-end tests and the synthetic demonstration on every push and pull request |
| `config/experiment_config.yaml` | Documentary mirror of the frozen settings |
| `config/paths.example.yaml` | Schema of the library-level path overrides, for callers of `src.utils.paths.resolve` |
| `data/README.md` | What the data directory contains and what it must not contain |
| `data/schema/data_dictionary.csv` | One row per documented column: name, type, description, role, encoding, and four usage flags |
| `data/synthetic/README.md` | Provenance and limits of the synthetic table |
| `data/synthetic/synthetic_cases.csv` | The synthetic demonstration dataset |
| `docs/reproducibility.md` | What can and cannot be reproduced, and how |
| `docs/data_access.md` | The schema a case table must satisfy, and the access position |
| `docs/model_description.md` | The method as the code implements it |
| `docs/experimental_protocol.md` | Folds, seeds, cross-fitting, variants, resampling |
| `docs/privacy_and_data_policy.md` | Privacy position and the leakage controls |
| `src/` | The reference implementation: the nine stage packages plus `pipeline.py` |
| `scripts/` | Nine executable entry points, one per stage plus the two generators |
| `tests/` | Six test modules covering aggregation, the item-balanced objective, the metrics, preprocessing, the full pipeline, and the command-line contract, plus shared fixtures |
| `examples/synthetic_demo/` | A runnable demonstration with its own README and expected output |
| `audit/` | These five records |

`.github/workflows/tests.yml` is the only file added after the initial construction. Its
three command lines were executed against this tree before being written down, and its
`set -o pipefail` guard was checked in both directions; the workflow runner itself has
not been exercised, because nothing has been pushed.

The code is the substantive part: preprocessing and schema validation, request-side and
meta feature construction, the eleven experts including the simplex-constrained convex
combination, the item-balanced objective and meta head, claim-structure-driven
aggregation, the evaluation metrics, the bootstrap and permutation machinery with
multiplicity correction, and the TreeSHAP interpretation layer.

## 2. Excluded files

Excluded by decision, not by oversight. Each was considered and left out.

| Material | Reason |
|---|---|
| The judgment corpus | Contains personally identifiable information, and is not publicly accessible from the original source |
| The case-level dataset | Claim and award amounts for identifiable individuals |
| The frozen meta feature matrix and its feature table | Real per-unit values; only its **column names** informed the code |
| Real out-of-fold predictions and any `*_predictions.csv` from the study | Per-case predicted rates |
| `full_results.csv`, `case_level_results.csv` | Per-case results, sometimes with identifiers retained |
| Real SHAP value tables | Per-unit attributions computed on real feature rows |
| Fitted model artefacts, weight caches, TabPFN checkpoints | Derived from real data, and large binaries |
| Experiment logs and local screenshots | Carry absolute paths, run identifiers and sometimes sampled rows |
| Notebooks with committed outputs | Cell outputs can render tables of real values |
| The manuscript and any reviewer correspondence | Unpublished |
| The study repository's per-experiment layout | Superseded code paths alongside live ones; the release is organised by stage instead |

## 3. Synthetic data

`data/synthetic/synthetic_cases.csv` holds 600 synthetic cases in 39 columns, generated
by `scripts/make_synthetic_data.py` from a recorded seed (`20260927`):

- sixteen claim/award pairs covering the full item universe;
- a claim total and an award total;
- request-side context columns, a procedural column, a year column spanning 2022 and
  2023, and an ordinal severity label marked synthetic;
- 3,813 defined case-item units out of 9,600, the rest being structurally absent.

Three properties make it safe to publish.

**It is not derived from the real data.** The generator samples from distributions
declared inside the script. No real record was read, perturbed or summarised in
producing it.

**It is not an anonymised copy.** A perturbed copy keeps the joint distribution of the
real records and can often be matched against them; a parametric draw keeps nothing of
the original.

**It announces itself.** Identifiers are `SYNTH-####`, the region column holds the
literal `SYNTHETIC-REGION`, the severity column is named for being synthetic, and the
file lives under `data/synthetic/`. The tests assert this naming.

The claim probabilities in the generator are **not** estimates of real-world
frequencies. They are set so that even the rare items have enough training units to fit
at the demonstration scale. They should not be read as a description of the study data.

The dataset is provided solely to demonstrate the execution of the public code. It was
not used to generate the empirical results reported in the manuscript.

## 4. Reproducibility

**The method is reproducible from this repository. The manuscript's numbers are not.**

What a reader can do without any restricted data:

- run the full chain end to end on the synthetic dataset, from the case table through
  feature construction, the expert pool, the disagreement block, the seventy-three
  dimensional meta representation, the item-balanced objective, the claim-weighted
  aggregation, and the evaluation, bootstrap, permutation and interpretation stages;
- verify the structural invariants the method depends on: the aggregation weights sum
  to one, the predicted case-level rate lies in the unit interval, the aggregation
  conservation identity holds, the meta representation has seventy-three columns in
  blocks of eleven, thirteen and forty-nine, and each of the sixteen items carries an
  equal share of the meta training weight;
- regenerate the synthetic dataset and the data dictionary from their recorded seeds and
  confirm both are byte-identical to the committed files;
- run the test suite, which exercises the above rather than only the happy path.

What a reader cannot do:

- reproduce the exact numerical results in the manuscript, because the underlying data
  is not distributed;
- verify the manuscript's reported metrics against the code, for the same reason.

The repository therefore supports methodological inspection, implementation reuse, and
execution of the provided synthetic demonstration. It does not support exact empirical
reproduction, and the README says so where a reader will see it.

The expected-output tables under `examples/synthetic_demo/expected_output/` record what
the demonstration produced on the synthetic data. They exist so a reader can tell
whether their own run matches. They are not manuscript results and are not comparable
with any value reported there.

## 5. Data restriction

The original case-level dataset is not publicly distributed because the underlying
court judgment documents contain personally identifiable information and are not
currently publicly accessible from the original source.

The repository provides code, configuration, schema and synthetic examples for
methodological reproducibility, but contains no judgment texts and no real case-level
data.

This is a restriction on distribution, not a claim about what the authors may share.
The data is not needed to inspect or reuse the method, so it is not distributed.

## 6. Known limitations

Stated plainly, because a reader who discovers one of these unannounced would be right
to distrust the rest.

- **The demonstration runs at reduced scale.** Two outer folds and a subset of cases by
  default, not the study protocol. Metrics from it are indicative of the code running,
  not of model quality.
- **The synthetic data is not a sample of the population of interest.** Any relationship
  the models fit on it is an artefact of the generator.
- **Two configurations ship, and only one of them instantiates the method's full
  pool.** The method is defined by the eleven-expert pool and the 73-dimensional meta
  representation: that is the **full paper configuration**, and it is what the empirical
  experiments reported in the manuscript use. The default is the **lightweight
  demonstration**, which omits TabPFN — the one expert whose pretrained weights are not
  distributed, and which is disabled unless `--with-tabpfn` is passed — so it reaches
  ten experts and its meta matrix is seventy-two columns wide. Only block A follows the
  pool size; the thirteen disagreement features and the forty-nine claim-side features
  are identical in both. Every run reports both widths and states which one its own
  matrix uses, and the seventy-three-column identity is verified against the frozen
  pool, so the structure the method is defined by is checked even when the default run
  does not instantiate it. The code, not the demonstration default, defines the method.
- **The severity label in the synthetic data is illustrative.** The method excludes
  post-outcome case attributes of that class from every feature set.
- **Item-level metrics for rare items are noisy by construction**, and the macro
  convention averages over items with at least a minimum number of units rather than
  over all units.
- **The aggregation weights encode the claim structure, not judicial behaviour.** They
  are a definition, not an estimate, and carry no causal content.
- **The disagreement block is a prediction-difficulty signal.** It is not calibrated
  uncertainty and not a causal quantity, and the code says so where the block is built.
- **Determinism is best-effort.** Thread counts and hash seeds are pinned, but
  floating-point reduction order can still differ across hardware and library builds. A
  rerun on the same machine and environment reproduces; a rerun elsewhere should be
  close rather than bit-identical for the tree ensembles.
- **The probe of the study's frozen column order could not be verified against the
  study matrix**, because that matrix is not distributed. The names and their order were
  recovered from it once; a reader cannot re-derive them from this repository alone.

## 7. Manual review items

Decisions that belong to the authors. The first three were open in an earlier draft and
have since been settled by the authors; they are listed with their resolution so that
the record is complete. The remainder are decisions about the manuscript rather than
gaps in the repository.

1. **The licence — settled.** MIT. The text is in `LICENSE` with the four copyright
   holders, `pyproject.toml` declares it and carries the matching classifier,
   `CITATION.cff` states `license: MIT`, and the README's License section states it
   without qualification.
2. **The repository URL — settled.** Recorded as
   `https://github.com/liyuan112531/traffic-accident-support-rate-prediction` in the
   README, `CITATION.cff` (`repository-code`), `pyproject.toml` (`[project.urls]`),
   `CHANGELOG.md` and the documentation. No placeholder for it remains anywhere.
3. **The two run configurations — settled and documented.** The lightweight
   demonstration is 10 experts and a 72-dimensional meta representation with TabPFN
   disabled; the full paper configuration is 11 experts and 73 dimensions with TabPFN
   enabled; the empirical experiments reported in the manuscript use the full
   configuration. States explicitly in the README, the model description, the
   experimental protocol, the reproducibility notes and the demonstration README.
4. **The DOI.** None is asserted, because none has been assigned. If one is minted, it
   belongs in `CITATION.cff` and the README.
5. **Author name and affiliation spelling.** Taken from the manuscript. Worth confirming
   against the submission system before publication.
6. **ORCID identifiers.** Omitted, because none was supplied. Worth adding to
   `CITATION.cff` if any author has one, which requires no change to anything else.
7. **Reconciliation with the final manuscript.** `docs/model_description.md` describes
   the method as implemented. If the manuscript's final wording differs, the two should
   be reconciled — the code was not changed to match any document, and the documents
   were not changed to flatter the code.
8. **Whether to enable TabPFN by default in the demonstration.** Left disabled, because
   it needs a network download and the weights are not distributed. Enabling it would
   make the demonstration match the eleven-expert method but break offline execution.
9. **Whether the synthetic generator's probabilities should be replaced** by anything
   closer to the real frequencies. Left as they are, because the real frequencies are
   not distributable and the current values are chosen for fold viability.
10. **The grant number and funder wording.** Reproduced exactly as supplied
    (`KM202410038001`, General Program of Science and Technology of Beijing Municipal
    Education Commission). Not to be altered.
