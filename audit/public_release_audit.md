# Public release audit

A checklist of the conditions this release must satisfy, with the evidence for each.
Every item was verified against the released tree as it stands.

Scan performed with an automated pass over every text file in the release, matching:

- absolute Windows drive paths, POSIX home paths, Windows user-profile paths, UNC and
  network paths;
- account and identifier patterns, email addresses, IPv4 addresses;
- secret-like assignments (`api_key`, `secret`, `password`, `token`, `access_key`,
  `private_key`, `client_secret`), bearer and Hugging Face tokens, and database
  connection strings;
- Chinese personal-information field labels, judgment numbers;
- internal experiment codes;
- hard-coded decimal results next to metric names;
- the five forbidden data-availability phrasings and two forbidden strong
  reproducibility claims;
- the presence of the required data-policy, synthetic-data, bounded-reproducibility,
  licence, and funding wordings;
- unfilled template placeholders: bracket-marked confirmation, completion and
  repository-URL slots, the all-caps and hyphenated username templates, the reserved
  documentation domain, an unassigned DOI written in DOI form, and the bare task markers
  an unfinished draft leaves behind. The patterns are described here rather than written
  out, on the same reasoning as item 9: an audit that reproduces the strings it certifies
  absent would falsify its own claim.
- The angle-bracket notation used throughout this release — `<item>_claim`,
  `<item>_award`, `config/<name>` — is deliberately **not** treated as a placeholder. It
  is the repository's naming convention for a family of columns and for a function
  parameter, and it is what makes the schema readable. A template placeholder is a slot a
  reader is expected to fill in before use; these are names, and they are already correct.

The scan is run against the final tree with all scratch directories removed. Its
results are reproduced in `data_leakage_audit.md` and `path_and_secret_audit.md`.

## 1. No raw judgment documents

- **Result: pass.** The release contains no `.docx`, `.doc`, `.wps` or `.rtf` file.
  The scan reports an empty document list.
- `.gitignore` excludes `*.docx`, `*.doc`, `*.wps` and `*.rtf` outright, so a judgment
  document placed anywhere in the tree cannot be committed by accident.
- No excerpt, quotation, paraphrase or screenshot of a judgment appears in any
  Markdown file. The documents are referred to generically and no case is described.

## 2. No real case-level dataset

- **Result: pass.** The only CSV files present are:
  - `data/schema/data_dictionary.csv` — the column dictionary, containing no data
    values;
  - `data/synthetic/synthetic_cases.csv` — the synthetic dataset;
  - `examples/synthetic_demo/expected_output/*.csv` — tables computed from the
    synthetic dataset.

  There is no file holding claim or award amounts for a real case, no real feature
  matrix, no real target vector and no real prediction vector.
- `.gitignore` excludes `data/**/*.csv` and re-includes only the two files above by
  name, so a real case table would have to be given one of those two names to be
  tracked, which is a deliberate and visible act.

## 3. No personally identifiable information

- **Result: pass.** The scan finds no name, identity number, address, telephone
  number, judgment number, or personal-information field label outside the scanner
  patterns themselves, which are not part of the release.
- No email address and no telephone number appears. `CITATION.cff` and the README
  record author names and roles, which are public bibliographic information and appear
  in the manuscript, and direct correspondence to the manuscript rather than
  reproducing contact details.
- The synthetic dataset's only identifier column holds values of the form
  `SYNTH-0001`, its region column holds the literal `SYNTHETIC-REGION`, and the file is
  documented as synthetic in its own README and in the data dictionary.

## 4. No real case identifiers

- **Result: pass.** Every identifier in the release is either synthetic (`SYNTH-####`)
  or a column name. No study identifier, docket number or hash of one appears.

## 5. No absolute local paths

- **Result: pass.** The scan finds no absolute Windows drive path, no POSIX home path,
  no Windows user-profile path and no UNC path in any released file.
- All path resolution goes through `src/utils/paths.py`, which derives the repository
  root from `Path(__file__).resolve().parents[2]`, and through
  `config/paths.example.yaml`, whose values are repository-relative paths and which no
  entry point is wired to.
- The scripts accept `--data` and `--output` and default to repository-relative
  locations, so no command in the documentation depends on where the repository sits.

## 6. No API keys, passwords or tokens

- **Result: pass.** The scan finds no secret-like assignment, no bearer token, no
  Hugging Face token, no database connection string and no private key material.
- No script in the release performs an authenticated network call. The only network
  access any code path can trigger is the optional TabPFN weight download, which is
  disabled by default and needs no credential.
- `.gitignore` excludes `.env` and its variants, `*.pem`, `*.key`, `*.p12`, `*.pfx`,
  `*.jks`, `secrets.yaml`, `credentials*`, `*credential*.json`, `*token*.txt`,
  `.git-credentials` and `.netrc`.

## 7. No private model caches or weights

- **Result: pass.** The scan reports no `.pkl`, `.pickle`, `.joblib`, `.parquet`,
  `.h5`, `.pt`, `.pth`, `.ckpt`, `.onnx`, `.safetensors` or `.bin` file. No
  `model_cache/`, `hub/` or `models--*/` directory is present.
- `.gitignore` excludes all of those file types and directory names, plus
  `tabpfn*/`, `models--*/`, `hub/` and `*.lock`.
- The README states that model weights and caches are not included in the repository,
  and TabPFN is reported as skipped in every default run.

## 8. No hard-coded empirical results

- **Result: pass.** The scan finds no decimal literal adjacent to a metric name.
- No script prints a figure it did not compute. Every number in the output tables is
  derived from the data the run was given. The conservation identity, for example, is
  reported with the deviation the run observed, not with a tolerance.
- The manuscript's reported values are not quoted anywhere in the release. The README
  states explicitly that they are not reproduced, which is the only honest position
  given that the data is not distributed.

## 9. No internal experiment codes in the public surface

- **Result: pass.** The scan finds no code from the study's `E`-series experiment
  numbering in any released file. The pattern is referred to by description rather than
  reproduced in full, so that this audit does not become the one file in the release
  containing it — which would make its own claim false. The
  end-to-end test checks that the codes do not appear among the string constants of the
  pipeline module, and it constructs the codes from a range so that the test does not
  itself introduce them.
- The analyses are referred to by their formal names: core model performance,
  statistical robustness, case-level aggregation validation, temporal extrapolation,
  interpretability analysis.

## 10. No author placeholder

- **Result: pass.** `CITATION.cff`, `LICENSE` and the README name the four authors, in
  the manuscript's order: Liyuan Gu (first author), Mingyue Miao (corresponding
  author), Lihua Huang, Fangxia Zhao. No author field is left for a reader to fill in,
  and the scan finds no bare task marker of the kind left by an unfinished draft. The
  markers are described rather than written out here, on the same reasoning as item 9:
  an audit that reproduces the strings it certifies absent would falsify its own claim.
- The two items that once awaited a decision have since been settled by the authors and
  are recorded as decisions rather than as open questions: the licence is MIT, and the
  repository URL is
  `https://github.com/liyuan112531/traffic-accident-support-rate-prediction`. Both are
  stated in `LICENSE`, `README.md`, `CITATION.cff`, `pyproject.toml` and `CHANGELOG.md`,
  and neither is described as pending anywhere in the tree.

## 11. No fake DOI or identifier

- **Result: pass.** `CITATION.cff` carries a `repository-code` field holding the real
  repository URL supplied by the authors, and no `doi` field at all. The README states
  that no DOI is asserted because none has been assigned. No ORCID identifier is
  present, because none was supplied.
- The `version` field is `1.0.0`, which is the software version declared identically in
  `pyproject.toml` and `CHANGELOG.md`. It is not a version of the manuscript, and no
  manuscript version or publication date is asserted anywhere.
- No accessioned identifier of any kind — DOI, ISBN, arXiv number, PubMed ID, ORCID,
  handle — appears in the release.

## 12. No unpublished confidential material

- **Result: pass.** The release contains no draft text, no reviewer correspondence, no
  manuscript file, no internal report and no experiment log. `experiments/`,
  `outputs/` and `reports/` directories do not exist in the release, and the README,
  the documentation and the audit records describe the method rather than quoting
  unpublished results.

## 13. Required wording present

- **Result: pass.** Each of the following occurs in the release, as found by the scan.
  Occurrences are counted per file, so a statement repeated across the README, the
  documentation and the audit records is counted more than once:
  - the data-availability paragraph — 4 files;
  - the synthetic-data disclaimer — 1 file;
  - the bounded-reproducibility paragraph — 2 files;
  - the statement that model weights and caches are not included — 4 files;
  - the MIT copyright line — 2 files, and the SPDX licence name — 4 files;
  - the repository URL — 9 files;
  - the funding statement — 2 files, and the grant number — 2 files.

  Matching is done on whitespace-normalised text with blockquote markers stripped.
  Both details matter: these are prose paragraphs, and a paragraph wrapped across lines
  inside a blockquote does not survive a raw substring test. Before that was corrected
  the scan reported three of the mandatory statements as absent when they were present,
  which is the failure mode that leads an author to rewrite wording that was already
  right.

  Matching is done on whitespace-normalised text with blockquote markers stripped.
  Both details matter: these are prose paragraphs, and a paragraph wrapped across lines
  inside a blockquote does not survive a raw substring test. Before that was corrected
  the scan reported three of the mandatory statements as absent when they were present,
  which is the failure mode that leads an author to rewrite wording that was already
  right.

## 14. Forbidden wording absent

- **Result: pass.** None of the five forbidden data-availability phrasings appears.
  Neither of the two forbidden strong reproducibility claims appears. The README states
  the weaker and accurate position: the repository supports methodological
  reproducibility and does not reproduce the manuscript's numbers.

## 15. Structure completeness

- **Result: pass.** Every path required by the release specification exists:
  `README.md`, `LICENSE`, `CITATION.cff`, `CHANGELOG.md`, `.gitignore`,
  `requirements.txt`, `environment.yml`, `pyproject.toml`, the nine ordered
  `src/` subpackages plus `src/pipeline.py`, the nine stage scripts, the six test
  modules plus `conftest.py`, `config/` with both YAML files, `data/` with the schema
  and the synthetic dataset, `examples/synthetic_demo/` with its runner, README and
  expected output, the five `docs/` documents, the five `audit/` records, and
  `.github/workflows/tests.yml`.
- The tree holds 78 files and 844 KB. It contains no generated artefact, no cache, no
  local configuration, no scratch directory and no file outside the intended set:
  `__pycache__`, `.pytest_cache` and every scratch directory used while verifying were
  removed after the last command that could recreate them.

## 16. Executability

- **Result: pass.** The test suite runs green: 135 tests, one third-party deprecation
  warning, no failures. Every item was re-run after the finalisation pass that added the
  repository URL, confirmed the licence and recorded the two configurations, so the
  numbers below describe the tree exactly as it stands.
- The demonstration runs to completion and its verification block reports all seven
  checks as passing, with `overall: PASS`. It takes roughly two and a half to five
  minutes depending on how many cores are free, since the expert pool is genuinely
  trained on every item in every fold.
- Every script was executed rather than read. All nine were run on the synthetic
  dataset: the two generators at their defaults, and the six analysis stages both at a
  reduced scale and at the demonstration scale set in `config/experiment_config.yaml`.
  Running them is what the three defects recorded under `Fixed` in `CHANGELOG.md` were
  found by.
- The tables the demonstration writes were compared against the committed
  `examples/synthetic_demo/expected_output/`. Four of the five files are byte-identical
  to a fresh default run. The fifth, `run_diagnostics.json`, differs only in
  `elapsed_seconds`, which is a wall-clock measurement and cannot reproduce by nature.
- The two generated inputs are byte-reproducible: `data/synthetic/synthetic_cases.csv`
  from its recorded seed, and `data/schema/data_dictionary.csv` from the generator.
  `tests/test_cli_contract.py` asserts both, so a stale copy fails the suite rather than
  shipping.
- **The continuous-integration commands were verified before being committed.** Each of
  the three command lines in `.github/workflows/tests.yml` was executed against the
  released tree on the pinned environment: the fast lane returned
  `115 passed, 20 deselected`, the full suite `135 passed`, and the demonstration's
  assertion line matched its own log. The `set -o pipefail` guard was checked in both
  directions, because without it a failing demonstration would be masked by the
  successful `tee` in the same pipeline. What has **not** been verified is the workflow
  runner: no repository was created, nothing was committed and nothing was pushed, so
  the first execution on GitHub Actions is still to happen. The YAML parses, the pin
  set has manylinux wheels for every package on Python 3.12, and the interpreter
  version is present in the `setup-python` manifest.

## 17. Nothing outside the release directory was modified

- **Result: pass.** All work was confined to the release directory. The study
  codebase was read only; no file in it was modified, moved, renamed or deleted.
  `source_mapping.md` records what was read and what it informed.

## 18. No unfilled placeholder and no unresolved decision

- **Result: pass.** Six placeholder patterns were searched across all seventy-eight
  files, and every one of them returns **zero occurrences**: bracketed confirmation and
  completion markers; bracketed repository-URL slots; the all-caps and hyphenated
  username templates; the reserved documentation domain; an unassigned DOI written in
  DOI form; and the bare task markers left behind by an unfinished draft. The categories
  are named here rather than written out, for the reason given at the head of this
  document.
- The `LICENSE` file carries the standard MIT text unmodified, with the four copyright
  holders and no bracketed status marker. `README.md` states the licence as MIT.
  `CITATION.cff` states `license: MIT` and carries a real `repository-code`.
- `config/paths.example.yaml` no longer contains any illustrative host. It previously
  held a commented-out domain reserved for documentation, which is not a placeholder
  awaiting a value but which reads like a configured endpoint; it has been removed
  rather than explained, and the key it illustrated is the only one that no code in the
  release consumes.
- **A documentation defect was found here and corrected.** An earlier draft of
  `docs/data_access.md` told the reader to copy `config/paths.example.yaml` to
  `config/paths.yaml` in order to avoid passing `--data` on every command. That does not
  work: no stage script consults those keys, because their defaults are fixed in
  `src/utils/cli.py`. The advice was replaced with the supported route, and the template
  header, the audit record and `public_release_summary.md` now state the scope exactly —
  the keys are read by `src.utils.paths.resolve` for library callers, and no entry point
  is wired to them. The code was left alone; the prose was wrong, not the code.
- Decisions that were open in an earlier draft — the licence, the repository URL and the
  two run configurations — are settled and stated rather than deferred. The items that
  remain genuinely open are listed in `CHANGELOG.md` under `Open items` and in
  `public_release_summary.md` under manual review, and each is a decision for the
  authors about the manuscript rather than a gap in the repository.

## Residual risks

Recorded rather than resolved, because they cannot be closed from inside the
repository.

- **Re-identification from aggregates.** No aggregate small enough to point at an
  individual is released, and the synthetic dataset is not derived from real records,
  so this route is closed by construction. It remains the reason the data is not
  distributed at all.
- **Interpretation of the synthetic results.** An inattentive reader could take a
  number from the expected output as a level of performance. The README, the synthetic
  dataset README, the example README and the script output all state otherwise, and the
  run settings record the demonstration scale.
- **Licence and URL.** Settled: MIT, and the repository URL above, recorded in five
  files. Neither is an open item any more.
- **Published identifiers.** No DOI and no ORCID is recorded, because none exists yet.
  If either is obtained later it belongs in `CITATION.cff` and the README, and the
  absence is a fact about publication status rather than an omission.
- **Consistency with the final manuscript.** The README documents the method as
  implemented here. If the manuscript's final wording differs, the method description
  should be reconciled; the repository describes the code, and the code was not changed
  to match any document.
