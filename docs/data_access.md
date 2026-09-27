# Data access

## Statement

> The original case-level dataset is not publicly distributed because the
> underlying court judgment documents contain personally identifiable information
> and are not currently publicly accessible from the original source.

This repository provides the code, the configuration, the data schema and synthetic
examples that support methodological reproducibility. It contains no judgment texts
and no real case-level data.

The code is hosted at
<https://github.com/liyuan112531/traffic-accident-support-rate-prediction>. Questions
and issues concerning the code belong in that repository's issue tracker; questions
concerning access to the restricted data belong to the corresponding author, at the
contact details given in the manuscript.

## What the study data is

The empirical analysis was conducted on a corpus of traffic accident compensation
judgments. Each judgment records, for one accident claim, the amounts claimed and the
amounts awarded under the statutory compensation items, together with the procedural
context of the decision. The modelling unit used here is the case-item pair: one
claimed head of loss within one case.

Two properties of that material determine everything in this document.

The judgments contain personal information. A compensation judgment identifies the
parties, describes the injury and the accident, and records monetary amounts that are
specific to an individual. The combination is identifying even when a name is
removed, so ordinary anonymisation is not sufficient to make a release safe.

The judgments are not currently publicly accessible from the original source. The
corpus was assembled for research purposes and the source does not offer a public
interface from which the same records could be retrieved by a third party.

## What is distributed

| Item | Contents | Why it is safe |
|---|---|---|
| `data/schema/data_dictionary.csv` | One row per documented column: name, type, description, role, allowed values or encoding, whether the experts use it, whether the meta-learner uses it, whether it is observable before judgment, and whether it is safe for release | It describes structure. It contains no data value, no name, no identifier and no amount record. |
| `data/synthetic/synthetic_cases.csv` | 600 cases with sixteen claim/award pairs, a graded injury-severity attribute, a procedural level and two academic years | Generated from parametric distributions with a recorded seed. Not derived from any real record, not an anonymised copy, no textual material, and no identifier from the study dataset. |
| `src/`, `scripts/`, `tests/`, `config/` | The implementation, the stage scripts, the tests and the configuration | Code and fixed configuration. No data values. |
| `examples/synthetic_demo/expected_output/` | The tables a demonstration run produces | Computed from the synthetic dataset only. |

## What is not distributed

- The case-level dataset, in any form.
- The judgment documents, including excerpts, screenshots and paraphrases.
- The claim and award amounts of any real case.
- The derived item-level or case-level rates of any real case.
- The real feature matrix, the real out-of-fold predictions and the real attribution
  values.
- Any case identifier, judgment number, name, identity number, address or telephone
  number.
- The fitted models and any model weight cache.

`audit/data_leakage_audit.md` records the scan performed over every data-bearing file
type, and `audit/public_release_audit.md` records the release checklist.

## The schema

`data/schema/data_dictionary.csv` is the contract a case table must satisfy to be
processed by this code. Its columns are:

| Column | Meaning |
|---|---|
| `feature` | Column name as it appears in a case table |
| `dtype` | Data type |
| `description` | What the column holds |
| `role` | Where it is used: key, target, input to the experts, or meta-learner only |
| `allowed_values_or_encoding` | The permitted range, or the encoding for a categorical |
| `used_in_experts` | Whether the expert stage reads it |
| `used_in_meta_learner` | Whether the meta-learner reads it |
| `pre_judgment` | Whether the value is observable before the judgment |
| `public_release_safe` | Whether the column may appear in a public artefact |

A case table must provide, at minimum, a `case_id`, the sixteen `<item>_claim`
columns and the sixteen `<item>_award` columns. `claim_total`, `award_total`,
`doc_year`, `doc_region` and `doc_proc_level` are read when present;
`claim_total` and `award_total` are recomputed from the item columns if absent.

The contract exists so that the pipeline can be run on a dataset it has never seen.
The code validates the contract on load and fails loudly if a required column is
missing, if a case identifier repeats, if an amount is not finite, or if an amount is
negative. It never fills a missing amount with a default, because a defaulted amount
would silently become a fabricated observation.

## Fields that must never be used as features

The following are post-outcome attributes: they are decided by, or observed only
after, the judgment. Using them would leak the label and inflate every metric.

- any award amount or anything derived from one;
- any support rate;
- the liability apportionment ratio;
- the disability grade;
- the count of heads of loss granted;
- any property of the judgment text, including its length.

`src.preprocessing.schema.FORBIDDEN_FEATURE_TOKENS` records these, and
`src.features.claim_features.assert_no_leakage` enforces the list on the
request-side block, so the guard is executable rather than documentary.

## Using the code on your own data

1. Arrange your table with one row per case and the columns named as the dictionary
   specifies. The sixteen `<item>_claim` and `<item>_award` pairs are required.
2. Point `--data` at the file:

   ```bash
   python scripts/evaluate.py --data path/to/cases.csv --output path/to/output
   ```

3. Pass the same arguments on every command, or wrap them in a short shell alias. The
   stage scripts always take `--data` and `--output` explicitly, and their defaults are
   the synthetic table and `examples/synthetic_demo/expected_output`, both relative to
   the repository root.

   `config/paths.example.yaml` is a separate mechanism for library callers rather than
   for the scripts. It documents the keys that `src.utils.paths.resolve` reads — from
   `config/paths.yaml` if you create one, or from an environment variable named
   `JRSP_<KEY>`, which takes precedence over the file. No stage script consults those
   keys on your behalf, so copying the template will not change where a script reads or
   writes. It is shipped so that a program built on this package has a documented
   interface and the schema is discoverable. `config/paths.yaml` is excluded by
   `.gitignore`, so a local override cannot be committed by accident.

If you have a dataset that does not follow this contract — a different item universe,
or claim totals without item detail — the pipeline will not run on it as-is. The
aggregation layer in particular requires the per-item claimed amounts, because the
aggregation weights are the case's own claim shares. There is no fallback that
substitutes uniform weights: a uniform aggregation discards the request structure and
is provided only as a diagnostic benchmark
(`src.aggregation.claim_weighted.uniform_benchmark`).
