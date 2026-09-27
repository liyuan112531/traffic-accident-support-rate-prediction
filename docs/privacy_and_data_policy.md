# Privacy and data policy

## The problem

The study data was extracted from traffic accident compensation judgments. A judgment
of that kind records the parties to an accident, describes how the injury occurred,
states the medical findings and the disability assessment, and lists the amounts
claimed and awarded under each head of loss. Most of that is identifying in
combination even after names are removed. The accident date and location, the injury
description, the procedure and the amounts together narrow a record to a small number
of individuals, and a compensation amount is personal financial information.

The judgments are also not currently publicly accessible from the original source, so
a reader cannot obtain them independently and the question of whether a release would
be lawful cannot be answered by pointing at an existing public record.

## The decision

The empirical dataset is not distributed.

> The original case-level dataset is not publicly distributed because the underlying
> court judgment documents contain personally identifiable information and are not
> currently publicly accessible from the original source.

What is released instead is everything needed to inspect and reuse the method: the
implementation, the fixed configuration, the data schema, the tests, and a synthetic
dataset that lets the code be executed end to end.

## Absolute exclusions

The following never appear in this repository, in any form:

| Category | Examples |
|---|---|
| Judgment text | full text, excerpts, quotations, paraphrases, screenshots |
| Case-level records | claim and award amounts, derived item and case rates, targets |
| Feature matrices | the real request-side matrix, the real meta representation |
| Predictions | real out-of-fold predictions, per-case predictions, per-item predictions |
| Attribution values | SHAP values computed on real data |
| Direct identifiers | names, identity numbers, addresses, telephone numbers |
| Quasi-identifiers | judgment numbers, docket numbers, court identifiers, case identifiers |
| Fitted models | weight files, caches, checkpoints, notebooks with committed outputs |
| Local material | experiment logs, local screenshots, absolute paths, credentials |

The list is enforced by construction rather than by intention:
`.gitignore` blocks the file types and directories through which any of the above
would arrive, `audit/data_leakage_audit.md` records the scan that was run over every
data-bearing file type in the release, and `audit/public_release_audit.md` records the
checklist.

## The synthetic dataset

The synthetic dataset is generated from parametric distributions by
`scripts/make_synthetic_data.py`, using a recorded seed. It is not a perturbed copy of
real records and it is not an anonymised version of the study data. Three properties
follow, and each was a design requirement.

It cannot be traced back. It shares no row with the study data, because it was never
derived from it. There is no record whose nearest real neighbour could be found.

It cannot be mistaken for real. Every case identifier begins with `SYNTH-`, the
region field is literally `SYNTHETIC-REGION`, the graded injury-severity field is
named `injury_severity_synthetic`, and the file lives under `data/synthetic/`. A
synthetic row is identifiable as synthetic at a glance and in an automated check;
`tests/test_synthetic_pipeline.py` asserts the naming.

It carries no information about real cases. The construction parameters are
documented in the script as construction parameters. In particular the rare items were
given higher claim probabilities than they have in reality, so that every item has
enough training units for the demonstration to be meaningful. Those probabilities are
therefore not estimates of anything and must not be read as frequencies.

The dataset does reproduce the *structure* of the study data — sixteen claim and award
pairs, a graded severity attribute, a procedural level, two academic years, a claim
weight structure — because the point is to exercise the code, and the code is written
against that structure.

## Personal data in the code

No personal data appears in the source, the configuration, the tests, the
documentation or the audit records. Specifically:

- no author contact details beyond the names and roles that appear in the manuscript
  and in `CITATION.cff`;
- no credentials of any kind: no API key, no access token, no database password, no
  Git credential, no private key;
- no absolute path from a developer machine, in any file;
- no network path, server name or internal address.

`audit/path_and_secret_audit.md` records the scan behind these claims.

## Handling a dataset of your own

If you run this code on your own judgments, the same considerations apply, and the
code does not protect you from them.

Two things the code does help with. It writes only to the directory you pass as
`--output`, so a run does not scatter derived case-level tables across the repository;
and the tables it writes are named so they can be excluded by a rule. Add the output
directory to `.gitignore` before your first run.

Two things it cannot do. It cannot tell whether your dataset is lawfully held, and it
cannot make a release of your data safe. If you intend to publish data, the questions
to settle first are whether the source permits redistribution, whether the individuals
concerned have a reasonable expectation that the records remain private, and whether
the records are identifying in combination. Item-level compensation amounts are
identifying more often than they look.

## Questions

If you believe some material in this repository discloses personal information, please
raise it with the corresponding author. The intent is that no such material is present,
and a specific report is worth more than a general policy statement.
