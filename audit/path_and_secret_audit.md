# Path and secret audit

A scan of every file in the release for local absolute paths, environment details and
credentials.

## Patterns searched

| Category | Pattern |
|---|---|
| Absolute Windows path | a drive letter followed by a path |
| Windows workspace path | any path containing the study workspace directory name |
| Windows user profile | `\Users\<name>\` |
| POSIX home path | `/home/<name>/`, `/Users/<name>/` |
| UNC or network path | `\\server\share` |
| Environment account identifier | the workstation account-code pattern |
| Email address | standard address pattern |
| IPv4 address | dotted quad |
| Secret-like assignment | `api_key`, `secret`, `password`, `token`, `access_key`, `private_key`, `client_secret`, `apikey` followed by a value |
| Bearer or Hugging Face token | `Bearer <token>`, `hf_<token>` |
| Database connection string | the MySQL, PostgreSQL, MongoDB and Redis URI schemes and their variants |
| Internal experiment code | the study's `E`-series experiment numbering, single- and double-digit |

## Result

**Pass**, with no finding in any released file.

## Item-by-item

### Absolute local paths

No absolute path appears in any released file. Path handling is centralised:

- `src/utils/paths.py` derives the repository root from
  `Path(__file__).resolve().parents[2]`, so every derived location is relative to
  wherever the repository is checked out.
- `config/paths.example.yaml` is a documented template. Its values are relative paths
  inside the repository — the same ones the scripts default to — so it carries no
  machine-specific path. `config/paths.yaml`, the file a user would create and which is
  where an absolute path would legitimately live, is excluded by `.gitignore`.
- No entry point is wired to that template. The stage scripts take `--data` and
  `--output` explicitly and default them in `src/utils/cli.py`; `config/paths.yaml` and
  the `JRSP_<KEY>` environment variables are read by `src.utils.paths.resolve` for
  library callers. `docs/data_access.md` states this scope, so the template cannot be
  mistaken for a control surface for the scripts.
- Every script accepts `--data` and `--output`. The defaults are repository-relative.
- Output paths are rendered for display by `src/utils/cli.display_path`, which prints a
  path relative to the repository when it lies inside it and the path as given
  otherwise. It never fabricates a relative form for an outside directory, which is why
  a run pointed at a directory elsewhere prints an absolute path — that is the user's
  own path being echoed back, not a path embedded in the release.
- No documentation instructs the reader to create a directory at a specific absolute
  location.

### Environment details

- No user name, home directory, workstation name, account code or machine identifier
  appears.
- No internal server name, internal address or network share appears.
- No mention of a private repository, an internal mirror or a corporate proxy appears.
- Two external hosts are named, and both are meant to be public. One is the project's
  own repository,
  `https://github.com/liyuan112531/traffic-accident-support-rate-prediction`, which the
  authors supplied. The other is the optional TabPFN weight source, which is referred to
  generically rather than named as a private endpoint. Neither is an internal address and
  neither carries a credential.

### Credentials

- No API key, access token, password, database credential, private key or Git
  credential appears.
- No script in the release authenticates to anything. The only network access any code
  path can trigger is the optional TabPFN weight download, which needs no credential
  and is disabled by default.
- No environment variable holding a credential is read. `src/utils/env.py` sets
  thread-count and hash-seed variables only, and it uses `setdefault` so it cannot
  overwrite a value the user has already set.
- `.github/workflows/tests.yml` contains no `secrets.*` expression and no token. It
  relies only on the default checkout credential, and it narrows that explicitly with
  `permissions: contents: read`.

### Model weight caches

- No cache directory is present: no `model_cache/`, `models--*/`, `hub/`, `cache/` or
  `pretrained/`.
- No weight file is present in any format.
- `.gitignore` excludes every weight format and cache directory name, so a cache
  created by running the optional TabPFN path cannot be committed.

### Internal experiment codes

- No code from the study's `E`-series experiment numbering appears in any released file.
  The pattern is described above rather than reproduced in full, so that this audit does
  not become the one file in the release containing it — which would make its own claim
  false.
- The end-to-end test asserts that the codes do not appear among the pipeline module's
  string constants, and assembles the codes from a numeric range so that the test does
  not itself introduce them.
- The analyses are named formally throughout: core model performance, statistical
  robustness, case-level aggregation validation, temporal extrapolation,
  interpretability analysis.

## Files inspected

The scan covers every text file in the release, including the code, the configuration,
the tests, the documentation, the audit records, the CI workflow
(`.github/workflows/tests.yml`), the CSV and JSON data files, and the extensionless
files (`LICENSE`, `.gitignore`). Binary files are none, so nothing was skipped for being
unreadable.

## Prevention

The `.gitignore` in the release blocks the routes by which any of the above would
arrive:

- `config/paths.yaml` and `config/*.local.yaml` for local path overrides;
- `.env` and variants, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.jks`, `secrets.yaml`,
  `credentials*`, `*credential*.json`, `*token*.txt`, `.git-credentials`, `.netrc` for
  credentials;
- `tabpfn*/`, `models--*/`, `hub/`, `model_cache/`, `pretrained/`, `.cache/` and every
  weight format for caches and weights;
- `tmp/`, `scratch/`, `scratch_*`, `*_scratch/`, `logs/`, `*.log` for scratch output,
  which is where a path or a sampled row most often leaks from;
- `.ipynb_checkpoints/` and `outputs/` for notebook and run artefacts.

## A note on the shape of this risk

A path or a secret rarely leaks from a deliberate decision. It leaks from an artefact
nobody thought of as content: a log file, a notebook checkpoint, a crash traceback
pasted into an issue, a `.pytest_cache` committed by accident. The defence that works
is a default of exclusion plus a scan, which is what is implemented here, rather than
relying on remembering to check each file.
