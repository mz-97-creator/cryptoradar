# Codex research version boundaries

- Work only in this independent checkout, `/workspace/cryptoradar-codex`.
- `/workspace/cryptoradar` is Claude's source checkout. Read access is allowed;
  never edit, delete, install dependencies there, change its refs, or run commands
  that write its configuration, cache, database, or generated files.
- Publish only to `codex/research-v1`. Never update `main`, `tick`, or `data`.
- No automatic trading, live notifications, production schedules, or merging.
- Preserve old prediction records with their original settlement method. Do not
  rewrite past outcomes to make results look better.
- Research labels use next-bar entry, explicit costs, purged time splits, and
  exclude unknown intrabar barrier ordering and censored observations.
- Improvements to code integrity are not evidence of predictive improvements.
  Report those separately and cite actual held-out/live results when available.

Verification: `python -m unittest tests.test_research_integrity -v` and
`python -m tests.cloud_selftest`. During this session, the original environment's
Python can be used with `-B` from this checkout to read installed dependencies.
