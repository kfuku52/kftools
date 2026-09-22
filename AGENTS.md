<!-- BEGIN KF AGENT POLICY: source=https://github.com/kfuku52/kf-agent-policy; version=10; sha256=82e3c0eb467582a414d9a6b2feaaaf6f5c8ae330d30f2e3efbf8c303155d0e2e -->
# Common agent policy

Repository-specific instructions override these defaults.

- Follow the user's task scope within higher-priority instructions and execution
  permissions. Complete implementation through affected verification and a result
  report; a plan or investigation ends with its requested deliverable. Continue
  authorized work without repeated approval; identify actual blocking boundaries.
- Inspect the worktree and preserve unrelated changes. Refresh remote information
  when needed; do not merge, rebase, or switch branches merely to inspect it.
- Prefer the default branch when starting work without an established branch.
  Preserve an existing task branch; follow explicit user branch instructions.
  Never create or switch branches solely for a commit, push, release, or PR.
- Change or recommend branch protection only when explicitly asked. Honor explicit
  repository-specific direct-push exceptions; otherwise report a rejected push
  without bypassing protection or inventing a branch or PR.
- Unpublished implementation details may be redesigned; preserve existing public
  APIs, file formats, and saved-data compatibility unless a breaking change is
  authorized. Update affected producers, consumers, tests, examples, and docs.
- Fix verified root causes; do not hide failures with fallbacks or weaker checks.
  Document unavoidable workarounds and their removal conditions.
- Read relevant docs and run the repository's check entrypoint for the change and
  phase. Verify affected behavior; report checks run and omitted. Repeat or broaden
  successful checks only for new changes, failures, or unresolved concerns.
- For library metadata, require demonstrated incompatibility for exact pins or
  upper bounds; keep reproducibility locks separate.
- When editing READMEs, keep them concise with useful visuals inline; put extended
  guides in linked documentation.
- For GitHub push/release work, use `prepare-github-push` in `.agents/skills/`.
  Local-only commits need no version bump; GitHub pushes require one.
- For software performance work, use `benchmark-performance` in `.agents/skills/`.
  Performance claims require comparable measurements and equivalent output.
- For GitHub Actions edits, use `optimize-github-actions` in `.agents/skills/`.
  Preserve required coverage; never run untrusted PR code on self-hosted runners.
<!-- END KF AGENT POLICY -->

# Working in kftools

- Start with [README.md](README.md) for the public module map, then
  [docs/development.md](docs/development.md) for setup and check selection.
  This is a Python library with no installed CLI. Public entry points are
  `kftools/kf*.py`; shared tree, table, and regression helpers are private
  modules used by those APIs. Follow their callers when choosing tests.
- From the repository root, create a Python 3.14 venv with
  `python3.14 -m venv .venv`, then `make install PYTHON=.venv/bin/python`.
  Use this interpreter explicitly; a system Python may lack the dev tools.
  Do not regenerate dependency snapshots as part of routine setup.
- Use `make test PYTHON=.venv/bin/python` for local runtime feedback,
  `make lint PYTHON=.venv/bin/python` and
  `make typecheck PYTHON=.venv/bin/python` for quality checks, and
  `make check PYTHON=.venv/bin/python` before delivery. The development guide's
  [change-to-check table](docs/development.md#choose-checks-for-a-change) gives
  focused test selections and separates local checks from network/install work.
  Use [verify-kftools-change](.agents/skills/verify-kftools-change/SKILL.md) when
  deciding and executing affected verification; do not substitute a focused
  subset for the delivery check.
- Before changing numerical behavior, defaults, mutations, labels, or output
  columns, read the relevant [data semantics](docs/data-semantics.md); for
  parsers, also read [file formats](docs/file-formats.md). Preserve expression
  scales, model support, branch IDs/missing-value conventions, tree-copy
  behavior, and regression defaults unless a behavior change is requested.
  Public annotations and `py.typed` are part of the supported API. Keep runtime
  compatibility with Python 3.10 even though type checks run on 3.14.
- Do not hand-edit generated `build/`, `dist/`, `kftools.egg-info/`, caches, or
  `.venv/`. Preserve pre-existing artifacts and unrelated files; `make clean`
  removes build artifacts and coverage. Keep example outputs in a temporary
  directory. Do not initialize/update the user's NCBI database for routine
  verification; taxonomy tests use fakes and real annotation may download data.
- Finish by reviewing the diff and reporting changed behavior/files, commands
  and results, and skipped checks with reasons. For push, use the existing
  `prepare-github-push` skill; the version source is `kftools/__init__.py`.
