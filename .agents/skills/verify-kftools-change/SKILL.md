---
name: verify-kftools-change
description: Select and run affected checks for kftools library, documentation, or packaging changes. Use for local change verification, not performance benchmarking, dependency upgrades, or release publication.
---

# Verify a kftools change

Input: the intended change or diff and an available development interpreter.
Work from the repository root. Read [the development guide](../../../docs/development.md)
for environment setup, the change-to-check table, and command definitions;
keep that guide as the single test-selection reference.

1. Inspect `git status --short` and the relevant diff. Identify public callers
   of changed private helpers, not just the file bearing a similar name.
   For numerical/default/output changes, read the affected section of
   [data semantics](../../../docs/data-semantics.md); for readers, also consult
   [file formats](../../../docs/file-formats.md).
2. Confirm the chosen interpreter has the documented dependencies. If setup is
   needed, use the guide's venv and constrained install; do not install into the
   user's global Python or regenerate pins. Missing tools/interpreters or a
   failed installation are environment blockers, not passing checks.
3. Choose the table's focused tests and inspect cross-module callers/tests with
   `rg`. Run them with the development interpreter and pytest's existing warning
   policy. Shared validation changes warrant the full suite. For documentation,
   execute affected examples in a temporary directory with `MPLBACKEND=Agg`
   and verify their assertions/output; check links and commands against source.
4. Run the guide's delivery check. For packaging changes, also build and test
   the newly built wheel outside the checkout using the existing environment
   helper. Preserve old artifacts and select the wheel explicitly. Treat
   minimum/latest dependency checks and audits as network work, separate from
   local tests; do not launch NCBI downloads or large benchmarks for this task.
5. On failure, retain the command, exit status, and useful diagnostic. Determine
   whether it is an environment or behavior failure before fixing it; do not
   hide failures by changing warning filters, coverage gates, or dependency pins.
   If blocked, continue independent checks and label the blocked check unverified.

Output: a concise mapping of changed behavior to checks run, their results,
and checks skipped with reasons. Distinguish execution from static inspection.
Success means the selected behavioral checks and delivery gate pass, plus any
affected packaging checks; a subset alone does not certify delivery. Review
`git diff --check` and the final status for unintended output or unrelated edits.
This skill does not authorize a push; use `prepare-github-push` when requested.
