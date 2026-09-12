# Native source presentation

Publication integration reformats the native engine and test file to satisfy the existing Ruff gate. The original scientific source, primary lock, and Supplement S1 remain byte-for-byte unchanged. `presentation_identity.json` records the original and presented engine digests. After excluding module-level import statements, the entire engine abstract syntax tree is identical; the only import removal is unused `Sequence`. All test cases are retained, with an unused loop variable renamed and a test-only lambda expressed as a named helper. No numerical operation, threshold, budget, or result is changed.

Use Supplement S1 for hash-locked reproduction. The native repository is the tested presentation derivative; its byte digest is not substituted into the frozen primary lock.
