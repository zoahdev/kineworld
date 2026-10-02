# Verification plan

This is a software utility, not a new model experiment or a benchmark result.
The following checks are fixed before implementation and test execution. They
are local engineering acceptance checks, not externally registered research.

- Enforce strict UTF-8 JSON, exact record fields, no duplicate object keys, finite
  metrics, safe integer types (not booleans), step ordering, seed pairs, digest
  syntax, and bounded input/bootstrap work.
- Detect repeated episode identities across declared train, validation and test
  splits, cross-split content hashes, and cross-split generator seed reuse.
- Report shared generator namespaces without implying that different seeds leak.
- Pair on source plus the complete (episode_idx, start_step, goal_step) tuple and
  evaluation seed. Reject duplicates, missing arms, inconsistent condition labels,
  non-test tasks, unregistered episodes, and unequal candidate budgets.
- Count rare-condition coverage in distinct paired test episodes, not windows.
- Compute paired candidate-minus-baseline differences within each task, their
  median within each episode, then the median across episodes. Bootstrap whole
  episodes with replacement using a fixed seed. Independently check known
  arithmetic, ties, zero/one episode, extreme floats, and ordering invariance.
- Suppress the effect report on audit errors. Warn on tiny samples and degenerate
  bootstrap distributions. Never output a significance/superiority claim.
- Run all unit tests and both documented CLI examples. Record exact commands,
  observed exit codes, Python version and artifact SHA-256 hashes.

Only public repository interfaces and synthetic fixtures are inputs. No private
model code, model recipes, user data, checkpoints, paid compute, pickle files,
executable configuration, or network operations are used by this utility.
