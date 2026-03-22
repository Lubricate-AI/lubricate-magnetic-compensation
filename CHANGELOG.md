# CHANGELOG

<!-- version list -->

## v1.23.0 (2026-03-22)

### Bug Fixes

- Resolve lint violations and strengthen CV test assertions
  ([#71](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/71),
  [`f8e6997`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f8e69977f835cb87ceb718ed7de454140e4f8308))

### Features

- Add --use-cv, --cv-folds, --auto-regularize CLI flags to calibrate command
  ([#71](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/71),
  [`f8e6997`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f8e69977f835cb87ceb718ed7de454140e4f8308))

- Add use_cv, cv_folds, auto_regularize fields to PipelineConfig
  ([#71](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/71),
  [`f8e6997`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f8e69977f835cb87ceb718ed7de454140e4f8308))

- Cross-validation and automatic regularization method selection
  ([#71](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/71),
  [`f8e6997`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f8e69977f835cb87ceb718ed7de454140e4f8308))

- Implement CV-based alpha selection using TimeSeriesSplit
  ([#71](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/71),
  [`f8e6997`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f8e69977f835cb87ceb718ed7de454140e4f8308))

### Testing

- Add precondition guard to auto_regularize explicit method test
  ([#71](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/71),
  [`f8e6997`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f8e69977f835cb87ceb718ed7de454140e4f8308))

- Add unit tests for auto_regularize trigger in calibrate()
  ([#71](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/71),
  [`f8e6997`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f8e69977f835cb87ceb718ed7de454140e4f8308))

- Rename vacuous CV test to accurately describe its assertion
  ([#71](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/71),
  [`f8e6997`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f8e69977f835cb87ceb718ed7de454140e4f8308))

- Strengthen CLI flag tests with positive exit code assertion
  ([#71](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/71),
  [`f8e6997`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f8e69977f835cb87ceb718ed7de454140e4f8308))


## v1.22.0 (2026-03-22)

### Bug Fixes

- Add HeadingType annotation to fix pyright error in compensate_heading_specific
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Add input validation guards to compute_vif
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Address PR review issues in heading-specific calibration
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Apply ruff format to vif module and tests
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Initialize interference array to zero; fix docstring; add missing heading test
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Make segmentation helpers public; fix pyright type errors
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Resolve lint errors in test files
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

### Chores

- Commit uv.lock and implementation plan doc
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Remove implementation plan doc from repo
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Update uv.lock ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

### Features

- Add heading-specific compensation routing
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Add per-heading calibration with VIF diagnostics
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Add use_heading_specific_calibration config field
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Add VIF computation for multicollinearity diagnostics
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Export heading-specific calibration API from lmc
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Heading-specific model selection for multicollinearity reduction
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

### Refactoring

- Clarify A-matrix rebuild comment; add multi-segment heading test
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

- Remove internal function names from config field description
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))

### Testing

- Add integration tests for heading-specific model selection
  ([#69](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/69),
  [`8e3caa5`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8e3caa535431fbba01ee31bf93ca024ba1ecf869))


## v1.21.0 (2026-03-22)

### Bug Fixes

- Address Copilot PR review comments
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

### Chores

- Commit uv.lock and implementation plan for RLS feature
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

- Fix lint issues in rls module
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

- Remove implementation plan (documented in issue #59)
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

- Remove unused imports from rls module (Task 1 cleanup)
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

### Documentation

- Fix covariance docstring (np.diag gives variances, not std devs)
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

### Features

- Add rls_to_calibration_result for compensate() compatibility
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

- Add RLSState dataclass and initialize_rls
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

- Add update_rls single-sample Kalman gain update
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

- Add update_rls_batch for DataFrame-based bulk updates
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

- Export RLS public API from lmc package
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

- Implement Recursive Least-Squares (RLS) for online coefficient updating
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

### Testing

- Verify forgetting factor enables adaptation to coefficient drift
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))

- Verify RLS convergence to OLS on static data
  ([#67](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/67),
  [`f892e1c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f892e1c9f4a6a387c236059cff9fc0680578a9fb))


## v1.20.0 (2026-03-21)

### Bug Fixes

- Persist regularization diagnostics to JSON output, tighten multicollinear test, add CLI
  mutual-exclusion test
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))

- Use compute_uv=False in SVD and add ridge diagnostic tests
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))

### Chores

- Add scikit-learn as a runtime dependency
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))

- Apply ruff format to config.py
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))

- Update uv.lock after merge from main
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))

### Features

- Add LASSO and ElasticNet CLI flags to calibrate command
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))

- Add LASSO and ElasticNet config fields with mutual-exclusion validator
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))

- Add LASSO and ElasticNet regularization for multicollinearity handling
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))

- Add selected_alpha and effective_dof diagnostic fields to CalibrationResult
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))

- Implement ElasticNet (L1+L2) regression branch in calibrate()
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))

- Implement LASSO (L1) regression branch in calibrate()
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))

### Testing

- Add multicollinear comparison tests for ridge, LASSO, and ElasticNet
  ([#64](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/64),
  [`3f0f4a1`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/3f0f4a137efa659a01204c2f873412945ec71ce6))


## v1.19.0 (2026-03-21)

### Documentation

- Fix n_terms docstring to include 21-term D-model
  ([#63](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/63),
  [`c815501`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/c815501469c70f51f9850c88f7c0d92d209b21d9))

### Features

- Add C-model calibration guardrails and singular value analysis
  ([#63](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/63),
  [`c815501`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/c815501469c70f51f9850c88f7c0d92d209b21d9))

- Add singular value analysis to CalibrationResult
  ([#63](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/63),
  [`c815501`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/c815501469c70f51f9850c88f7c0d92d209b21d9))

- Warn when C/D-model calibration has < 10,000 samples
  ([#63](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/63),
  [`c815501`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/c815501469c70f51f9850c88f7c0d92d209b21d9))


## v1.18.0 (2026-03-20)

### Bug Fixes

- Clarify config description to specify only pitch/roll/yaw suppressed
  ([#55](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/55),
  [`582e448`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/582e4485ec5a29f160bd6f52bf7c926f80027436))

- Clarify warning message and docs for ill-conditioned baseline
  ([#55](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/55),
  [`582e448`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/582e4485ec5a29f160bd6f52bf7c926f80027436))

- Strengthen no-suppression test, update config description and docstring for
  condition_number_threshold
  ([#55](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/55),
  [`582e448`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/582e4485ec5a29f160bd6f52bf7c926f80027436))

- Suppress duplicate calibrate() warnings in calibrate_adaptive_maneuvers, tighten test assertions
  ([#55](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/55),
  [`582e448`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/582e4485ec5a29f160bd6f52bf7c926f80027436))

- Wrap long warning lines to satisfy ruff E501 (88 char limit)
  ([#55](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/55),
  [`582e448`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/582e4485ec5a29f160bd6f52bf7c926f80027436))

### Continuous Integration

- Upgrade slack-github-action to v3 for Node.js 24
  ([#53](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/53),
  [`2826713`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/282671345ddbd131acc4904c795205dd5e5fffd7))

### Documentation

- Add data quality guidance for adaptive compensation in theory.md
  ([#55](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/55),
  [`582e448`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/582e4485ec5a29f160bd6f52bf7c926f80027436))

- Move data quality section before References, clarify result.baseline label
  ([#55](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/55),
  [`582e448`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/582e4485ec5a29f160bd6f52bf7c926f80027436))

### Features

- Emit named per-maneuver warning when condition number exceeds threshold in
  calibrate_adaptive_maneuvers
  ([#55](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/55),
  [`582e448`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/582e4485ec5a29f160bd6f52bf7c926f80027436))

- Suppress blending weight for ill-conditioned maneuver types in compensate_adaptive
  ([#55](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/55),
  [`582e448`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/582e4485ec5a29f160bd6f52bf7c926f80027436))

- Validate coefficient stability in adaptive compensation
  ([#55](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/55),
  [`582e448`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/582e4485ec5a29f160bd6f52bf7c926f80027436))


## v1.17.0 (2026-03-20)

### Chores

- Add implementation plan and update lockfile
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))

- Exclude docs/plans from typos spellcheck (contains code examples)
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))

### Features

- Adaptive maneuver-based compensation with coefficient blending
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))

- Add _rolling_variance helper for maneuver intensity detection
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))

- Add adaptive maneuver config fields to PipelineConfig
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))

- Add AdaptiveCalibrationResult dataclass
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))

- Export adaptive compensation API from lmc
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))

- Implement calibrate_adaptive_maneuvers
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))

- Implement compensate_adaptive with rolling variance blending
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))

### Refactoring

- Add forward-use imports and fix f-string in adaptive module
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))

- Clean up adaptive module imports
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))

### Testing

- Add integration test for adaptive maneuver compensation
  ([#51](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/51),
  [`8bd5cfc`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8bd5cfc45bb31b71aec90f51f75aac0e50cd9cf5))


## v1.16.0 (2026-03-19)

### Bug Fixes

- Update CLI and config to support model_terms='d'
  ([#49](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/49),
  [`825517f`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/825517f2d2351f80b2c58f3bec311a3a2b3cd66e))

### Documentation

- Update term set descriptions to include model_terms='d' (rate derivatives)
  ([#49](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/49),
  [`825517f`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/825517f2d2351f80b2c58f3bec311a3a2b3cd66e))

### Features

- Add attitude rate derivatives as standalone features (model_terms='d')
  ([#49](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/49),
  [`825517f`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/825517f2d2351f80b2c58f3bec311a3a2b3cd66e))

- Add dcos_x/y/z column constants for rate derivative features
  ([#49](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/49),
  [`825517f`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/825517f2d2351f80b2c58f3bec311a3a2b3cd66e))

- Add model_terms='d' exposing raw dcos derivatives as features
  ([#49](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/49),
  [`825517f`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/825517f2d2351f80b2c58f3bec311a3a2b3cd66e))

- Extend model_terms to accept 'd' for rate-derivative term set
  ([#49](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/49),
  [`825517f`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/825517f2d2351f80b2c58f3bec311a3a2b3cd66e))


## v1.15.1 (2026-03-18)

### Bug Fixes

- Move B_total positivity check to validate_dataframe()
  ([#47](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/47),
  [`77a619c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/77a619c1d785f418207a3ab5ff21d793fae3ed73))

- Normalize direction cosines by fluxgate magnitude
  ([#47](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/47),
  [`77a619c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/77a619c1d785f418207a3ab5ff21d793fae3ed73))

- Normalize direction cosines by fluxgate magnitude, not B_total
  ([#47](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/47),
  [`77a619c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/77a619c1d785f418207a3ab5ff21d793fae3ed73))

### Chores

- Add ND to typos.toml allow-list
  ([#47](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/47),
  [`77a619c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/77a619c1d785f418207a3ab5ff21d793fae3ed73))

- Remove plan docs (moved to issue #46 comments)
  ([#47](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/47),
  [`77a619c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/77a619c1d785f418207a3ab5ff21d793fae3ed73))

- Update uv.lock for v1.15.0
  ([#47](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/47),
  [`77a619c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/77a619c1d785f418207a3ab5ff21d793fae3ed73))

### Documentation

- Add design and implementation plan for cosine normalization fix
  ([#47](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/47),
  [`77a619c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/77a619c1d785f418207a3ab5ff21d793fae3ed73))

### Testing

- Add unit-vector assertion and zero-fluxgate error test
  ([#47](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/47),
  [`77a619c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/77a619c1d785f418207a3ab5ff21d793fae3ed73))

- Use realistic COL_BTOTAL in feature test fixtures
  ([#47](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/47),
  [`77a619c`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/77a619c1d785f418207a3ab5ff21d793fae3ed73))


## v1.15.0 (2026-02-24)

### Bug Fixes

- Address review comments on coefficients JSON validation
  ([#45](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/45),
  [`c8d640f`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/c8d640fa29c20c80f06200224e330d00eecd26c3))

### Documentation

- Add physical justification for IMU rate substitution
  ([#44](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/44),
  [`ed43535`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/ed435350ec06c2939c574da89cf446f6bf146dcc))

- Add physical justification for IMU rate substitution in eddy-current terms
  ([#44](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/44),
  [`ed43535`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/ed435350ec06c2939c574da89cf446f6bf146dcc))

- Address Copilot review comments on IMU rate substitution section
  ([#44](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/44),
  [`ed43535`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/ed435350ec06c2939c574da89cf446f6bf146dcc))

### Features

- Validate coefficients JSON schema in compensate command
  ([#45](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/45),
  [`c8d640f`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/c8d640fa29c20c80f06200224e330d00eecd26c3))


## v1.14.0 (2026-02-24)

### Features

- Validate segment bounds in calibrate()
  ([#43](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/43),
  [`22e5fdf`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/22e5fdfa631faeb3042b24699623e69d06650fe5))


## v1.13.0 (2026-02-24)

### Features

- Guard auto-detect segmentation against under-sized DataFrames
  ([#42](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/42),
  [`fde1ff6`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/fde1ff65534695391e8e9cbf6a0f08d41263cacb))


## v1.12.0 (2026-02-24)

### Features

- Validate steady_mask length in _steady_mean_baseline
  ([#41](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/41),
  [`32b4b45`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/32b4b45345982b2f86edf7bc3601383fc0095130))


## v1.11.0 (2026-02-24)

### Bug Fixes

- Loosen typer dependency
  ([#39](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/39),
  [`ca9ea82`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/ca9ea8296ded25b64c3f871fac36cb9f348e6899))

### Documentation

- #37 - fix hallucinated Tolles-Lawson DOI and add Gnadt et al. reference
  ([#38](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/38),
  [`d2b46c8`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/d2b46c8c3b241fc5342bd68aa50509a3809fcbd2))

### Features

- Add validate_dataframe to build_feature_matrix
  ([#40](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/40),
  [`9b24221`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/9b24221b77bbe4dc98ea7101f32be65e43905776))


## v1.10.1 (2026-02-23)

### Bug Fixes

- Skip only __main__ in gen_ref_pages, convert __init__ to index pages
  ([#36](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/36),
  [`ee74735`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/ee74735f1905f3229588d00e82ffa7c2a214cdcd))

### Documentation

- #33 - auto-generate API reference pages with mkdocs-gen-files and literate-nav
  ([#36](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/36),
  [`ee74735`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/ee74735f1905f3229588d00e82ffa7c2a214cdcd))

- #33 - auto-generate API reference with mkdocs-gen-files and literate-nav
  ([#36](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/36),
  [`ee74735`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/ee74735f1905f3229588d00e82ffa7c2a214cdcd))

- #34 - flesh out README with badges, features, and API reference
  ([#35](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/35),
  [`103dc6b`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/103dc6b8fce92b40558017ae1be9986f1e28fe3b))


## v1.10.0 (2026-02-23)

### Bug Fixes

- Address PR review comments on #14
  ([#31](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/31),
  [`8dfab59`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8dfab59a7cae179c53d09aeb8f28a76035b00383))

### Documentation

- #22 - document reference_heading_deg normalisation behaviour
  ([#29](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/29),
  [`6744f0d`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/6744f0d836b4d933e85d071ccc72f1d1a44ad867))

### Features

- #14 - optional IMU angular rates for eddy-current term estimation
  ([#31](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/31),
  [`8dfab59`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/8dfab59a7cae179c53d09aeb8f28a76035b00383))


## v1.9.0 (2026-02-23)

### Features

- #13 - CLI integration & end-to-end integration tests
  ([#27](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/27),
  [`2a15ca9`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/2a15ca94ef57363bbed6e1f8d02a2361102a1af2))

### Refactoring

- Address PR #27 review comments
  ([#27](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/27),
  [`2a15ca9`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/2a15ca94ef57363bbed6e1f8d02a2361102a1af2))


## v1.8.0 (2026-02-22)

### Bug Fixes

- Move COL_TMI_COMPENSATED to correct alphabetical position in __all__
  ([#26](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/26),
  [`391f79b`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/391f79ba07816a43d2f878355dfb01753f097f7b))

### Features

- #12 - apply compensation to survey data
  ([#26](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/26),
  [`391f79b`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/391f79ba07816a43d2f878355dfb01753f097f7b))


## v1.7.0 (2026-02-22)

### Features

- #11 - FOM quality metrics
  ([#25](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/25),
  [`1d1fa38`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/1d1fa383eeac226dce8d034c538851afe0fd6806))

### Refactoring

- Use df.slice() for consistency with calibration module
  ([#25](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/25),
  [`1d1fa38`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/1d1fa383eeac226dce8d034c538851afe0fd6806))


## v1.6.0 (2026-02-22)

### Documentation

- Clarify CalibrationResult.residuals alignment in docstring
  ([#23](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/23),
  [`483c4cd`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/483c4cdb9ef4e9b6b67b5e3e57c8b10e00b1ad8e))

### Features

- #10 - Tolles-Lawson coefficient estimation
  ([#23](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/23),
  [`483c4cd`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/483c4cdb9ef4e9b6b67b5e3e57c8b10e00b1ad8e))


## v1.5.0 (2026-02-22)

### Bug Fixes

- Address PR #20 review comments
  ([#20](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/20),
  [`e7dd414`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/e7dd41423f6eae1ec7fa5af7690f7fe449ca3717))

### Features

- #9 - Form test segmentation
  ([#20](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/20),
  [`e7dd414`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/e7dd41423f6eae1ec7fa5af7690f7fe449ca3717))


## v1.4.0 (2026-02-22)

### Features

- Earth field baseline via IGRF
  ([#18](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/18),
  [`74a1c19`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/74a1c19ec048e3f5dd9be42dd3a1b2344155d440))


## v1.3.0 (2026-02-21)

### Chores

- Sync uv.lock to pyproject.toml version 1.2.0
  ([#16](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/16),
  [`f1b4710`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f1b471031d65f8de1e1dc7761e39169f2e9615fe))

### Continuous Integration

- Add typos.toml to allow ANE journal abbreviation
  ([#16](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/16),
  [`f1b4710`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f1b471031d65f8de1e1dc7761e39169f2e9615fe))

### Documentation

- Add Tolles-Lawson theory page
  ([#16](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/16),
  [`f1b4710`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f1b471031d65f8de1e1dc7761e39169f2e9615fe))

### Features

- Direction cosines & Tolles-Lawson feature matrix
  ([#16](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/16),
  [`f1b4710`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f1b471031d65f8de1e1dc7761e39169f2e9615fe))

### Testing

- Assert column order for term sets b and c
  ([#16](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/16),
  [`f1b4710`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f1b471031d65f8de1e1dc7761e39169f2e9615fe))


## v1.2.0 (2026-02-21)

### Bug Fixes

- Use COL_LAT and COL_LON constants in test helper
  ([#15](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/15),
  [`f895db7`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f895db7408674b083e5497c1ec82ee3b46333d41))

### Features

- Add polars, data schema, and pipeline config
  ([#15](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/15),
  [`f895db7`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f895db7408674b083e5497c1ec82ee3b46333d41))

- Data schema & polars integration
  ([#15](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/15),
  [`f895db7`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/f895db7408674b083e5497c1ec82ee3b46333d41))


## v1.1.1 (2026-02-20)

### Bug Fixes

- Add missing docs directory to fix documentation CI
  ([#5](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/5),
  [`2566cdd`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/2566cdd04ddd4685507449d522c8245eaefeebbf))


## v1.1.0 (2026-02-20)

### Bug Fixes

- Add `from None` to typer.Exit raise to satisfy ruff B904
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

- Remove enablement: true from configure-pages step
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

- Resolve duplicate mkdocstrings handler key and add build-system table
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

### Chores

- #1 - Linting ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

- Update uv.lock to reflect v1.0.0 version bump
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

### Features

- #1 - Add CLAUDE.md ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

- #1 - Add CLI entry-point stub
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

- #1 - Add Makefile with dev-friendly commands
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

- #1 - Add mkdocs config
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

- #1 - Add pyproject.toml
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

- #1 - Add virtual environment lockfile
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

- #1 - Github skeleton
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

- #1 - Simple unit/integration test stubs
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))

- #1 - Typos configuration
  ([#4](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/pull/4),
  [`919c63e`](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/commit/919c63e9dda995fe960d5a57d4986c274bea1a17))


## v1.0.0 (2026-02-20)

- Initial Release
