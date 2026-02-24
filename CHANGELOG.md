# CHANGELOG

<!-- version list -->

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
