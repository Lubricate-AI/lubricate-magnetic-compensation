# CHANGELOG

<!-- version list -->

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
