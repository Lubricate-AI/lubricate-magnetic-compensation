# CLAUDE.md

## Project Overview
A repo for calculating magnetic compensation coefficients using the Tolles-Lawson model.

## Repository Conventions

### Branch Naming
- Feature branches: `<issue-number>-brief-description` (e.g., `7-add-agents-md`)
- Main branch: `main`
- Always create PR branches from `main`

### Commit Messages
- Follow conventional commits: `<type>: <description>`
- Types: `feat`, `fix`, `docs`, `ci`, `refactor`, `test`, `chore`
- Auto-include attribution footer when making commits

### GitHub Repository
- GitHub repo: `Lubricate-AI/lubricate-magnetic-compensation`
- Always use `--repo Lubricate-AI/lubricate-magnetic-compensation` when running `gh` CLI commands for issues, PRs, etc.

### GitHub Issues
- When creating issues, structure them using the templates in `.github/ISSUE_TEMPLATE/`
- Include Overview (brief description)
- Include Details (expected/actual behavior, proposed solution)
- Include Context (type, priority, affected component)
- For bugs: provide steps to reproduce

### Pull Request Process
- When creating PRs, structure the body using `.github/PULL_REQUEST_TEMPLATE.md`
- Include INFO section (what, key changes, breaking changes, testing, TODOs)
- Include REFERENCES section (related issues, PRs, docs, external resources)
- CI runs linting and tests automatically

## Development Environment

### Package Manager
- Use `uv` (NOT pip) for all dependency management
- Run `make install` to install/update dependencies

### Python Version
- Python 3.12+ required
- Virtual environments are managed by `uv`

### Available Make Commands
- `make install` - Install/update dependencies
- `make lint` - Run all linting (ruff, typos, yamllint)
- `make type-checking` - Run pyright
- `make format` - Auto-format code
- `make test` - Run pytest

## Versioning

- `python-semantic-release` automatically bumps the version on merge to `main` based on commit types
- `refactor`, `fix`, `perf` → patch bump; `feat` → minor bump; `BREAKING CHANGE` → major bump
- ❌ DON'T manually bump the version in `pyproject.toml` or `lmc/__init__.py`

## Common Pitfalls & Anti-patterns

### Package Management
❌ DON'T use `pip install` directly
✅ DO use `uv add` or update `pyproject.toml` dependencies

### Dependencies
❌ DON'T add unnecessary dependencies
✅ DO keep project dependencies minimal

## Code Quality Standards
- Python code must pass ruff linting for reusable utilities
- Type hints encouraged for reusable utilities
