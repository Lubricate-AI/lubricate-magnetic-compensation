# RLSState Mutability Contract Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clarify the `RLSState` mutability contract by making it `frozen=True` and updating its docstring to reflect the functional/return-by-value update pattern.

**Architecture:** `RLSState` is a `@dataclass` in `lmc/rls.py` whose public update API (`update_rls`, `update_rls_batch`) never mutates input state — each call returns a new object. Adding `frozen=True` prevents accidental field reassignment (numpy array *content* can still be mutated, but that is a separate concern). The docstring must be updated to match.

**Tech Stack:** Python 3.12, `dataclasses` stdlib, `numpy`

---

### Task 1: Add a failing test for frozen `RLSState`

**Files:**
- Modify: `tests/unit/test_rls.py`

**Step 1: Write the failing test**

Add the following test at the end of the file (after `test_rls_symbols_exported_from_package`):

```python
def test_rlsstate_is_frozen() -> None:
    """RLSState must be immutable — field reassignment must raise an error."""
    state = _make_state(3)
    with pytest.raises((AttributeError, TypeError)):
        state.n_samples = 99  # type: ignore[misc]
```

**Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/unit/test_rls.py::test_rlsstate_is_frozen -v
```

Expected: FAIL — `state.n_samples = 99` succeeds without raising because `RLSState` is not yet frozen.

---

### Task 2: Make `RLSState` frozen and update its docstring

**Files:**
- Modify: `lmc/rls.py:17-36`

**Step 3: Apply the change**

Replace:

```python
@dataclass
class RLSState:
    """Mutable state for Recursive Least-Squares online coefficient updating.
```

With:

```python
@dataclass(frozen=True)
class RLSState:
    """Immutable value object for Recursive Least-Squares online coefficient updating.

    Each call to :func:`update_rls` or :func:`update_rls_batch` returns a *new*
    ``RLSState`` without modifying the input.  Field reassignment is prohibited
    by ``frozen=True``; however, the *content* of numpy array fields
    (``coefficients``, ``covariance``) can still be mutated in-place — avoid
    doing so.
```

**Step 4: Run the full RLS test suite**

```bash
uv run pytest tests/unit/test_rls.py -v
```

Expected: ALL PASS (the new frozen test now passes; no existing test does in-place field reassignment).

**Step 5: Run the full test suite to catch any regressions**

```bash
uv run pytest -v
```

Expected: ALL PASS.

**Step 6: Run the linter**

```bash
make lint
```

Expected: no errors.

**Step 7: Commit**

```bash
git add lmc/rls.py tests/unit/test_rls.py
git commit -m "refactor: make RLSState frozen and clarify immutable update contract

Resolves #68.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
