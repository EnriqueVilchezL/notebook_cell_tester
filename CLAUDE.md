# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`notebook-cell-tester` is a Python library for automatically grading University level engineering student code in Google Colab and Jupyter notebooks. Instructors drop a test cell into a notebook; the framework reads the *previous* cell's source from the IPython history, executes it, and displays a color-coded HTML results table.

## Setup & Commands

The project uses `uv` with Python 3.12.

```bash
# Install in editable mode
uv pip install -e .

# Build a distribution
python -m build
```

There is no test suite in this repository. Manual testing requires running inside a live Jupyter/Colab environment.

## Architecture

Everything lives in a single file: `src/notebook_cell_tester/tester.py`.

**Core classes:**

- `TestCase` — dataclass describing one test. `test_type` selects the strategy; `__post_init__` validates all required companion fields.
- `TestSection` — dataclass grouping a named list of `TestCase` objects into a visual section. Fields: `name: str`, `tests: List[TestCase]`.
- `TestResult` — dataclass returned by every test method: `test_name`, `passed`, `message`, `error`, `description`.
- `ColabTestFramework` — orchestrates the full workflow:
  1. `load_last_cell()` reads `In[-2]` from the IPython namespace (three fallback strategies) so it captures the student's cell, not the test cell.
  2. `_dispatch_test(test, code_loaded)` — private helper that routes a single `TestCase` to the right method; returns `None` when source is unavailable and the test requires it.
  3. `run_tests(tests)` calls `_dispatch_test` for each `TestCase` in a flat list; results stored in `self.results`.
  4. `run_sections(sections)` calls `_dispatch_test` per section; results stored grouped in `self.section_results` (list of `(name, List[TestResult])`) and flat in `self.results`.
  5. `display_results()` renders an HTML table via `IPython.display`. When `self.section_results` is non-empty, each section gets its own header and table; otherwise a single flat table is shown.

**Module-level utility:** `levenshtein_similarity(s1, s2)` — O(n·m) space-optimized DP, returns float in [0, 1]. Used only by `test_partial_output`.

---

## TestCase Fields

| Field | Type | Required for | Notes |
|-------|------|--------------|-------|
| `name` | `str` | all | Display name in results table |
| `test_type` | `str` | all | See dispatch table below |
| `function_name` | `str` | function tests, `type_check` (one of) | Function looked up from `get_ipython().user_ns` |
| `variable_name` | `str` | `variable`, `type_check` (one of) | Variable looked up from `get_ipython().user_ns` |
| `inputs` | `list` | function tests | Args passed to the function; defaults to `[]` |
| `stdin_input` | `str` | output tests | Simulates `input()` calls; lines separated by `\n` |
| `expected` | `Any` | most tests | Return value, output string, exception type, or target type |
| `similarity_threshold` | `float` | `partial_output` | Must be in `(0.0, 1.0]` |
| `tolerance` | `float` | `return` (optional) | If set, passes when `abs(got - expected) <= tolerance`; must be `>= 0` |
| `validator` | `Callable` | `variable` | `lambda value: bool` |
| `pattern` | `str` | `regex`, `not_regex`, `regex_output` | Python regex string |
| `description` | `str` | optional on all | Shown as italic subtitle under test name in the results table |
| `error_message` | `str` | optional on all | Custom failure message; use `{value}` placeholder in `variable` tests |

---

## Test Types and Dispatch

| `test_type` | Method called | What it checks |
|---|---|---|
| `return` | `test_function` | Function return value; supports `tolerance` for floats |
| `output` | `test_function` / `test_cell_output` | Exact stdout match |
| `exception` | `test_function` | Function raises the expected exception type |
| `regex` | `test_code_pattern` | Source code matches regex |
| `not_regex` | `test_code_pattern(negate=True)` | Source code does NOT match regex |
| `variable` | `test_variable` | Variable satisfies `validator` callable |
| `partial_output` | `test_partial_output` | stdout similarity ≥ `similarity_threshold` (Levenshtein) |
| `regex_output` | `test_regex_output` | stdout matches regex pattern |
| `contains_output` | `test_contains_output` | stdout contains `expected` as a substring |
| `type_check` | `test_type_check` | Return value or variable is `isinstance(value, expected)` |
| `multiline_output` | `test_multiline_output` | Every line of `expected` appears somewhere in stdout (order-independent) |

---

## Key Design Constraints

**Function lookup:** All function-calling methods look up the function from `get_ipython().user_ns`. The student's cell must have been executed before the test cell runs.

**Cell-level execution:** Tests without `function_name` re-execute `self.student_code` in an isolated `exec` namespace with stdin redirected. This avoids polluting or recursing into the test cell's namespace.

**Namespace-only tests:** `variable` and `type_check`-on-variable tests only need the IPython namespace. Both `run_tests` and `run_sections` run these even when `load_last_cell()` returns an empty string (e.g., when the student hasn't run their cell yet).

---

## Display Behavior

`display_results()` renders an HTML table with:

- **Summary banner** — gradient bar showing `passed/total (%)` and a congratulatory or warning message.
- **Section headers** — when `run_sections()` was used, each section renders a purple gradient header bar with the section name on the left and `n/total passed` badge on the right, followed by its own table.
- **Status column** — green `✓ PASS` or red `✗ FAIL` badge.
- **Test column** — test name, with `description` shown as a small italic subtitle if set.
- **Details column** — `result.message` (HTML-escaped, newlines converted to `<br>`).
- **Collapsible error** — when `result.error` is set, a `<details><summary>⚠ Show technical details</summary>` block hides the raw traceback by default. This keeps the table readable for students while preserving diagnostic info.

All user-visible strings in `result.message` are written in plain, student-friendly language. Raw Python exceptions never appear in `message`; they go into `result.error` only.
