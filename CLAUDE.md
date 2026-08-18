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
- `TestResult` — dataclass returned by every test method: `test_name`, `passed`, `message`, `error`, `description`, `skipped`.
- `ColabTestFramework` — orchestrates the full workflow:
  1. `load_last_cell(tests)` finds the student's cell. It does **not** use `In[-2]`: `In` is execution order, not notebook order, so `In[-2]` points at the wrong cell whenever the student re-runs the test cell, runs a scratch cell, or jumps around the notebook. Two strategies, in order:
     - **Document order (Colab, exact).** `_document_cells()` asks the Colab frontend for the notebook via `google.colab._message.blocking_request('get_ipynb')`; `_student_cell_from_document()` locates the running test cell by source and collects *every* code cell above it, joined in document order, stopping at the previous test cell. One exercise therefore means "the run of code cells between two test cells" — students are told they may split their work across cells, and half a solution is not gradeable on its own because the names it needs are defined in the other half. Trivial cells (blank, comments-only, `!shell`/`%magic`) are left out; an empty range produces the "no code yet" warning. Execution order is irrelevant to this path.
     - **Execution history (Jupyter, heuristic).** Scans backwards over `In`, discarding test cells (`_is_test_cell`) and trivial cells (`_is_trivial_cell` — blank, comments-only, `!shell`/`%magic`); prefers the most recent cell defining a name the tests reference (`_cell_defines`); then `_anchor_from_previous_run()`, which re-uses the cell graded the last time this same test cell ran (guarded by a `levenshtein_similarity` check so an edited-and-re-run solution still wins); then the most recent remaining cell.
  2. `_dispatch_test(test, code_loaded)` — private helper that routes a single `TestCase` to the right method. When source is unavailable and the test needs it, the test comes back marked `skipped=True` rather than being dropped, so the size of the results table never silently changes between runs.
  3. `run_tests(tests)` calls `_dispatch_test` for each `TestCase` in a flat list; results stored in `self.results`.
  4. `run_sections(sections)` calls `_dispatch_test` per section; results stored grouped in `self.section_results` (list of `(name, List[TestResult])`) and flat in `self.results`.
  5. `display_results()` renders an HTML table via `IPython.display`. When `self.section_results` is non-empty, each section gets its own header and table; otherwise a single flat table is shown.

**Module-level utilities:**

- `levenshtein_similarity(s1, s2)` — O(n·m) space-optimized DP, returns float in [0, 1]. Used only by `test_partial_output`.
- `_shadowed_builtin_names()` — returns the builtin names the notebook namespace has rebound to **non-callables** (see *Session poisoning* below). Callable rebindings are ignored because IPython and Colab legitimately replace `open` and `exit` with their own functions.
- `_shadowing_hint()` — explanatory suffix appended to `TestCase.__post_init__` errors, since a poisoned `str`/`list`/`int` makes the instructor's own `TestCase(..., expected=str)` raise before any framework code runs.
- `StdinExhausted(EOFError)` — raised when student code calls `input()` more times than `stdin_input` supplies.
- `ExecutionTimeout(BaseException)` / `OutputTooLarge(BaseException)` / `_CappedOutput` — the runaway-code guards described below.
- `_STUDENT_FAILURES` — the tuple every test method catches: `Exception`, `SystemExit`, and the two guards. `KeyboardInterrupt` is deliberately excluded so a student can always stop a cell.

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
| `success_message` | `str` | optional on all | Custom message shown when the test passes; replaces the default pass message |

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

**Cell-level execution:** Tests without `function_name` re-execute `self.student_code` in an isolated `exec` namespace with stdin redirected. This avoids polluting or recursing into the test cell's namespace. Because `student_code` is the whole exercise block, a scratch/debug cell inside that block is executed too: harmless for `contains_output`, `regex_output` and `multiline_output`, but its printed output does join the comparison for strict `output` and `partial_output`, and any `input()` it calls consumes lines meant for the solution.

**Namespace-only tests:** `variable` and `type_check`-on-variable tests only need the IPython namespace. Both `run_tests` and `run_sections` run these even when `load_last_cell()` returns an empty string (e.g., when the student hasn't run their cell yet).

**Session poisoning:** a student who writes `str = "Ana"` or `list = [1, 2]` in *any* cell rebinds that name in the shared IPython user namespace for the rest of the session. Functions defined in a notebook resolve their globals from that same namespace, so correct-looking code then fails with `'str' object is not callable`, and only a runtime restart clears it. `run_tests`/`run_sections` call `_detect_shadowed_builtins()` up front, store the result in `self.shadowed_builtins`, surface it as a banner in `display_results()`, and let `_friendly_error()` translate the resulting `TypeError`/`NameError` into an explanation instead of a traceback.

**Argument isolation:** `_safe_args()` deep-copies `inputs` before every call. `TestCase` objects live in the notebook and survive re-runs of the test cell, so a student function that mutates its argument in place (`lst.sort()`, `pila.pop()`) would otherwise make the same test pass on the first run and fail on the second.

**Runaway code:** a `while True` in student code used to hang the test cell forever, and if it printed, grow the captured buffer until the Colab runtime ran out of memory and died. `_time_limited()` (SIGALRM, repeating so a bare `except` cannot swallow it) stops execution after `time_limit_seconds`, and `_CappedOutput` raises after `output_limit_chars`. Both are class attributes on `ColabTestFramework`, overridable per instance. `ExecutionTimeout` and `OutputTooLarge` derive from `BaseException` for that reason, so every test method catches `_STUDENT_FAILURES` rather than `Exception`.

**SystemExit:** `exit()` / `sys.exit()` in a student cell is treated as a normal end-of-program — the output printed so far is graded. Previously `SystemExit` escaped every handler and killed the whole test cell with no results table.

**Notebook magics:** `_exec_student_code()` compiles the cell, and only on `SyntaxError` retries with `_strip_notebook_magics()`, which comments out `!shell` / `%magic` lines (preserving line numbers). A solution cell opening with `!pip install pandas` therefore still runs; a genuine syntax error still reports the original error.

**Blaming the right thing:** `_looks_like_shadowing()` gates the shadowed-builtin explanation on error shapes a shadowed name actually produces ("object is not callable/subscriptable", or a `NameError` naming one). Otherwise a student whose real bug is `"a" + 1` would be told to rename variables and restart. Likewise `_implicated_shadowed_builtins()` limits the banner to names the student's code actually *calls* — `sum = 0` is the textbook accumulator and must not put an alarm above a perfect score — or to a run where `shadowing_suspected` was set.

**Instructor mistakes fail loudly:** `TestCase.__post_init__` rejects a non-exception `expected` on `exception` tests, a non-list `inputs`, a missing `expected` on `output`/`partial_output`, a non-callable `validator`, and an uncompilable `pattern`. These used to reach the student as "Your code produced an error".

**Stdin exhaustion:** `_stdin_redirected` raises `StdinExhausted` once the supplied input runs out, instead of returning `''` forever — which used to turn "your program asks for more input than expected" into a hang or an unrelated `ValueError` deep inside student code. For cell-level tests `_exec_student_code()` catches it and returns `stopped_early=True`: the output printed up to that point is still graded, and `_EARLY_STOP_NOTE` is appended to the message only if the test then fails. A program that prints everything correctly and ends with `input("Presione Enter")` therefore still passes. Function-level tests let it propagate to `_friendly_error()`.

---

## Display Behavior

`display_results()` renders an HTML table with:

- **Summary banner** — gradient bar showing `passed/total (%)` and a congratulatory or warning message.
- **Section headers** — when `run_sections()` was used, each section renders a purple gradient header bar with the section name on the left and `n/total passed` badge on the right, followed by its own table.
- **Notice banners** — amber blocks under the summary for a poisoned namespace (`self.shadowed_builtins`) and for code-discovery problems (`self.warnings`).
- **Status column** — green `✓ PASS`, red `✗ FAIL`, or amber `⚠ NOT RUN` badge (the last for `result.skipped`).
- **Test column** — test name, with `description` shown as a small italic subtitle if set.
- **Details column** — `result.message` (HTML-escaped, newlines converted to `<br>`).
- **Collapsible error** — when `result.error` is set, a `<details><summary>⚠ Show technical details</summary>` block hides the raw traceback by default. This keeps the table readable for students while preserving diagnostic info.

All user-visible strings in `result.message` are written in plain, student-friendly language. Raw Python exceptions never appear in `message`; they go into `result.error` only.
