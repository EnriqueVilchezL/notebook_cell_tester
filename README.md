# notebook-cell-tester

[![PyPI version](https://img.shields.io/pypi/v/notebook-cell-tester)](https://pypi.org/project/notebook-cell-tester/)
[![Python](https://img.shields.io/pypi/pyversions/notebook-cell-tester)](https://pypi.org/project/notebook-cell-tester/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automatic grading framework for student code in **Google Colab** and **Jupyter notebooks**.

Instructors drop a single test cell below the student's code cell. The framework reads the previous cell's source from IPython history, executes it, runs the configured tests, and renders a color-coded HTML results table — no extra setup required from students.

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Test types](#test-types)
  - [return — function return value](#return--function-return-value)
  - [output — exact stdout match](#output--exact-stdout-match)
  - [exception — expected exception](#exception--expected-exception)
  - [regex / not_regex — source code pattern](#regex--not_regex--source-code-pattern)
  - [variable — value with validator](#variable--value-with-validator)
  - [type_check — isinstance check](#type_check--isinstance-check)
  - [partial_output — fuzzy match](#partial_output--fuzzy-match)
  - [regex_output — stdout regex](#regex_output--stdout-regex)
  - [contains_output — stdout substring](#contains_output--stdout-substring)
  - [multiline_output — order-independent lines](#multiline_output--order-independent-lines)
- [TestCase field reference](#testcase-field-reference)
- [Results display](#results-display)
- [Common patterns](#common-patterns)
- [License](#license)

---

## Installation

```bash
pip install notebook-cell-tester
```

**Colab one-liner** — paste this at the top of the test cell if the package is not pre-installed in the environment:

```python
!pip install -q notebook-cell-tester
```

---

## Quick start

Place this cell immediately after the student's code cell in the notebook:

```python
from notebook_cell_tester import ColabTestFramework, TestCase

tester = ColabTestFramework()

tests = [
    TestCase(
        name="add_numbers returns correct sum",
        test_type="return",
        function_name="add_numbers",
        inputs=[3, 4],
        expected=7,
        description="add_numbers(3, 4) should return 7",
    ),
    TestCase(
        name="Prints greeting",
        test_type="output",
        stdin_input="Alice",
        expected="Hello, Alice!",
    ),
]

tester.run_tests(tests)
tester.display_results()
```

Running this cell produces a color-coded HTML table showing each test's pass/fail status, expected vs. actual values, and (when something crashes) a collapsible traceback that students can expand.

---

## How it works

1. `ColabTestFramework()` is instantiated in the **test cell** — the cell below the student's code.
2. `run_tests(tests)` calls `load_last_cell()`, which reads `In[-2]` from the IPython namespace to get the second-to-last cell (the student's code, not the test cell).
3. Each `TestCase` is dispatched to the appropriate internal method based on `test_type`.
4. `display_results()` renders an HTML table with pass/fail badges, result details, and collapsible error tracebacks.

Tests that only inspect the IPython namespace (`variable`, `type_check` on a variable) run even when the student hasn't executed their cell yet.

---

## Test types

### Summary

| `test_type` | Checks | Key fields |
|---|---|---|
| `return` | Function return value | `function_name`, `inputs`, `expected`, `tolerance` |
| `output` | Exact printed output | `stdin_input`, `expected`, optional `function_name` |
| `exception` | Function raises expected exception | `function_name`, `inputs`, `expected` |
| `regex` | Source code matches pattern | `pattern` |
| `not_regex` | Source code does NOT match pattern | `pattern` |
| `variable` | Variable satisfies validator | `variable_name`, `validator` |
| `type_check` | Value is correct type | `function_name` or `variable_name`, `expected` |
| `partial_output` | Output similarity ≥ threshold (Levenshtein) | `expected`, `similarity_threshold` |
| `regex_output` | Printed output matches regex | `pattern` |
| `contains_output` | Printed output contains substring | `expected` |
| `multiline_output` | Every expected line appears in output | `expected` |

---

### `return` — function return value

Calls `function_name(*inputs)` and checks that the return value equals `expected`.

```python
TestCase(
    name="Square of 5",
    test_type="return",
    function_name="square",
    inputs=[5],
    expected=25,
)
```

Use `tolerance` for floating-point comparisons (`abs(result - expected) <= tolerance`):

```python
TestCase(
    name="Average of [1, 2, 3]",
    test_type="return",
    function_name="average",
    inputs=[[1, 2, 3]],
    expected=2.0,
    tolerance=1e-9,
)
```

---

### `output` — exact stdout match

Captures everything printed to stdout and compares it to `expected` (after stripping leading/trailing whitespace).

```python
# Test the whole cell — no function_name needed
TestCase(
    name="Greeting output",
    test_type="output",
    stdin_input="Alice",
    expected="Hello, Alice!",
)

# Test a specific function's printed output
TestCase(
    name="print_square(4)",
    test_type="output",
    function_name="print_square",
    inputs=[4],
    expected="16",
)
```

Supply `stdin_input` to simulate one or more `input()` calls. Separate multiple lines with `\n`.

---

### `exception` — expected exception

Passes when `function_name(*inputs)` raises exactly the expected exception type. Fails if it returns normally or raises a different exception.

```python
TestCase(
    name="Division by zero raises ValueError",
    test_type="exception",
    function_name="safe_divide",
    inputs=[10, 0],
    expected=ValueError,
)
```

---

### `regex` / `not_regex` — source code pattern

Searches the student's **source code** for a Python regex. Useful for requiring or forbidding specific language constructs without running the code.

```python
# Require a for loop
TestCase(
    name="Uses a for loop",
    test_type="regex",
    pattern=r"\bfor\b",
    description="Your solution must use a for loop.",
)

# Forbid the built-in sum()
TestCase(
    name="Does not call sum()",
    test_type="not_regex",
    pattern=r"\bsum\s*\(",
    error_message="Implement the summation manually — do not use sum().",
)
```

Both types use `re.MULTILINE | re.DOTALL` flags.

---

### `variable` — value with validator

Reads a variable from the IPython namespace and passes it to a `validator` callable. The test passes when `validator(value)` returns `True`.

```python
TestCase(
    name="score is between 0 and 100",
    test_type="variable",
    variable_name="score",
    validator=lambda v: 0 <= v <= 100,
    error_message="score = {value}, but it must be between 0 and 100.",
)
```

Use `{value}` as a placeholder in `error_message` — it is replaced with the actual variable value when the test fails.

---

### `type_check` — isinstance check

Passes when `isinstance(value, expected)` is `True`. Works with a function's return value or a named variable.

```python
# Check a function's return type
TestCase(
    name="get_scores() returns a list",
    test_type="type_check",
    function_name="get_scores",
    inputs=[],
    expected=list,
)

# Check a variable's type
TestCase(
    name="result is a float",
    test_type="type_check",
    variable_name="result",
    expected=float,
)

# Accept multiple types using a tuple
TestCase(
    name="index is int or float",
    test_type="type_check",
    variable_name="index",
    expected=(int, float),
)
```

---

### `partial_output` — fuzzy match

Passes when the Levenshtein similarity between the actual and expected output is ≥ `similarity_threshold`. Useful when minor formatting differences (extra spaces, punctuation) should still receive credit.

```python
TestCase(
    name="Greeting (fuzzy match)",
    test_type="partial_output",
    stdin_input="Bob",
    expected="Hello, Bob! Welcome.",
    similarity_threshold=0.85,
    description="At least 85% similar to the expected greeting.",
)
```

`similarity_threshold` must be a `float` in `(0.0, 1.0]`. A value of `1.0` requires an exact match; `0.8` allows up to ~20% edit distance.

---

### `regex_output` — stdout regex

Passes when `re.search(pattern, output)` finds a match in the printed output. Good for checking output format without requiring an exact string.

```python
TestCase(
    name="Output contains a decimal number",
    test_type="regex_output",
    pattern=r"\d+\.\d+",
    error_message="Expected a decimal number in the output.",
)

# With a specific function
TestCase(
    name="format_price() output includes currency symbol",
    test_type="regex_output",
    function_name="format_price",
    inputs=[9.99],
    pattern=r"\$\d+\.\d{2}",
)
```

---

### `contains_output` — stdout substring

Passes when `expected` is a substring of the printed output (case-sensitive).

```python
TestCase(
    name="Output mentions 'Error'",
    test_type="contains_output",
    expected="Error",
    description="The program must print an error message.",
)
```

---

### `multiline_output` — order-independent lines

Splits `expected` on newlines and verifies that every non-empty line appears somewhere in the output. Order does not matter — useful when students may print items in any sequence.

```python
TestCase(
    name="Prints name and age",
    test_type="multiline_output",
    expected="Name: Alice\nAge: 30",
    description="Both lines must appear in the output (any order).",
)
```

---

## TestCase field reference

| Field | Type | Required for | Notes |
|---|---|---|---|
| `name` | `str` | all | Display name shown in the results table |
| `test_type` | `str` | all | One of the test type strings above |
| `function_name` | `str` | function tests, `type_check` (one of) | Looked up from `get_ipython().user_ns` |
| `variable_name` | `str` | `variable`, `type_check` (one of) | Looked up from `get_ipython().user_ns` |
| `inputs` | `list` | function tests | Positional arguments; defaults to `[]` |
| `stdin_input` | `str` | output tests | Simulates `input()` calls; lines separated by `\n` |
| `expected` | `Any` | most tests | Return value, output string, exception type, or target type |
| `similarity_threshold` | `float` | `partial_output` | Must be in `(0.0, 1.0]` |
| `tolerance` | `float` | `return` (optional) | Passes when `abs(got - expected) <= tolerance`; must be `>= 0` |
| `validator` | `Callable` | `variable` | `lambda value: bool` |
| `pattern` | `str` | `regex`, `not_regex`, `regex_output` | Python regex string |
| `description` | `str` | optional on all | Shown as an italic subtitle under the test name |
| `error_message` | `str` | optional on all | Custom failure message; use `{value}` placeholder in `variable` tests |

---

## Results display

`display_results()` renders an HTML table inside the notebook:

- **Summary banner** — gradient bar showing `passed / total (%)` with a congratulatory message when all tests pass.
- **Status column** — green `✓ PASS` or red `✗ FAIL` badge per test row.
- **Test column** — test name, with `description` shown as a small italic subtitle when set.
- **Details column** — human-readable message (expected vs. actual values, similarity percentages, missing lines, etc.).
- **Collapsible traceback** — when an unhandled exception occurs during execution, a `▶ Show technical details` disclosure element hides the raw traceback by default, keeping the table readable for beginners.

---

## Common patterns

### Test an entire cell (no function)

Omit `function_name`. The framework re-executes the student's full cell source in an isolated namespace with stdin redirected.

```python
TestCase(
    name="Reads two numbers and prints their sum",
    test_type="output",
    stdin_input="3\n7",
    expected="10",
)
```

### Combine structural and functional tests

```python
tests = [
    # 1. Check the algorithm used
    TestCase(
        name="Uses recursion",
        test_type="regex",
        pattern=r"def\s+factorial.*?\n(?:.*\n)*?.*\bfactorial\s*\(",
        description="factorial() must call itself recursively.",
    ),
    # 2. Check correctness
    TestCase(
        name="factorial(5) == 120",
        test_type="return",
        function_name="factorial",
        inputs=[5],
        expected=120,
    ),
    # 3. Check error handling
    TestCase(
        name="Raises ValueError for negative input",
        test_type="exception",
        function_name="factorial",
        inputs=[-1],
        expected=ValueError,
    ),
]
```

### Check a variable's value and type together

```python
tests = [
    TestCase(
        name="result is a float",
        test_type="type_check",
        variable_name="result",
        expected=float,
    ),
    TestCase(
        name="result is positive",
        test_type="variable",
        variable_name="result",
        validator=lambda v: v > 0,
        error_message="result = {value}, but it must be greater than 0.",
    ),
]
```

### Flexible output with `partial_output`

When students may phrase output slightly differently, use `partial_output` to award credit for near-correct answers:

```python
TestCase(
    name="Farewell message",
    test_type="partial_output",
    stdin_input="Maria",
    expected="Goodbye, Maria! See you soon.",
    similarity_threshold=0.80,
    description="Your farewell message must be at least 80% similar to the expected output.",
)
```

---

## License

MIT — see [LICENSE](LICENSE).
