"""Colab Automated Testing Framework.

A comprehensive testing framework for automatically grading student code in Google Colab
and Jupyter notebooks. Supports testing entire cells, specific functions, code patterns,
and variable validations with beautiful color-coded result tables.

Example:
    Basic usage of the testing framework::

        tester = ColabTestFramework()

        tests = [
            TestCase(
                name="Addition test",
                test_type="return",
                function_name="add_numbers",
                inputs=[2, 3],
                expected=5
            )
        ]

        tester.run_tests(tests)
        tester.display_results()

Attributes:
    Module constants and global variables (none in this module).
"""

import re
import sys
import io
import copy
import builtins
import html as html_module
from contextlib import redirect_stdout, redirect_stderr, contextmanager, nullcontext
from typing import List, Dict, Any, Callable, Optional, Sequence
from dataclasses import dataclass, field
from IPython.display import HTML, display
from IPython import get_ipython
import traceback


# Builtin names students commonly reuse as ordinary variables.  Once a student
# writes e.g. ``str = input(...)`` or ``list = [1, 2]`` in *any* cell, that name
# lives in the shared IPython user namespace for the rest of the session.  Every
# later call to ``str(...)`` then raises "'str' object is not callable" — even
# from inside a function whose source is perfectly correct, because a function
# defined in a notebook resolves its globals from that same shared namespace.
# Restarting the runtime is the only cure, which is exactly why the failure
# looks like a corrupted package instead of a shadowed name.
_SHADOWABLE_BUILTINS = (
    'input', 'print', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set',
    'tuple', 'sum', 'min', 'max', 'len', 'type', 'sorted', 'reversed', 'abs',
    'round', 'range', 'map', 'filter', 'zip', 'open', 'id', 'next', 'iter',
    'all', 'any', 'bytes', 'format', 'vars', 'dir', 'hash', 'pow', 'divmod',
)

# Substrings that mark a cell as a *test* cell rather than a student cell.
_TEST_CELL_MARKERS = (
    'ColabTestFramework', 'notebook_cell_tester', 'run_tests(', 'run_sections(',
    'display_results(', 'TestCase(', 'TestSection(',
)


# A runaway ``while True`` in student code used to hang the test cell forever and,
# if it printed, grow the captured output until the Colab runtime ran out of memory
# and died. Both limits turn that into an ordinary failed test with an explanation.
_OUTPUT_LIMIT_CHARS = 1_000_000
_TIME_LIMIT_SECONDS = 15.0


class StdinExhausted(EOFError):
    """Raised when student code calls ``input()`` more times than the test supplies.

    Subclasses :class:`EOFError` so that student code written to read until
    end-of-input keeps behaving the way it would outside the notebook.
    """


class ExecutionTimeout(BaseException):
    """Raised when student code runs past :data:`_TIME_LIMIT_SECONDS`.

    Deliberately not an :class:`Exception`, so a blanket ``except Exception``
    inside student code cannot swallow it and keep looping.
    """


class OutputTooLarge(BaseException):
    """Raised when student code prints past :data:`_OUTPUT_LIMIT_CHARS`.

    Also outside :class:`Exception` for the same reason: the runaway loop that
    triggers it is usually wrapped in a bare ``except``.
    """


class _CappedOutput(io.StringIO):
    """stdout buffer that refuses to grow without bound."""

    def __init__(self, limit: int = _OUTPUT_LIMIT_CHARS):
        super().__init__()
        self._limit = limit
        self._written = 0

    def write(self, text):
        self._written += len(text)
        if self._written > self._limit:
            raise OutputTooLarge(
                f"student code printed more than {self._limit} characters"
            )
        return super().write(text)


# Everything a test may raise on the student's behalf. KeyboardInterrupt is left
# out on purpose so a student can always stop a cell themselves.
_STUDENT_FAILURES = (Exception, SystemExit, ExecutionTimeout, OutputTooLarge)


def _shadowed_builtin_names() -> List[str]:
    """Return builtin names the notebook namespace currently rebinds to non-callables.

    A student who writes ``str = "Ana"`` or ``list = [1, 2]`` in any cell poisons the
    shared IPython namespace for the rest of the session: every later ``str(...)``
    raises "'str' object is not callable", including inside functions whose source is
    perfectly correct, and including the instructor's own test cell. Only non-callable
    rebindings are reported — IPython and Colab legitimately replace some builtins
    (``open``, ``exit``) with callables of their own.
    """
    try:
        ipython = get_ipython()
        if ipython is None:
            return []
        ns = ipython.user_ns
    except Exception:
        return []

    shadowed = []
    for name in _SHADOWABLE_BUILTINS:
        if name not in ns:
            continue
        value = ns[name]
        if value is getattr(builtins, name, None):
            continue
        if not callable(value):
            shadowed.append(name)
    return shadowed


def _shadowing_hint() -> str:
    """Return an explanatory suffix for errors caused by a poisoned namespace."""
    shadowed = _shadowed_builtin_names()
    if not shadowed:
        return ""
    names = ', '.join(f"'{n}'" for n in shadowed)
    single = len(shadowed) == 1
    noun = "variable" if single else "variables"
    verb = "hides" if single else "hide"
    return (
        f"\n\nThis is almost certainly caused by the {noun} {names} in this "
        f"notebook, which {verb} Python's built-in functions of the same name. "
        f"Rename them, then restart the runtime (Runtime -> Restart session) and "
        f"run every cell again from the top."
    )


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute the Levenshtein similarity ratio between two strings.

    Similarity is defined as::

        1 - (edit_distance / max(len(s1), len(s2)))

    so identical strings yield 1.0 and completely different strings of the
    same length yield 0.0.  Both strings are compared after stripping leading
    and trailing whitespace.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        A float in [0.0, 1.0] representing the similarity ratio.

    Examples:
        >>> levenshtein_similarity("hello", "hello")
        1.0
        >>> levenshtein_similarity("kitten", "sitting")
        0.5384615384615384
        >>> levenshtein_similarity("", "")
        1.0
    """
    s1 = s1.strip()
    s2 = s2.strip()

    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)

    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0

    # Standard DP Levenshtein — O(len1 * len2) time, O(len2) space
    prev = list(range(len2 + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i] + [0] * len2
        for j, c2 in enumerate(s2, 1):
            if c1 == c2:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    distance = prev[len2]
    return 1.0 - distance / max(len1, len2)



@dataclass
class TestCase:
    """A test case for validating student code.

    Args:
        name: Display name for the test shown in the results table.
        test_type: Type of test to perform. Options are:

            - ``'output'``: Test printed output (stdout) — exact match.
            - ``'return'``: Test function return value.
            - ``'exception'``: Test if function raises expected exception.
            - ``'regex'``: Test if *source code* matches a regex pattern.
            - ``'not_regex'``: Test if *source code* does NOT match a regex pattern.
            - ``'variable'``: Test variable value using a validator function.
            - ``'partial_output'``: Test printed output via Levenshtein similarity.
              Passes when ``similarity >= similarity_threshold``.
            - ``'regex_output'``: Test that printed output matches a regex pattern.
            - ``'contains_output'``: Test that printed output contains a substring.
            - ``'type_check'``: Test that a return value or variable is a specific type.
            - ``'multiline_output'``: Test that every expected line appears in the output
              (order-independent).

        function_name: Name of the function to test. If None, tests entire cell
            execution. Required for function-level tests.
        variable_name: Name of the variable to validate. Required when
            ``test_type='variable'`` or ``test_type='type_check'`` (variable variant).
        inputs: List of arguments to pass to the function.
        stdin_input: String to provide as standard input (simulates ``input()``).
            Multiple lines separated by ``'\\n'``.
        expected: Expected value for comparison:

            - ``'return'``: Expected return value.
            - ``'output'`` / ``'partial_output'`` / ``'contains_output'`` /
              ``'multiline_output'``: Expected printed output string.
            - ``'exception'``: Expected exception type (e.g. ``ValueError``).
            - ``'variable'``: Optional, used in error messages.
            - ``'type_check'``: The expected type (e.g. ``int``, ``list``), or a
              tuple of types.

        similarity_threshold: Required for ``'partial_output'``. Float in ``(0.0, 1.0]``
            representing the minimum Levenshtein similarity ratio to pass.
        tolerance: Optional for ``'return'`` tests. If set, the test passes when
            ``abs(result - expected) <= tolerance``. Useful for floating-point comparisons.
            Must be ``>= 0``.
        validator: Callable that takes the variable value and returns ``bool``.
            Required when ``test_type='variable'``.
        pattern: Regex pattern. Required for ``'regex'``, ``'not_regex'``, and
            ``'regex_output'`` tests.
        description: Short description of what this test checks. Shown as a subtitle
            in the results table so students understand the purpose of each test.
        error_message: Custom message shown when the test fails.
            For variable tests, use ``{value}`` as a placeholder.
        success_message: Optional custom message shown when the test passes.
            When set, replaces the default pass message so students see this
            text instead of the technical expected/got details.

    Examples:
        Regex pattern in output::

            TestCase(
                name="Output contains a float",
                test_type="regex_output",
                pattern=r"\\d+\\.\\d+",
                error_message="Expected a float value in the output"
            )

        Fuzzy output match::

            TestCase(
                name="Greet user (fuzzy)",
                test_type="partial_output",
                stdin_input="Alice",
                expected="Hello, Alice!",
                similarity_threshold=0.8
            )

        Function return value::

            TestCase(
                name="Addition",
                test_type="return",
                function_name="add_numbers",
                inputs=[2, 3],
                expected=5
            )

        Float comparison with tolerance::

            TestCase(
                name="Average result",
                test_type="return",
                function_name="average",
                inputs=[[1, 2, 3]],
                expected=2.0,
                tolerance=1e-9
            )

        Type check::

            TestCase(
                name="Returns a list",
                test_type="type_check",
                function_name="get_scores",
                inputs=[],
                expected=list,
                description="get_scores() must return a list"
            )

        Multiline output (order-independent)::

            TestCase(
                name="Prints name and age",
                test_type="multiline_output",
                expected="Name: Alice\\nAge: 30",
                description="Both lines must appear in the output"
            )
    """
    name: str
    test_type: str
    function_name: Optional[str] = None
    variable_name: Optional[str] = None
    inputs: Optional[List[Any]] = None
    stdin_input: Optional[str] = None
    expected: Any = None
    similarity_threshold: Optional[float] = None
    tolerance: Optional[float] = None
    validator: Optional[Callable] = None
    pattern: Optional[str] = None
    description: str = ""
    error_message: str = ""
    success_message: str = ""

    def __post_init__(self):
        """Validate fields and apply defaults."""
        if self.inputs is None:
            self.inputs = []
        if self.test_type == 'partial_output':
            if self.similarity_threshold is None:
                raise ValueError(
                    f"TestCase '{self.name}': 'similarity_threshold' is required "
                    "for test_type='partial_output'."
                )
            if not (0.0 < self.similarity_threshold <= 1.0):
                raise ValueError(
                    f"TestCase '{self.name}': 'similarity_threshold' must be in "
                    f"(0.0, 1.0], got {self.similarity_threshold}."
                )
        if self.test_type in ('regex', 'not_regex', 'regex_output') and self.pattern is None:
            raise ValueError(
                f"TestCase '{self.name}': 'pattern' is required for "
                f"test_type='{self.test_type}'."
            )
        if self.test_type == 'contains_output' and self.expected is None:
            raise ValueError(
                f"TestCase '{self.name}': 'expected' is required for "
                "test_type='contains_output'."
            )
        if self.test_type == 'multiline_output' and self.expected is None:
            raise ValueError(
                f"TestCase '{self.name}': 'expected' is required for "
                "test_type='multiline_output'."
            )
        if self.test_type == 'type_check':
            if self.function_name is None and self.variable_name is None:
                raise ValueError(
                    f"TestCase '{self.name}': 'type_check' requires either "
                    "'function_name' or 'variable_name'."
                )
            if self.expected is None:
                raise ValueError(
                    f"TestCase '{self.name}': 'expected' (the target type) is required "
                    "for test_type='type_check'."
                )
            valid = (
                isinstance(self.expected, type)
                or (
                    isinstance(self.expected, tuple)
                    and all(isinstance(t, type) for t in self.expected)
                )
            )
            if not valid:
                raise ValueError(
                    f"TestCase '{self.name}': 'expected' for 'type_check' must be a "
                    f"type or tuple of types, got {self.expected!r}."
                    + _shadowing_hint()
                )
        if self.test_type == 'exception':
            valid_exc = (
                isinstance(self.expected, type)
                and issubclass(self.expected, BaseException)
            )
            if not valid_exc:
                raise ValueError(
                    f"TestCase '{self.name}': 'expected' for 'exception' must be an "
                    f"exception class such as ValueError, got {self.expected!r}."
                    + _shadowing_hint()
                )
        if self.inputs is not None and not isinstance(self.inputs, (list, tuple)):
            raise ValueError(
                f"TestCase '{self.name}': 'inputs' must be a list of arguments, "
                f"got {self.inputs!r}. Use inputs=[{self.inputs!r}] for a single argument."
            )
        if self.test_type in ('output', 'partial_output') and self.expected is None:
            raise ValueError(
                f"TestCase '{self.name}': 'expected' is required for "
                f"test_type='{self.test_type}'."
            )
        if self.test_type == 'variable' and not callable(self.validator):
            raise ValueError(
                f"TestCase '{self.name}': 'validator' must be a function for "
                f"test_type='variable', got {self.validator!r}."
            )
        if self.pattern is not None:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(
                    f"TestCase '{self.name}': 'pattern' is not a valid regular "
                    f"expression ({exc})."
                ) from None
        if self.tolerance is not None:
            if self.test_type != 'return':
                raise ValueError(
                    f"TestCase '{self.name}': 'tolerance' is only valid for "
                    "test_type='return'."
                )
            if self.tolerance < 0:
                raise ValueError(
                    f"TestCase '{self.name}': 'tolerance' must be >= 0, "
                    f"got {self.tolerance}."
                )


@dataclass
class TestSection:
    """A named group of test cases rendered as a separate table section.

    Args:
        name: Section heading displayed above the table.
        tests: List of TestCase objects belonging to this section.

    Examples:
        Grouping tests by topic::

            sections = [
                TestSection("Part 1: Input handling", [test1, test2]),
                TestSection("Part 2: Computation", [test3, test4]),
            ]
            tester.run_sections(sections)
            tester.display_results()
    """
    name: str
    tests: List['TestCase']


@dataclass
class TestResult:
    """Result of a single test execution.

    Args:
        test_name: Name of the test that was executed.
        passed: Whether the test passed (True) or failed (False).
        message: Detailed message describing the test result.
        error: Optional error message if an exception occurred during testing.
        description: Optional subtitle shown under the test name in the results table.
        skipped: True when the test could not be checked at all (e.g. the student's
            code cell was never found). Skipped tests are never counted as passed,
            but are rendered distinctly so students don't read them as wrong answers.

    Examples:
        Creating a test result::

            result = TestResult(
                test_name="Addition test",
                passed=True,
                message="Expected: 5 | Got: 5",
                error=None
            )
    """
    test_name: str
    passed: bool
    message: str
    error: Optional[str] = None
    description: str = ""
    skipped: bool = False


class ColabTestFramework:
    """Framework for testing student code in Google Colab and Jupyter notebooks.

    This class provides methods to load student code from the last executed cell,
    run various types of tests, and display results in a formatted table.

    Attributes:
        results: List of TestResult objects from the last test run.
        student_code: String containing the code from the last executed cell.

    Examples:
        Basic workflow::

            # Initialize framework
            tester = ColabTestFramework()

            # Define tests
            tests = [
                TestCase(name="Test 1", test_type="return",
                         function_name="my_func", inputs=[5], expected=10)
            ]

            # Run tests and display results
            tester.run_tests(tests)
            tester.display_results()
    """

    #: Seconds a single student cell or function call may run before it is stopped.
    #: Raise it on the instance for exercises that are legitimately slow.
    time_limit_seconds: float = _TIME_LIMIT_SECONDS

    #: Characters of output a single test may capture before it is stopped.
    output_limit_chars: int = _OUTPUT_LIMIT_CHARS

    def __init__(self):
        """Initialize the testing framework with empty results and code."""
        self.results: List[TestResult] = []
        self.section_results: List[tuple] = []  # List of (section_name, List[TestResult])
        self.student_code = ""
        self.warnings: List[str] = []
        self.shadowed_builtins: List[str] = []
        self.shadowing_suspected = False

    @contextmanager
    def _stdin_redirected(self, stdin_input: str):
        """Simulate stdin for the duration of a test call.

        Patches both ``sys.stdin`` and the ``input`` builtin. A live
        Jupyter/Colab kernel intercepts ``input()`` itself and never consults
        ``sys.stdin``, so swapping ``sys.stdin`` alone is not enough — the
        builtin must be patched too for stdin simulation to work under a
        real kernel (it's also what makes exec()-based cell tests, which
        already override ``input`` in their own namespace, keep working).

        When the supplied input runs out, :class:`StdinExhausted` is raised
        instead of handing back an endless stream of empty strings. Feeding
        ``''`` forever turns "your program asks for more input than expected"
        into a hang or an unrelated ``ValueError`` deep inside student code.
        """
        old_stdin = sys.stdin
        old_input = builtins.input
        stream = io.StringIO(stdin_input)

        def _fake_input(prompt=''):
            line = stream.readline()
            if line == '':
                raise StdinExhausted(
                    "input() was called more times than this test provides input for"
                )
            return line.rstrip('\n').rstrip('\r')

        sys.stdin = stream
        builtins.input = _fake_input
        try:
            yield
        finally:
            sys.stdin = old_stdin
            builtins.input = old_input

    @staticmethod
    def _safe_args(inputs: Sequence[Any]) -> List[Any]:
        """Return a private copy of *inputs* so tests can't contaminate each other.

        Student functions frequently mutate their arguments in place (``lst.sort()``,
        ``d.pop()``). Without a copy, the ``TestCase`` objects — which live in the
        notebook and survive across re-runs of the test cell — carry the mutation
        forward, so the same test passes the first time it is run and fails the
        second. Objects that cannot be deep-copied are passed through unchanged.
        """
        try:
            return copy.deepcopy(list(inputs))
        except Exception:
            return list(inputs)

    def _friendly_error(self, exc: BaseException) -> str:
        """Translate an exception raised by student code into student-facing text."""
        if isinstance(exc, StdinExhausted):
            return (
                "Your program asked for more input than this test provides. "
                "Check how many times your code calls input() — it should match "
                "the number of values described in the exercise."
            )
        if isinstance(exc, EOFError):
            return (
                "Your program tried to read input that wasn't there. "
                "Check how many times your code calls input()."
            )
        if isinstance(exc, RecursionError):
            return (
                "Your code kept calling itself until Python gave up "
                "(infinite recursion). Check your stopping condition."
            )
        if isinstance(exc, ExecutionTimeout):
            return (
                "Your program was still running after "
                f"{self.time_limit_seconds:g} seconds, so it was stopped. This almost "
                "always means a loop that never ends — check the condition of your "
                "while, and make sure something inside the loop eventually changes it."
            )
        if isinstance(exc, OutputTooLarge):
            return (
                "Your program printed far more text than this exercise expects, so "
                "it was stopped. This almost always means a print() inside a loop "
                "that never ends."
            )
        if isinstance(exc, SystemExit):
            return (
                "Your program ended early by calling exit(). Remove that call so the "
                "rest of your code can run."
            )
        if self._looks_like_shadowing(exc):
            names = ', '.join(f"'{n}'" for n in self.shadowed_builtins)
            single = len(self.shadowed_builtins) == 1
            noun = "a variable" if single else "variables"
            verb = "hides" if single else "hide"
            self.shadowing_suspected = True
            return (
                f"This notebook has {noun} named {names}, which {verb} Python "
                f"functions of the same name. That breaks code that looks correct "
                f"(for example: \"'str' object is not callable\").\n"
                f"Fix: rename those variables, then restart the runtime "
                f"(Runtime → Restart session) and run your cells again from the top."
            )
        return "Your code produced an error while running — see details below."

    def _looks_like_shadowing(self, exc: BaseException) -> bool:
        """True when *exc* is plausibly caused by a shadowed builtin.

        Blaming shadowing for every TypeError is worse than saying nothing: a
        student whose real bug is ``"a" + 1`` would be told to rename variables and
        restart the runtime while their actual mistake goes unmentioned. Only the
        error shapes a shadowed name actually produces qualify.
        """
        if not self.shadowed_builtins:
            return False
        text = str(exc)
        if 'object is not callable' in text:
            return True
        if 'object is not subscriptable' in text:
            return True
        if isinstance(exc, NameError):
            return any(name in text for name in self.shadowed_builtins)
        return False

    def _implicated_shadowed_builtins(self) -> List[str]:
        """Shadowed builtins that this student's code actually calls.

        ``sum = 0`` is the textbook accumulator and breaks nothing unless the
        student also calls ``sum(...)``. Warning on every rebinding would put an
        alarming banner above a perfect score, so the banner is reserved for names
        the code really uses as functions — or for an error that looked like
        shadowing while the tests ran.
        """
        if not self.shadowed_builtins or not self.student_code:
            return []
        return [
            name for name in self.shadowed_builtins
            if re.search(rf'\b{re.escape(name)}\s*\(', self.student_code)
        ]

    def _detect_shadowed_builtins(self) -> List[str]:
        """Return builtin names the notebook namespace currently shadows.

        See :data:`_SHADOWABLE_BUILTINS` for why this matters: a shadowed builtin
        poisons every later cell in the session and is the single most common
        cause of "it worked yesterday / it works on another machine".
        """
        return _shadowed_builtin_names()

    @staticmethod
    def _is_test_cell(source: str) -> bool:
        """True when a cell's source looks like a test cell, not a student cell."""
        return any(marker in source for marker in _TEST_CELL_MARKERS)

    @staticmethod
    def _is_trivial_cell(source: str) -> bool:
        """True for cells with no student Python in them (blank, comments, magics).

        Students routinely run ``!pip install ...``, ``%%time`` or a bare comment
        between their solution and the test cell. Treating such a cell as "the
        previous cell" is what makes the same tests pass on one run and fail on
        the next.
        """
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if stripped.startswith(('!', '%')):
                continue
            return False
        return True

    @staticmethod
    def _wanted_names(tests: Optional[Sequence['TestCase']]) -> List[str]:
        """Collect every function/variable name the given tests refer to."""
        names: List[str] = []
        for test in tests or []:
            for attr in (test.function_name, test.variable_name):
                if attr and attr not in names:
                    names.append(attr)
        return names

    @staticmethod
    def _cell_defines(source: str, names: Sequence[str]) -> bool:
        """True when *source* defines any of *names* as a function or variable."""
        for name in names:
            escaped = re.escape(name)
            if re.search(rf'^\s*(?:def|class)\s+{escaped}\b', source, re.MULTILINE):
                return True
            if re.search(rf'^\s*{escaped}\s*(?::[^=\n]+)?\s*(?:[-+*/|&^]|//|\*\*|>>|<<)?=(?!=)',
                         source, re.MULTILINE):
                return True
        return False

    def _cell_history(self) -> List[str]:
        """Return this session's executed cells, oldest first, minus the test cell.

        Prefers the ``In`` list from the user namespace and falls back to the
        history manager. The last entry is always the currently-running test
        cell (IPython records a cell before executing it), so it is dropped.
        """
        cells: List[str] = []
        try:
            ipython = get_ipython()
            if ipython is None:
                return []

            raw = ipython.user_ns.get('In')
            if isinstance(raw, (list, tuple)) and len(raw) > 1:
                # In[0] is always '' — the placeholder for "no execution yet".
                cells = [c for c in raw[1:] if isinstance(c, str)]

            if not cells:
                history = ipython.history_manager.get_range(output=False)
                cells = [entry[2] for entry in history if isinstance(entry[2], str)]
        except Exception:
            return []

        return cells[:-1] if cells else []

    _EARLY_STOP_NOTE = (
        "\nNote: your program asked for more input than this test provides, so it "
        "stopped at that point. Only what it printed before then was checked."
    )

    @contextmanager
    def _time_limited(self, seconds: Optional[float] = None):
        """Abort student code that runs longer than *seconds*.

        Uses SIGALRM, which only exists on the main thread of a Unix process —
        exactly where a notebook kernel runs. Anywhere else this is a no-op and a
        runaway loop behaves as it did before. The timer repeats after firing so
        that student code which swallows the first :class:`ExecutionTimeout` in a
        bare ``except`` still gets stopped.
        """
        import signal

        if seconds is None:
            seconds = self.time_limit_seconds

        try:
            previous = signal.getsignal(signal.SIGALRM)
        except (AttributeError, ValueError):
            yield
            return

        def _fire(signum, frame):
            raise ExecutionTimeout(
                f"student code ran longer than {seconds:g} seconds"
            )

        try:
            signal.signal(signal.SIGALRM, _fire)
            signal.setitimer(signal.ITIMER_REAL, seconds, 1.0)
        except (AttributeError, ValueError):
            yield
            return

        try:
            yield
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            except (AttributeError, ValueError):
                pass

    def _call_student(self, func: Callable, inputs: Sequence[Any]) -> Any:
        """Call a student function with copied arguments under the time limit."""
        with self._time_limited():
            return func(*self._safe_args(inputs))

    @staticmethod
    def _strip_notebook_magics(source: str) -> str:
        """Comment out ``!shell`` and ``%magic`` lines so the cell can be exec'd.

        Colab accepts these, plain ``exec`` does not. A student whose solution cell
        opens with ``!pip install pandas`` or ``%%time`` would otherwise fail every
        output test with a SyntaxError while their code runs fine in the notebook.
        Lines are replaced rather than removed so reported line numbers still match
        what the student sees.
        """
        return '\n'.join(
            '# ' + line if line.strip().startswith(('!', '%')) else line
            for line in source.splitlines()
        )

    def _exec_student_code(self) -> bool:
        """Run the student's cell in a private namespace; True if it stopped early.

        Stopping early means the code called ``input()`` more times than the test
        supplies. Whatever it printed before that point is still graded: a program
        that prints everything correctly and then waits for one more keypress
        ("Presione Enter para salir") must not be marked wrong for the keypress.
        Raising rather than returning ``''`` forever still prevents the endless
        loops that the old behaviour produced.
        """
        exec_namespace = {
            '__builtins__': builtins,
            '__name__': '__main__',
        }

        try:
            code = compile(self.student_code, '<celda del estudiante>', 'exec')
        except SyntaxError as original:
            try:
                code = compile(self._strip_notebook_magics(self.student_code),
                               '<celda del estudiante>', 'exec')
            except SyntaxError:
                raise original from None

        try:
            with self._time_limited():
                exec(code, exec_namespace)
        except StdinExhausted:
            return True
        except SystemExit:
            # exit() / sys.exit() simply ends the program. Grade what it printed
            # instead of letting SystemExit escape and kill the whole test cell.
            return False
        return False

    def _document_cells(self) -> Optional[List[str]]:
        """Return the notebook's code cells in *document* order, or None.

        Execution history can never answer "which cell is directly above this test
        cell?" — students run cells out of order, and for output-only tests there is
        no function or variable name to anchor on. Colab can answer it exactly: the
        frontend will hand back the whole .ipynb on request. This is best-effort and
        silently returns None outside Colab, where the history heuristic takes over.
        """
        try:
            from google.colab import _message  # type: ignore
        except Exception:
            return None

        try:
            reply = _message.blocking_request('get_ipynb', request='', timeout_sec=10)
        except Exception:
            return None

        try:
            cells = reply['ipynb']['cells']
        except (TypeError, KeyError):
            return None

        sources = []
        for cell in cells:
            if cell.get('cell_type') != 'code':
                continue
            source = cell.get('source', '')
            if isinstance(source, list):
                source = ''.join(source)
            sources.append(source)
        return sources

    def _student_cell_from_document(self, current_cell: str) -> Optional[str]:
        """Return the student's solution for the exercise this test cell belongs to.

        An exercise is the run of code cells between the previous test cell and this
        one, so the solution is *all* of them joined in document order — students are
        told they may split their work across cells, and a solution split in two is
        not gradeable from its second half alone (the names it needs live in the
        first). Joining also matches the rule the notebook itself states: the cells
        must work when run in order.

        Trivial cells (blank, comments-only, ``!shell``/``%magic``) are left out, and
        another test cell above ends the range — walking past it would pull in a
        different exercise's code. Execution order is irrelevant throughout, so
        re-running this test cell or jumping between exercises changes nothing.
        """
        document = self._document_cells()
        if not document or not current_cell:
            return None

        target = current_cell.strip()
        position = None
        for index, source in enumerate(document):
            if source.strip() == target:
                position = index
        if position is None:
            return None

        collected: List[str] = []
        for source in reversed(document[:position]):
            if self._is_test_cell(source):
                break
            if self._is_trivial_cell(source):
                continue
            collected.append(source)

        if not collected:
            self.warnings.append(
                "The cell above this test does not contain any code yet. Write your "
                "solution in it and run it, then run this test cell again."
            )
            return ""

        collected.reverse()
        return '\n\n'.join(source.rstrip() for source in collected) + '\n' 

    def _current_cell(self) -> str:
        """Return the source of the cell currently executing (the test cell)."""
        try:
            ipython = get_ipython()
            if ipython is None:
                return ""
            raw = ipython.user_ns.get('In')
            if isinstance(raw, (list, tuple)) and raw:
                last = raw[-1]
                return last if isinstance(last, str) else ""
        except Exception:
            pass
        return ""

    def _anchor_from_previous_run(self, current_cell: str,
                                  history: List[str]) -> Optional[str]:
        """Resolve the student cell by looking at the last run of this same test cell.

        A student who finishes exercise 3 and then scrolls back to re-run the test
        for exercise 2 leaves exercise 3's solution as the most recent code cell, so
        "most recent" grades the wrong exercise. The cell that preceded the previous
        run of *this* test cell is a better answer — unless the student has since
        edited and re-run that same solution, which shows up as a recent cell closely
        resembling the old one.
        """
        if not current_cell:
            return None

        previous = [i for i, src in enumerate(history) if src == current_cell]
        if not previous:
            return None
        last_run = previous[-1]

        def usable(sources):
            return [src for src in sources
                    if not self._is_test_cell(src) and not self._is_trivial_cell(src)]

        anchor_candidates = usable(history[:last_run])
        if not anchor_candidates:
            return None
        anchor = anchor_candidates[-1]

        since = usable(history[last_run + 1:])
        if not since:
            # Nothing new ran: the student just clicked the test cell again.
            return anchor

        # Something ran since. If it looks like a revision of what we graded before,
        # it is the student's new attempt; otherwise they moved to another exercise.
        if any(levenshtein_similarity(src, anchor) >= 0.5 for src in since):
            return None
        return anchor

    def load_last_cell(self, tests: Optional[Sequence['TestCase']] = None) -> str:
        """Find and load the student's solution code from the session history.

        The student's solution is **not** reliably ``In[-2]``. ``In`` is the list
        of cells executed in this session, in execution order — not the order the
        cells appear in the notebook. Students run the test cell twice, run a
        scratch cell in between, run an install cell, or jump around the notebook,
        and every one of those makes ``In[-2]`` point at the wrong source. That is
        why regex and output tests appear to pass or fail at random.

        This method instead scans the history backwards and picks the most recent
        cell that is plausibly the solution:

        1. Test cells and trivial cells (blank, comments-only, ``!pip``/``%magic``)
           are discarded.
        2. If *tests* is given, the most recent cell that actually defines one of
           the functions or variables under test wins.
        3. Otherwise the most recent remaining cell is used.

        Args:
            tests: Optional test cases, used to recognise the right cell by the
                names it defines. Strongly recommended.

        Returns:
            The student's code, or an empty string when no plausible cell exists.
            Diagnostics for the student are appended to ``self.warnings``.

        Examples:
            Load student code::

                tester = ColabTestFramework()
                code = tester.load_last_cell(tests)
                print(f"Loaded {len(code)} characters of code")
        """
        self.student_code = ""

        try:
            ipython = get_ipython()
        except Exception:
            ipython = None

        if ipython is None:
            self.warnings.append(
                "This test cell is not running inside Jupyter or Colab, so your code "
                "could not be read. Checks that need your source code were skipped."
            )
            return ""

        # Strategy 1 (exact, Colab only): the code cell directly above this one.
        current_cell = self._current_cell()
        from_document = self._student_cell_from_document(current_cell)
        if from_document is not None:
            self.student_code = from_document
            return self.student_code

        # Strategy 2 (heuristic): reconstruct from execution history.
        history = self._cell_history()
        if not history:
            self.warnings.append(
                "No previous cell was found in this session. Run the cell with your "
                "solution first, then run this test cell again."
            )
            return ""

        candidates = [
            src for src in history
            if not self._is_test_cell(src) and not self._is_trivial_cell(src)
        ]
        if not candidates:
            self.warnings.append(
                "I could only find test cells in this session. Run the cell with YOUR "
                "code (the one above this cell), then run this test cell again."
            )
            return ""

        wanted = self._wanted_names(tests)
        chosen = None
        if wanted:
            for src in reversed(candidates):
                if self._cell_defines(src, wanted):
                    chosen = src
                    break

        if chosen is None:
            chosen = self._anchor_from_previous_run(current_cell, history)

        if chosen is None:
            chosen = candidates[-1]
            if wanted and history and self._is_test_cell(history[-1]):
                # The most recent thing that ran was another test cell, and nothing
                # in the session defines what we're testing.
                self.warnings.append(
                    "I could not find a cell defining "
                    + ", ".join(f"'{n}'" for n in wanted)
                    + ". Make sure you ran the cell with your solution, then run this "
                    "test cell again."
                )

        self.student_code = chosen
        return self.student_code

    def test_cell_output(self, test_name: str, stdin_input: str, expected_output: str) -> TestResult:
        """Test the entire cell's output with given stdin input.

        Executes the student's entire cell code in an isolated namespace with
        provided standard input and compares the printed output.

        Args:
            test_name: Name of the test for display purposes.
            stdin_input: String to provide as standard input (simulates user typing).
            expected_output: Expected output string that should be printed.

        Returns:
            TestResult object indicating pass/fail status and details.

        Note:
            The cell is executed in an isolated namespace to prevent conflicts
            with existing variables and avoid recursion issues.
        """
        try:
            # Capture stdout
            f = _CappedOutput(self.output_limit_chars)

            with self._stdin_redirected(stdin_input), redirect_stdout(f):
                # Execute the student code in an isolated namespace. __name__
                # is set to '__main__' so that the student's own
                # `if __name__ == "__main__": main()` guard actually fires —
                # exec() otherwise leaves __name__ as 'builtins'.
                stopped_early = self._exec_student_code()

            output = f.getvalue().strip()
            expected = expected_output.strip()
            passed = output == expected

            output_display = f"'{output}'" if output else "(nothing printed)"
            expected_display = f"'{expected}'" if expected else "(nothing)"

            message = f"Expected: {expected_display} | Got: {output_display}"
            if not passed and stopped_early:
                message += self._EARLY_STOP_NOTE

            return TestResult(test_name, passed, message, None)

        except _STUDENT_FAILURES as e:
            return TestResult(
                test_name,
                False,
                self._friendly_error(e),
                traceback.format_exc()
            )

    def test_contains_output(self, test_name: str, stdin_input: str, expected: str,
                             function_name: Optional[str] = None,
                             inputs: Optional[List[Any]] = None) -> TestResult:
        """Test that the printed output contains an expected substring.

        Captures stdout produced by running the whole cell or by calling a specific
        function, then checks whether *expected* is a substring of that output.

        Args:
            test_name: Name of the test for display purposes.
            stdin_input: String to provide as standard input.
            expected: Substring that must appear in the output.
            function_name: If provided, call this function instead of the whole cell.
            inputs: Arguments to pass to *function_name*.

        Returns:
            TestResult indicating whether the expected substring was found.
        """
        inputs = inputs or []

        try:
            captured = _CappedOutput(self.output_limit_chars)

            with self._stdin_redirected(stdin_input), redirect_stdout(captured):
                if function_name:
                    func = get_ipython().user_ns.get(function_name)
                    if func is None:
                        return TestResult(
                            test_name, False,
                            f"No function named '{function_name}' was found. "
                            f"Make sure you defined it in the previous cell and ran that cell first.",
                            None
                        )
                    self._call_student(func, inputs)
                    stopped_early = False
                else:
                    stopped_early = self._exec_student_code()

            output = captured.getvalue().strip()
            expected_str = expected.strip()
            passed = expected_str in output

            output_display = f"'{output}'" if output else "(nothing printed)"
            if passed:
                message = f"Output correctly contains: '{expected_str}'"
            else:
                message = (
                    f"Your output did not contain the expected text.\n"
                    f"  Expected to find: '{expected_str}'\n"
                    f"  Got: {output_display}"
                )
            if not passed and stopped_early:
                message += self._EARLY_STOP_NOTE

            return TestResult(test_name, passed, message, None)

        except _STUDENT_FAILURES as e:
            return TestResult(
                test_name, False,
                self._friendly_error(e),
                traceback.format_exc()
            )

    def test_multiline_output(self, test_name: str, stdin_input: str, expected: str,
                              function_name: Optional[str] = None,
                              inputs: Optional[List[Any]] = None) -> TestResult:
        """Test that every expected line appears somewhere in the output (order-independent).

        Splits *expected* on newlines and verifies that each non-empty line can be
        found as a substring anywhere in the captured output. Useful when the exact
        format of the output may vary but all key lines must be present.

        Args:
            test_name: Name of the test for display purposes.
            stdin_input: String to provide as standard input.
            expected: Newline-separated string of lines that must all appear in the output.
            function_name: If provided, call this function instead of the whole cell.
            inputs: Arguments to pass to *function_name*.

        Returns:
            TestResult indicating whether all expected lines were found. On failure,
            the message lists which lines were missing.
        """
        inputs = inputs or []

        try:
            captured = _CappedOutput(self.output_limit_chars)

            with self._stdin_redirected(stdin_input), redirect_stdout(captured):
                if function_name:
                    func = get_ipython().user_ns.get(function_name)
                    if func is None:
                        return TestResult(
                            test_name, False,
                            f"No function named '{function_name}' was found. "
                            f"Make sure you defined it in the previous cell and ran that cell first.",
                            None
                        )
                    self._call_student(func, inputs)
                    stopped_early = False
                else:
                    stopped_early = self._exec_student_code()

            output = captured.getvalue()
            expected_lines = [line.strip() for line in expected.split('\n') if line.strip()]
            missing = [line for line in expected_lines if line not in output]
            passed = len(missing) == 0

            if passed:
                message = f"All {len(expected_lines)} expected line(s) found in output."
            else:
                missing_fmt = "\n  • ".join(missing)
                message = (
                    f"Your output is missing {len(missing)} expected line(s):\n"
                    f"  • {missing_fmt}"
                )
            if not passed and stopped_early:
                message += self._EARLY_STOP_NOTE

            return TestResult(test_name, passed, message, None)

        except _STUDENT_FAILURES as e:
            return TestResult(
                test_name, False,
                self._friendly_error(e),
                traceback.format_exc()
            )

    def test_type_check(self, test_name: str, expected_type: Any,
                        function_name: Optional[str] = None,
                        inputs: Optional[List[Any]] = None,
                        variable_name: Optional[str] = None) -> TestResult:
        """Test that a function's return value or a variable is of an expected type.

        Args:
            test_name: Name of the test for display purposes.
            expected_type: The type (or tuple of types) the value must be an instance of.
            function_name: If provided, call this function and check its return value.
            inputs: Arguments to pass to *function_name*.
            variable_name: If provided (and *function_name* is None), check this variable
                from the IPython namespace.

        Returns:
            TestResult indicating whether the value is an instance of *expected_type*.
        """
        inputs = inputs or []

        def _type_name(t):
            if isinstance(t, tuple):
                return " or ".join(x.__name__ for x in t)
            return t.__name__

        try:
            if function_name:
                func = get_ipython().user_ns.get(function_name)
                if func is None:
                    return TestResult(
                        test_name, False,
                        f"No function named '{function_name}' was found. "
                        f"Make sure you defined it in the previous cell and ran that cell first.",
                        None
                    )
                value = self._call_student(func, inputs)
            else:
                if variable_name not in get_ipython().user_ns:
                    return TestResult(
                        test_name, False,
                        f"The variable '{variable_name}' does not exist. "
                        f"Make sure your code assigns a value to '{variable_name}' in the previous cell.",
                        None
                    )
                value = get_ipython().user_ns[variable_name]

            passed = isinstance(value, expected_type)
            expected_name = _type_name(expected_type)
            actual_name = type(value).__name__

            if passed:
                message = f"Type is correct: {actual_name} ✓"
            else:
                message = (
                    f"Expected a value of type '{expected_name}', "
                    f"but got '{actual_name}' ({value!r})."
                )
            return TestResult(test_name, passed, message, None)

        except _STUDENT_FAILURES as e:
            return TestResult(
                test_name, False,
                self._friendly_error(e),
                traceback.format_exc()
            )

    def test_partial_output(self, test_name: str, stdin_input: str,
                            expected_output: str, similarity_threshold: float,
                            function_name: Optional[str] = None,
                            inputs: Optional[List[Any]] = None) -> TestResult:
        """Test output using Levenshtein similarity instead of exact matching.

        Captures the printed output produced either by running the whole cell or
        by calling a specific function, then passes the test when the similarity
        between the actual output and *expected_output* is greater than or equal
        to *similarity_threshold*.

        Similarity is computed with :func:`levenshtein_similarity`, which returns
        a value in [0.0, 1.0] where 1.0 means identical strings.

        Args:
            test_name: Name of the test for display purposes.
            stdin_input: String to provide as standard input.
            expected_output: The output the student's code should approximately
                produce.
            similarity_threshold: Minimum similarity ratio (0.0, 1.0] required
                to pass the test.  E.g. ``0.8`` → 80 % similar.
            function_name: If provided, call this function instead of running
                the whole cell.
            inputs: Arguments to pass to *function_name* (ignored for cell tests).

        Returns:
            TestResult with pass/fail status, the computed similarity percentage,
            and the threshold that was required.
        """
        inputs = inputs or []

        try:
            captured = _CappedOutput(self.output_limit_chars)

            with self._stdin_redirected(stdin_input), redirect_stdout(captured):
                if function_name:
                    func = get_ipython().user_ns.get(function_name)
                    if func is None:
                        return TestResult(
                            test_name, False,
                            f"No function named '{function_name}' was found. "
                            f"Make sure you defined it in the previous cell and ran that cell first.",
                            None
                        )
                    self._call_student(func, inputs)
                    stopped_early = False
                else:
                    stopped_early = self._exec_student_code()

            output = captured.getvalue().strip()
            expected = expected_output.strip()

            # Levenshtein costs len(output) * len(expected). A runaway loop can make
            # output enormous, so short-circuit using the bound
            # similarity <= min(len) / max(len): when that ceiling is already below
            # the threshold the full matrix cannot change the verdict.
            longest = max(len(output), len(expected))
            shortest = min(len(output), len(expected))
            ceiling = 1.0 if longest == 0 else shortest / longest
            if ceiling < similarity_threshold:
                similarity = ceiling
            else:
                similarity = levenshtein_similarity(output, expected)
            passed = similarity >= similarity_threshold

            output_display = f"'{output}'" if output else "(nothing printed)"
            expected_display = f"'{expected}'" if expected else "(nothing)"
            threshold_pct = f"{similarity_threshold * 100:.1f}%"
            similarity_pct = f"{similarity * 100:.1f}%"

            message = (
                f"Expected (≥{threshold_pct} similar): {expected_display} | "
                f"Got: {output_display} | "
                f"Similarity: {similarity_pct}"
            )
            if not passed and stopped_early:
                message += self._EARLY_STOP_NOTE

            return TestResult(test_name, passed, message, None)

        except _STUDENT_FAILURES as e:
            return TestResult(
                test_name, False,
                self._friendly_error(e),
                traceback.format_exc()
            )

    def test_regex_output(self, test_name: str, stdin_input: str, pattern: str,
                          error_message: str = "",
                          function_name: Optional[str] = None,
                          inputs: Optional[List[Any]] = None) -> TestResult:
        """Test that the printed output matches a regex pattern.

        Captures stdout produced by running the whole cell or by calling a
        specific function, then checks whether *pattern* can be found anywhere
        in that output using :func:`re.search`.

        Args:
            test_name: Name of the test for display purposes.
            stdin_input: String to provide as standard input.
            pattern: Regex pattern to search for in the captured output.
                Uses ``re.MULTILINE | re.DOTALL`` flags.
            error_message: Custom message shown when the test fails.  If empty,
                a default message with the pattern and actual output is shown.
            function_name: If provided, call this function instead of running
                the whole cell.
            inputs: Arguments to pass to *function_name* (ignored for cell tests).

        Returns:
            TestResult indicating whether the pattern was found in the output.
        """
        inputs = inputs or []

        try:
            captured = _CappedOutput(self.output_limit_chars)

            with self._stdin_redirected(stdin_input), redirect_stdout(captured):
                if function_name:
                    func = get_ipython().user_ns.get(function_name)
                    if func is None:
                        return TestResult(
                            test_name, False,
                            f"No function named '{function_name}' was found. "
                            f"Make sure you defined it in the previous cell and ran that cell first.",
                            None
                        )
                    self._call_student(func, inputs)
                    stopped_early = False
                else:
                    stopped_early = self._exec_student_code()

            output = captured.getvalue().strip()
            match = re.search(pattern, output, re.MULTILINE | re.DOTALL)
            passed = match is not None

            if passed:
                message = f"Output matches the expected pattern ✓"
            else:
                if error_message:
                    message = error_message
                else:
                    output_display = f"'{output}'" if output else "(nothing printed)"
                    message = f"The expected pattern was not found in your output. Got: {output_display}"

            if not passed and stopped_early:
                message += self._EARLY_STOP_NOTE

            return TestResult(test_name, passed, message, None)

        except _STUDENT_FAILURES as e:
            return TestResult(
                test_name, False,
                self._friendly_error(e),
                traceback.format_exc()
            )

    def test_function(self, test_name: str, func_name: str, test_type: str,
                      inputs: List[Any], stdin_input: str, expected: Any,
                      tolerance: Optional[float] = None) -> TestResult:
        """Test a specific function with various test types.

        Tests a function by calling it with provided inputs and validating the result
        based on the test type (return value, output, or exception).

        Args:
            test_name: Name of the test for display purposes.
            func_name: Name of the function to test.
            test_type: Type of test - 'return', 'output', or 'exception'.
            inputs: List of arguments to pass to the function.
            stdin_input: Standard input to provide during function execution.
            expected: Expected result (return value, output string, or exception type).
            tolerance: If set (for 'return' tests), passes when
                ``abs(result - expected) <= tolerance``.

        Returns:
            TestResult object indicating pass/fail status and details.

        Note:
            The function must already be defined in the IPython namespace
            (i.e., already executed by the student).
        """
        try:
            # Get the function from globals
            func = get_ipython().user_ns.get(func_name)
            if func is None:
                return TestResult(
                    test_name,
                    False,
                    f"No function named '{func_name}' was found. "
                    f"Make sure you defined it in the previous cell and ran that cell first.",
                    None
                )

            # Prepare stdin if provided
            stdin_ctx = self._stdin_redirected(stdin_input) if stdin_input else nullcontext()

            with stdin_ctx:
                args_repr = ', '.join(map(repr, inputs))
                call_repr = f"{func_name}({args_repr})"

                if test_type == 'return':
                    result = self._call_student(func, inputs)
                    if tolerance is not None:
                        try:
                            passed = abs(result - expected) <= tolerance
                        except TypeError:
                            passed = result == expected
                        message = (
                            f"{call_repr} | Expected: {expected} ± {tolerance} | Got: {result}"
                        )
                    else:
                        passed = result == expected
                        message = (
                            f"{call_repr} | Expected: {repr(expected)} | Got: {repr(result)}"
                        )
                    return TestResult(test_name, passed, message, None)

                elif test_type == 'output':
                    f = _CappedOutput(self.output_limit_chars)
                    with redirect_stdout(f):
                        self._call_student(func, inputs)

                    output = f.getvalue().strip()
                    expected_str = expected.strip() if isinstance(expected, str) else str(expected)
                    passed = output == expected_str

                    output_display = f"'{output}'" if output else "(nothing printed)"
                    expected_display = f"'{expected_str}'" if expected_str else "(nothing)"

                    return TestResult(
                        test_name,
                        passed,
                        f"{call_repr} | Expected output: {expected_display} | Got: {output_display}",
                        None
                    )

                elif test_type == 'exception':
                    try:
                        result = self._call_student(func, inputs)
                        return TestResult(
                            test_name,
                            False,
                            f"{call_repr} | Expected a {expected.__name__} error to be raised, "
                            f"but the function returned {repr(result)} without any error.",
                            None
                        )
                    except expected:
                        return TestResult(
                            test_name,
                            True,
                            f"{call_repr} | Correctly raised {expected.__name__} ✓",
                            None
                        )
                    except _STUDENT_FAILURES as e:
                        return TestResult(
                            test_name,
                            False,
                            f"{call_repr} | Expected a {expected.__name__} error, "
                            f"but got {type(e).__name__} instead: {str(e)}",
                            None
                        )
                else:
                    return TestResult(
                        test_name,
                        False,
                        f"Unknown test type: {test_type}",
                        None
                    )

        except _STUDENT_FAILURES as e:
            return TestResult(
                test_name,
                False,
                self._friendly_error(e),
                traceback.format_exc()
            )

    def test_code_pattern(self, test_name: str, pattern: str, description: str,
                          error_message: str = "", negate: bool = False) -> TestResult:
        """Test if code contains (or does not contain) a specific regex pattern.

        Searches the student's code for a regex pattern match. Useful for verifying
        that students use specific language constructs (loops, conditionals, etc.)
        or avoid certain patterns (like global variables, print statements, etc.).

        Args:
            test_name: Name of the test for display purposes.
            pattern: Regex pattern to search for in the code.
            description: Description of what the pattern checks.
            error_message: Custom error message shown to students when test fails.
            negate: If True, test passes when pattern is NOT found (for not_regex tests).

        Returns:
            TestResult object indicating if pattern was found (or not found if negated).

        Note:
            Pattern matching uses re.MULTILINE and re.DOTALL flags.
        """
        try:
            match = re.search(pattern, self.student_code, re.MULTILINE | re.DOTALL)

            if negate:
                passed = match is None
                if passed:
                    message = description if description else "Required pattern is absent from your code ✓"
                else:
                    message = error_message if error_message else (
                        f"Your code should not contain this pattern."
                        + (f" {description}" if description else "")
                    )
            else:
                passed = match is not None
                if passed:
                    message = description if description else "Required pattern found in your code ✓"
                else:
                    message = error_message if error_message else (
                        "Your code does not contain the expected pattern."
                        + (f" {description}" if description else "")
                    )

            return TestResult(test_name, passed, message, None)
        except _STUDENT_FAILURES as e:
            return TestResult(
                test_name,
                False,
                "Your code produced an error while checking the pattern — see details below.",
                traceback.format_exc()
            )

    def test_variable(self, test_name: str, variable_name: str, validator: Callable,
                     expected: Any = None, error_message: str = "") -> TestResult:
        """Test a variable's value using a validator function.

        Retrieves a variable from the IPython namespace and validates it using
        a provided validator function (typically a lambda). Useful for checking
        variable properties like range, type, length, etc.

        Args:
            test_name: Name of the test for display purposes.
            variable_name: Name of the variable to check.
            validator: Function that takes the variable value and returns bool.
                Must return True if validation passes, False otherwise.
            expected: Optional expected value, used in default error messages.
            error_message: Custom error message for students. Use {value} as
                placeholder for the actual variable value.

        Returns:
            TestResult object indicating if validation passed.
        """
        try:
            # Get the variable from IPython namespace
            if variable_name not in get_ipython().user_ns:
                return TestResult(
                    test_name,
                    False,
                    f"The variable '{variable_name}' does not exist. "
                    f"Make sure your code assigns a value to '{variable_name}' in the previous cell.",
                    None
                )

            value = get_ipython().user_ns[variable_name]

            try:
                passed = validator(value)

                if not isinstance(passed, bool):
                    return TestResult(
                        test_name,
                        False,
                        f"Validator must return True or False, got {type(passed).__name__}",
                        None
                    )

                if passed:
                    message = f"'{variable_name}' = {value!r} ✓"
                else:
                    if error_message:
                        message = error_message.replace("{value}", repr(value))
                    elif expected is not None:
                        message = f"'{variable_name}' = {value!r} | Expected: {expected!r}"
                    else:
                        message = f"'{variable_name}' = {value!r} did not pass the check."

                return TestResult(test_name, passed, message, None)
            except _STUDENT_FAILURES as e:
                return TestResult(
                    test_name,
                    False,
                    f"Your code produced an error while checking '{variable_name}' — see details below.",
                    traceback.format_exc()
                )

        except _STUDENT_FAILURES as e:
            return TestResult(
                test_name,
                False,
                f"Your code produced an error while checking '{variable_name}' — see details below.",
                traceback.format_exc()
            )

    def _dispatch_test(self, test: TestCase, code_loaded: bool) -> Optional[TestResult]:
        """Dispatch a single TestCase to the appropriate test method.

        Tests that need the student's source but could not get it come back marked
        ``skipped`` rather than as ``None``. Dropping them silently changed the size
        of the results table between runs, so a student could see "3/3 passed" on a
        run where five checks never happened.
        """
        _namespace_only = {'variable', 'type_check'}

        if not code_loaded and test.test_type not in _namespace_only:
            return TestResult(
                test.name,
                False,
                "This check could not run because your code cell was not found. "
                "Run the cell with your solution (the one above this test), then "
                "run this test cell again.",
                None,
                test.description,
                skipped=True,
            )

        if test.test_type == 'regex':
            result = self.test_code_pattern(
                test.name, test.pattern, test.description, test.error_message, negate=False
            )
        elif test.test_type == 'not_regex':
            result = self.test_code_pattern(
                test.name, test.pattern, test.description, test.error_message, negate=True
            )
        elif test.test_type == 'variable':
            result = self.test_variable(
                test.name, test.variable_name, test.validator, test.expected, test.error_message
            )
        elif test.test_type == 'partial_output':
            result = self.test_partial_output(
                test.name, test.stdin_input or "", test.expected, test.similarity_threshold,
                function_name=test.function_name, inputs=test.inputs,
            )
        elif test.test_type == 'regex_output':
            result = self.test_regex_output(
                test.name, test.stdin_input or "", test.pattern, test.error_message,
                function_name=test.function_name, inputs=test.inputs,
            )
        elif test.test_type == 'contains_output':
            result = self.test_contains_output(
                test.name, test.stdin_input or "", test.expected,
                function_name=test.function_name, inputs=test.inputs,
            )
        elif test.test_type == 'multiline_output':
            result = self.test_multiline_output(
                test.name, test.stdin_input or "", test.expected,
                function_name=test.function_name, inputs=test.inputs,
            )
        elif test.test_type == 'type_check':
            result = self.test_type_check(
                test.name, test.expected,
                function_name=test.function_name, inputs=test.inputs,
                variable_name=test.variable_name
            )
        elif test.test_type == 'output' and not test.function_name:
            result = self.test_cell_output(test.name, test.stdin_input or "", test.expected)
        elif test.function_name:
            result = self.test_function(
                test.name, test.function_name, test.test_type, test.inputs,
                test.stdin_input or "", test.expected, tolerance=test.tolerance
            )
        else:
            result = TestResult(
                test.name, False,
                f"Invalid test configuration for test type '{test.test_type}'", None
            )

        result.description = test.description
        if result.passed and test.success_message:
            result.message = test.success_message
        return result

    def run_tests(self, tests: List[TestCase]) -> List[TestResult]:
        """Run all tests and store results.

        Executes all provided test cases, loads the student's code from the last
        executed cell, and stores the results. Tests that only read from the IPython
        namespace (``variable``, ``type_check`` on a variable) are run even when
        cell code could not be loaded.

        Args:
            tests: List of TestCase objects to execute.

        Returns:
            List of TestResult objects containing the results of all tests.

        Note:
            Results are also stored in self.results for later access.
        """
        self.results = []
        self.section_results = []
        self.warnings = []
        self.shadowing_suspected = False
        self.shadowed_builtins = self._detect_shadowed_builtins()
        code = self.load_last_cell(tests)
        code_loaded = bool(code)

        for test in tests:
            result = self._dispatch_test(test, code_loaded)
            if result is not None:
                self.results.append(result)

        return self.results

    def run_sections(self, sections: List[TestSection]) -> List[TestResult]:
        """Run grouped tests and store results by section.

        Loads the student's code once, then executes each section's tests in order.
        Results are stored both in ``self.section_results`` (grouped) and
        ``self.results`` (flat list) so that ``display_results()`` renders the
        sectioned view automatically.

        Args:
            sections: List of TestSection objects, each with a name and test list.

        Returns:
            Flat list of all TestResult objects across all sections.

        Examples:
            ::

                tester = ColabTestFramework()
                tester.run_sections([
                    TestSection("Part 1: Basics", [test1, test2]),
                    TestSection("Part 2: Edge cases", [test3]),
                ])
                tester.display_results()
        """
        self.results = []
        self.section_results = []
        self.warnings = []
        self.shadowing_suspected = False
        self.shadowed_builtins = self._detect_shadowed_builtins()
        all_tests = [test for section in sections for test in section.tests]
        code = self.load_last_cell(all_tests)
        code_loaded = bool(code)

        for section in sections:
            section_res: List[TestResult] = []
            for test in section.tests:
                result = self._dispatch_test(test, code_loaded)
                if result is not None:
                    section_res.append(result)
                    self.results.append(result)
            self.section_results.append((section.name, section_res))

        return self.results

    def _results_table_rows(self, results: List[TestResult]) -> str:
        """Return the HTML <tr> rows for a list of TestResult objects."""
        rows = ""
        for result in results:
            if result.skipped:
                status_class, status_text = "status-skip", "⚠ NOT RUN"
            elif result.passed:
                status_class, status_text = "status-pass", "✓ PASS"
            else:
                status_class, status_text = "status-fail", "✗ FAIL"

            safe_message = html_module.escape(result.message).replace('\n', '<br>')

            error_html = ""
            if result.error:
                safe_error = html_module.escape(result.error)
                error_html = (
                    '<details class="error-details">'
                    '<summary>⚠ Show technical details</summary>'
                    f'<pre>{safe_error}</pre>'
                    '</details>'
                )

            description_html = ""
            if result.description:
                description_html = (
                    f'<div class="test-description">{html_module.escape(result.description)}</div>'
                )

            rows += f"""
                <tr>
                    <td class="{status_class}">{status_text}</td>
                    <td>{html_module.escape(result.test_name)}{description_html}</td>
                    <td>{safe_message}{error_html}</td>
                </tr>
            """
        return rows

    def display_results(self):
        """Display test results in a colorful HTML table.

        Renders all test results in a formatted HTML table with color-coded
        pass/fail status, summary statistics, and detailed messages for each test.

        When ``run_sections()`` was used, each section is rendered as a separate
        table under its own named header. Otherwise a single flat table is shown.

        The table includes:
            - Summary bar showing total passed/failed and percentage
            - Status column with green (pass) or red (fail) indicators
            - Test name column (with optional description subtitle)
            - Details column with expected vs actual values
            - Collapsible technical error details when applicable

        Note:
            This method uses IPython's display functionality and will only work
            in notebook environments.
        """
        total = len(self.results)

        if total == 0:
            print("⚠️  No tests were executed.")
            print("📝 Make sure to execute the cell with your solution code first,")
            print("   then run this test cell.")
            for warning in self.warnings:
                print(f"   {warning}")
            return

        passed = sum(1 for r in self.results if r.passed)
        skipped = sum(1 for r in self.results if r.skipped)
        percentage = (passed / total * 100)

        shared_styles = """
        <style>
            .test-results {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                border-collapse: collapse;
                width: 100%;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin: 12px 0 24px 0;
            }
            .test-results th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
                font-size: 14px;
            }
            .test-results td {
                padding: 12px 15px;
                border-bottom: 1px solid #e0e0e0;
                font-size: 13px;
                vertical-align: top;
            }
            .test-results tr:hover {
                background-color: #f8f9fa;
            }
            .status-pass {
                background-color: #d4edda;
                color: #155724;
                font-weight: bold;
                text-align: center;
                border-radius: 4px;
            }
            .status-fail {
                background-color: #f8d7da;
                color: #721c24;
                font-weight: bold;
                text-align: center;
                border-radius: 4px;
            }
            .status-skip {
                background-color: #fff3cd;
                color: #7a5b00;
                font-weight: bold;
                text-align: center;
                border-radius: 4px;
            }
            .notice {
                background: #fff8e1;
                border-left: 5px solid #f0ad4e;
                color: #6b4e00;
                padding: 12px 16px;
                border-radius: 6px;
                margin: 16px 0 0 0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 13px;
                line-height: 1.5;
            }
            .notice b {
                display: block;
                margin-bottom: 4px;
                font-size: 14px;
            }
            .summary {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0 16px 0;
                font-size: 16px;
                font-weight: 600;
                text-align: center;
            }
            .section-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 10px 16px;
                border-radius: 6px;
                margin: 20px 0 0 0;
                font-size: 15px;
                font-weight: 600;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .section-badge {
                font-size: 12px;
                font-weight: 500;
                opacity: 0.9;
            }
            .test-description {
                color: #6c757d;
                font-size: 11px;
                font-style: italic;
                margin-top: 3px;
            }
            .error-details summary {
                cursor: pointer;
                color: #c0392b;
                font-size: 12px;
                margin-top: 6px;
                user-select: none;
            }
            .error-details pre {
                font-size: 11px;
                color: #721c24;
                white-space: pre-wrap;
                word-break: break-word;
                background: #fff5f5;
                border: 1px solid #f5c6cb;
                border-radius: 4px;
                padding: 8px;
                margin: 6px 0 0 0;
            }
        </style>
        """

        if passed == total:
            verdict = '🎉 All tests passed!'
        elif skipped:
            verdict = (
                f'⚠️ {skipped} check(s) could not run — see the notes below.'
            )
        else:
            verdict = '⚠️ Some tests failed — review the details below.'

        summary_html = f"""
        <div class="summary">
            Test Results: {passed}/{total} passed ({percentage:.1f}%)
            {verdict}
        </div>
        """

        notices_html = ""
        implicated = self._implicated_shadowed_builtins()
        if self.shadowing_suspected and not implicated:
            implicated = list(self.shadowed_builtins)
        if implicated:
            names = ', '.join(
                f'<code>{html_module.escape(n)}</code>' for n in implicated
            )
            single = len(implicated) == 1
            noun = "a variable" if single else "variables"
            verb = "hides" if single else "hide"
            notices_html += f"""
        <div class="notice">
            <b>⚠ This notebook's session is in a broken state</b>
            You have {noun} named {names}, which {verb} Python functions of the same
            name. This makes correct code fail with errors like
            <code>'str' object is not callable</code>, and it stays broken until the
            session is restarted.<br>
            <b style="display:inline">What to do:</b> rename those variables in your
            code, then go to <b style="display:inline">Runtime → Restart session</b>
            and run your cells again from the top.
        </div>
            """
        for warning in self.warnings:
            notices_html += (
                '<div class="notice">'
                + html_module.escape(warning).replace('\n', '<br>')
                + '</div>'
            )

        table_header = """
        <table class="test-results">
            <thead>
                <tr>
                    <th style="width: 10%;">Status</th>
                    <th style="width: 30%;">Test</th>
                    <th style="width: 60%;">Details</th>
                </tr>
            </thead>
            <tbody>
        """
        table_footer = """
            </tbody>
        </table>
        """

        if self.section_results:
            html = shared_styles + summary_html + notices_html
            for section_name, section_res in self.section_results:
                sec_passed = sum(1 for r in section_res if r.passed)
                sec_total = len(section_res)
                safe_name = html_module.escape(section_name)
                html += f"""
                <div class="section-header">
                    <span>{safe_name}</span>
                    <span class="section-badge">{sec_passed}/{sec_total} passed</span>
                </div>
                """
                html += table_header
                html += self._results_table_rows(section_res)
                html += table_footer
        else:
            html = shared_styles + summary_html + notices_html + table_header
            html += self._results_table_rows(self.results)
            html += table_footer

        display(HTML(html))
