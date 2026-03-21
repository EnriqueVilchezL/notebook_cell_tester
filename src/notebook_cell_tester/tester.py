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
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from IPython.display import HTML, display
import traceback


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

        function_name: Name of the function to test. If None, tests entire cell
            execution. Required for function-level tests.
        variable_name: Name of the variable to validate. Required when
            ``test_type='variable'``.
        inputs: List of arguments to pass to the function.
        stdin_input: String to provide as standard input (simulates ``input()``).
            Multiple lines separated by ``'\\n'``.
        expected: Expected value for comparison:

            - ``'return'``: Expected return value.
            - ``'output'`` / ``'partial_output'``: Expected printed output string.
            - ``'exception'``: Expected exception type (e.g. ``ValueError``).
            - ``'variable'``: Optional, used in error messages.

        similarity_threshold: Required for ``'partial_output'``. Float in ``(0.0, 1.0]``
            representing the minimum Levenshtein similarity ratio to pass.
        validator: Callable that takes the variable value and returns ``bool``.
            Required when ``test_type='variable'``.
        pattern: Regex pattern. Required for ``'regex'``, ``'not_regex'``, and
            ``'regex_output'`` tests.
        description: Additional description (currently unused).
        error_message: Custom message shown when the test fails.
            For variable tests, use ``{value}`` as a placeholder.

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
    """
    name: str
    test_type: str
    function_name: Optional[str] = None
    variable_name: Optional[str] = None
    inputs: Optional[List[Any]] = None
    stdin_input: Optional[str] = None
    expected: Any = None
    similarity_threshold: Optional[float] = None
    validator: Optional[Callable] = None
    pattern: Optional[str] = None
    description: str = ""
    error_message: str = ""

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


@dataclass
class TestResult:
    """Result of a single test execution.
    
    Args:
        test_name: Name of the test that was executed.
        passed: Whether the test passed (True) or failed (False).
        message: Detailed message describing the test result.
        error: Optional error message if an exception occurred during testing.
    
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
    
    def __init__(self):
        """Initialize the testing framework with empty results and code."""
        self.results: List[TestResult] = []
        self.student_code = ""
        
    def load_last_cell(self) -> str:
        """Load the code from the last executed cell.
        
        Attempts multiple methods to retrieve the last executed cell's code from
        the IPython environment, including the In variable, _i variable, and
        history manager.
        
        Returns:
            The code from the last executed cell as a string. Returns empty string
            if code cannot be loaded or not running in IPython environment.
        
        Note:
            This method gets the second-to-last cell to avoid reading the test cell itself.
        
        Examples:
            Load student code::
            
                tester = ColabTestFramework()
                code = tester.load_last_cell()
                print(f"Loaded {len(code)} characters of code")
        """
        try:
            # Try to get IPython instance
            ipython = get_ipython()
            if ipython is None:
                print("Warning: Not running in an IPython environment")
                return ""
            
            # Method 1: Use In variable (most reliable)
            last_input = ipython.user_ns.get('In', [])
            if last_input and len(last_input) > 1:
                # Get second to last (current cell is last)
                self.student_code = last_input[-2] if len(last_input) >= 2 else last_input[-1]
                
                # Check if the loaded code contains test framework code
                if 'ColabTestFramework' in self.student_code or 'run_tests' in self.student_code:
                    print("⚠️  WARNING: It looks like you executed the test cell twice!")
                    print("📝 Please execute the PREVIOUS cell (with your solution code) first.")
                    print("   Then run this test cell again.")
                    return ""
                
                return self.student_code
            
            # Method 2: Use _i variable
            last_input = ipython.user_ns.get('_i', '')
            if last_input:
                self.student_code = last_input
                
                # Check if the loaded code contains test framework code
                if 'ColabTestFramework' in self.student_code or 'run_tests' in self.student_code:
                    print("⚠️  WARNING: It looks like you executed the test cell twice!")
                    print("📝 Please execute the PREVIOUS cell (with your solution code) first.")
                    print("   Then run this test cell again.")
                    return ""
                
                return last_input
            
            # Method 3: Use history manager
            history = list(ipython.history_manager.get_range(output=False))
            if history and len(history) >= 2:
                # Get second to last entry
                self.student_code = history[-2][2]
                
                # Check if the loaded code contains test framework code
                if 'ColabTestFramework' in self.student_code or 'run_tests' in self.student_code:
                    print("⚠️  WARNING: It looks like you executed the test cell twice!")
                    print("📝 Please execute the PREVIOUS cell (with your solution code) first.")
                    print("   Then run this test cell again.")
                    return ""
                
                return self.student_code
            
            return ""
        except Exception as e:
            print(f"Error loading cell: {e}")
            return ""
    
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
        
        Examples:
            Test cell that greets user::
            
                result = tester.test_cell_output(
                    test_name="Greet Alice",
                    stdin_input="Alice",
                    expected_output="Hello, Alice!"
                )
        
        Note:
            The cell is executed in an isolated namespace to prevent conflicts
            with existing variables and avoid recursion issues.
        """
        try:
            # Prepare stdin
            old_stdin = sys.stdin
            sys.stdin = io.StringIO(stdin_input)
            
            # Capture stdout
            f = io.StringIO()
            
            try:
                with redirect_stdout(f):
                    # Create a namespace with built-in input redirected
                    exec_namespace = {
                        '__builtins__': __builtins__,
                        'input': lambda prompt='': sys.stdin.readline().rstrip('\n')
                    }
                    # Execute the student code in namespace with custom input
                    exec(self.student_code, exec_namespace)
                
                output = f.getvalue().strip()
                expected = expected_output.strip()
                passed = output == expected
                
                # Format output message
                output_display = f"'{output}'" if output else "Nothing printed"
                expected_display = f"'{expected}'" if expected else "Nothing"
                
                return TestResult(
                    test_name,
                    passed,
                    f"Expected: {expected_display} | Got: {output_display}",
                    None
                )
            finally:
                sys.stdin = old_stdin
                
        except Exception as e:
            sys.stdin = old_stdin
            return TestResult(
                test_name,
                False,
                f"Error executing cell",
                str(e)
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

        Examples:
            Cell-level fuzzy output test::

                result = tester.test_partial_output(
                    test_name="Greeting (fuzzy)",
                    stdin_input="Alice",
                    expected_output="Hello, Alice!",
                    similarity_threshold=0.8
                )

            Function-level fuzzy output test::

                result = tester.test_partial_output(
                    test_name="greet() fuzzy",
                    stdin_input="",
                    expected_output="Hello, Alice!",
                    similarity_threshold=0.9,
                    function_name="greet",
                    inputs=["Alice"]
                )
        """
        inputs = inputs or []

        try:
            old_stdin = sys.stdin
            sys.stdin = io.StringIO(stdin_input)
            captured = io.StringIO()

            try:
                with redirect_stdout(captured):
                    if function_name:
                        func = get_ipython().user_ns.get(function_name)
                        if func is None:
                            return TestResult(
                                test_name, False,
                                f"Function '{function_name}' not found", None
                            )
                        func(*inputs)
                    else:
                        exec_namespace = {
                            '__builtins__': __builtins__,
                            'input': lambda prompt='': sys.stdin.readline().rstrip('\n')
                        }
                        exec(self.student_code, exec_namespace)
            finally:
                sys.stdin = old_stdin

            output = captured.getvalue().strip()
            expected = expected_output.strip()
            similarity = levenshtein_similarity(output, expected)
            passed = similarity >= similarity_threshold

            output_display = f"'{output}'" if output else "Nothing printed"
            expected_display = f"'{expected}'" if expected else "Nothing"
            threshold_pct = f"{similarity_threshold * 100:.1f}%"
            similarity_pct = f"{similarity * 100:.1f}%"

            message = (
                f"Expected (≥{threshold_pct} similar): {expected_display} | "
                f"Got: {output_display} | "
                f"Similarity: {similarity_pct}"
            )
            return TestResult(test_name, passed, message, None)

        except Exception as e:
            sys.stdin = old_stdin
            return TestResult(
                test_name, False,
                "Error executing partial_output test",
                str(e)
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

        Examples:
            Check that a float appears anywhere in cell output::

                TestCase(
                    name="Output contains a float",
                    test_type="regex_output",
                    pattern=r"\\d+\\.\\d+",
                    error_message="Expected a float value in the output"
                )

            Check that a function prints a greeting::

                TestCase(
                    name="greet() prints Hello",
                    test_type="regex_output",
                    function_name="greet",
                    inputs=["Alice"],
                    pattern=r"Hello.*Alice",
                    error_message="Expected 'Hello ... Alice' in output"
                )
        """
        inputs = inputs or []

        try:
            old_stdin = sys.stdin
            sys.stdin = io.StringIO(stdin_input)
            captured = io.StringIO()

            try:
                with redirect_stdout(captured):
                    if function_name:
                        func = get_ipython().user_ns.get(function_name)
                        if func is None:
                            return TestResult(
                                test_name, False,
                                f"Function '{function_name}' not found", None
                            )
                        func(*inputs)
                    else:
                        exec_namespace = {
                            '__builtins__': __builtins__,
                            'input': lambda prompt='': sys.stdin.readline().rstrip('\n')
                        }
                        exec(self.student_code, exec_namespace)
            finally:
                sys.stdin = old_stdin

            output = captured.getvalue().strip()
            match = re.search(pattern, output, re.MULTILINE | re.DOTALL)
            passed = match is not None

            if passed:
                message = f"Pattern '{pattern}' found in output"
            else:
                if error_message:
                    message = error_message
                else:
                    output_display = f"'{output}'" if output else "Nothing printed"
                    message = f"Pattern '{pattern}' not found in output | Got: {output_display}"

            return TestResult(test_name, passed, message, None)

        except Exception as e:
            sys.stdin = old_stdin
            return TestResult(
                test_name, False,
                "Error executing regex_output test",
                str(e)
            )

    def test_function(self, test_name: str, func_name: str, test_type: str, 
                     inputs: List[Any], stdin_input: str, expected: Any) -> TestResult:
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
        
        Returns:
            TestResult object indicating pass/fail status and details.
        
        Examples:
            Test function return value::
            
                result = tester.test_function(
                    test_name="Add 2+3",
                    func_name="add_numbers",
                    test_type="return",
                    inputs=[2, 3],
                    stdin_input="",
                    expected=5
                )
            
            Test function raises exception::
            
                result = tester.test_function(
                    test_name="Division by zero",
                    func_name="divide",
                    test_type="exception",
                    inputs=[10, 0],
                    stdin_input="",
                    expected=ZeroDivisionError
                )
        
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
                    f"Function '{func_name}' not found",
                    None
                )
            
            # Prepare stdin if provided
            old_stdin = None
            if stdin_input:
                old_stdin = sys.stdin
                sys.stdin = io.StringIO(stdin_input)
            
            try:
                if test_type == 'return':
                    # Test return value
                    result = func(*inputs)
                    passed = result == expected
                    return TestResult(
                        test_name,
                        passed,
                        f"{func_name}({', '.join(map(repr, inputs))}) | Expected: {repr(expected)} | Got: {repr(result)}",
                        None
                    )
                
                elif test_type == 'output':
                    # Test printed output
                    f = io.StringIO()
                    with redirect_stdout(f):
                        func(*inputs)
                    
                    output = f.getvalue().strip()
                    expected_str = expected.strip() if isinstance(expected, str) else str(expected)
                    passed = output == expected_str
                    
                    # Format output message
                    output_display = f"'{output}'" if output else "Nothing printed"
                    expected_display = f"'{expected_str}'" if expected_str else "Nothing"
                    
                    return TestResult(
                        test_name,
                        passed,
                        f"{func_name}({', '.join(map(repr, inputs))}) | Expected output: {expected_display} | Got: {output_display}",
                        None
                    )
                
                elif test_type == 'exception':
                    # Test if exception is raised
                    try:
                        result = func(*inputs)
                        # Function didn't raise an exception
                        return TestResult(
                            test_name,
                            False,
                            f"{func_name}({', '.join(map(repr, inputs))}) | Expected {expected.__name__} to be raised, but function returned: {repr(result)}",
                            None
                        )
                    except expected:
                        # Correct exception was raised
                        return TestResult(
                            test_name,
                            True,
                            f"{func_name}({', '.join(map(repr, inputs))}) | Correctly raised {expected.__name__}",
                            None
                        )
                    except Exception as e:
                        # Wrong exception was raised
                        return TestResult(
                            test_name,
                            False,
                            f"{func_name}({', '.join(map(repr, inputs))}) | Expected {expected.__name__}, but got {type(e).__name__}: {str(e)}",
                            None
                        )
                else:
                    return TestResult(
                        test_name,
                        False,
                        f"Unknown test type: {test_type}",
                        None
                    )
            finally:
                if old_stdin:
                    sys.stdin = old_stdin
                    
        except Exception as e:
            if old_stdin:
                sys.stdin = old_stdin
            return TestResult(
                test_name,
                False,
                f"Error executing function {func_name}({', '.join(map(repr, inputs))})",
                str(e)
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
            description: Description of what the pattern checks (currently unused).
            error_message: Custom error message shown to students when test fails.
            negate: If True, test passes when pattern is NOT found (for not_regex tests).
        
        Returns:
            TestResult object indicating if pattern was found (or not found if negated).
        
        Examples:
            Check for for loop::
            
                result = tester.test_code_pattern(
                    test_name="Uses for loop",
                    pattern=r"for\s+\w+\s+in\s+",
                    description="Check for for loop",
                    error_message="Your code must use a for loop"
                )
            
            Check that global keyword is NOT used::
            
                result = tester.test_code_pattern(
                    test_name="No global variables",
                    pattern=r"global\s+\w+",
                    description="",
                    error_message="Your code should not use global variables",
                    negate=True
                )
        
        Note:
            Pattern matching uses re.MULTILINE and re.DOTALL flags.
        """
        try:
            match = re.search(pattern, self.student_code, re.MULTILINE | re.DOTALL)
            
            # For not_regex, we want to pass when there's NO match
            if negate:
                passed = match is None
                if not passed and error_message:
                    message = error_message
                else:
                    message = "Test passed" if passed else "Test did not pass"
            else:
                # Regular regex - pass when there IS a match
                passed = match is not None
                if not passed and error_message:
                    message = error_message
                else:
                    message = "Test passed" if passed else "Test did not pass"
            
            return TestResult(
                test_name,
                passed,
                message,
                None
            )
        except Exception as e:
            return TestResult(
                test_name,
                False,
                f"Error checking pattern",
                str(e)
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
        
        Examples:
            Check if variable is positive::
            
                result = tester.test_variable(
                    test_name="Age is positive",
                    variable_name="age",
                    validator=lambda x: x > 0,
                    error_message="Age must be positive, got {value}"
                )
            
            Check if list has correct length::
            
                result = tester.test_variable(
                    test_name="List has 5 elements",
                    variable_name="scores",
                    validator=lambda x: isinstance(x, list) and len(x) == 5,
                    error_message="scores must be a list with 5 elements"
                )
            
            Check if value in range::
            
                result = tester.test_variable(
                    test_name="Average in valid range",
                    variable_name="average",
                    validator=lambda x: 0 <= x <= 100,
                    expected="0-100",
                    error_message="Average must be between 0 and 100"
                )
        
        Note:
            The variable must exist in the IPython namespace (i.e., already
            defined by the student in their code).
        """
        try:
            # Get the variable from IPython namespace
            if variable_name not in get_ipython().user_ns:
                return TestResult(
                    test_name,
                    False,
                    f"Variable '{variable_name}' not found",
                    None
                )
            
            value = get_ipython().user_ns[variable_name]
            
            # Run the validator
            try:
                passed = validator(value)
                
                if not isinstance(passed, bool):
                    return TestResult(
                        test_name,
                        False,
                        f"Validator must return True or False, got {type(passed).__name__}",
                        None
                    )
                
                # Build message
                if passed:
                    message = f"Variable '{variable_name}' = {repr(value)} passed validation"
                else:
                    if error_message:
                        message = error_message.replace("{value}", repr(value))
                    elif expected is not None:
                        message = f"Variable '{variable_name}' = {repr(value)} | Expected: {repr(expected)}"
                    else:
                        message = f"Variable '{variable_name}' = {repr(value)} failed validation"
                
                return TestResult(
                    test_name,
                    passed,
                    message,
                    None
                )
            except Exception as e:
                return TestResult(
                    test_name,
                    False,
                    f"Error running validator on '{variable_name}'",
                    str(e)
                )
                
        except Exception as e:
            return TestResult(
                test_name,
                False,
                f"Error checking variable '{variable_name}'",
                str(e)
            )
    
    def run_tests(self, tests: List[TestCase]) -> List[TestResult]:
        """Run all tests and store results.
        
        Executes all provided test cases, loads the student's code from the last
        executed cell, and stores the results.
        
        Args:
            tests: List of TestCase objects to execute.
        
        Returns:
            List of TestResult objects containing the results of all tests.
        
        Examples:
            Run multiple tests::
            
                tester = ColabTestFramework()
                tests = [
                    TestCase(name="Test 1", ...),
                    TestCase(name="Test 2", ...),
                ]
                results = tester.run_tests(tests)
                print(f"Passed {sum(r.passed for r in results)}/{len(results)}")
        
        Note:
            Results are also stored in self.results for later access.
        """
        self.results = []
        code = self.load_last_cell()
        
        # If load_last_cell detected test cell executed twice, stop here
        if not code:
            return self.results
        
        for test in tests:
            if test.test_type == 'regex':
                # Code pattern test (must match)
                result = self.test_code_pattern(
                    test.name,
                    test.pattern,
                    test.description,
                    test.error_message,
                    negate=False
                )
            elif test.test_type == 'not_regex':
                # Code pattern test (must NOT match)
                result = self.test_code_pattern(
                    test.name,
                    test.pattern,
                    test.description,
                    test.error_message,
                    negate=True
                )
            elif test.test_type == 'variable':
                # Variable validation test
                result = self.test_variable(
                    test.name,
                    test.variable_name,
                    test.validator,
                    test.expected,
                    test.error_message
                )
            elif test.test_type == 'partial_output':
                # Fuzzy output test using Levenshtein similarity
                result = self.test_partial_output(
                    test.name,
                    test.stdin_input or "",
                    test.expected,
                    test.similarity_threshold,
                    function_name=test.function_name,
                    inputs=test.inputs,
                )
            elif test.test_type == 'regex_output':
                # Regex pattern search in captured output
                result = self.test_regex_output(
                    test.name,
                    test.stdin_input or "",
                    test.pattern,
                    test.error_message,
                    function_name=test.function_name,
                    inputs=test.inputs,
                )
            elif test.test_type == 'output' and not test.function_name:
                # Cell output test (no function specified)
                result = self.test_cell_output(
                    test.name,
                    test.stdin_input or "",
                    test.expected
                )
            elif test.function_name:
                # Function test (return, output, exception)
                result = self.test_function(
                    test.name,
                    test.function_name,
                    test.test_type,
                    test.inputs,
                    test.stdin_input or "",
                    test.expected
                )
            else:
                result = TestResult(
                    test.name,
                    False,
                    f"Invalid test configuration for test type '{test.test_type}'",
                    None
                )
            
            self.results.append(result)
        
        return self.results
    
    def display_results(self):
        """Display test results in a colorful HTML table.
        
        Renders all test results in a formatted HTML table with color-coded
        pass/fail status, summary statistics, and detailed messages for each test.
        
        The table includes:
            - Summary bar showing total passed/failed and percentage
            - Status column with green (pass) or red (fail) indicators
            - Test name column
            - Details column with expected vs actual values
            - Error messages when applicable
        
        Examples:
            Display results after running tests::
            
                tester = ColabTestFramework()
                tester.run_tests(tests)
                tester.display_results()
        
        Note:
            This method uses IPython's display functionality and will only work
            in notebook environments.
        """
        total = len(self.results)
        
        # Handle case where no tests were run
        if total == 0:
            print("⚠️  No tests were executed.")
            print("📝 Make sure to execute the cell with your solution code first,")
            print("   then run this test cell.")
            return
        
        passed = sum(1 for r in self.results if r.passed)
        percentage = (passed/total*100)
        
        # Build HTML table
        html = f"""
        <style>
            .test-results {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                border-collapse: collapse;
                width: 100%;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin: 20px 0;
            }}
            .test-results th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
                font-size: 14px;
            }}
            .test-results td {{
                padding: 12px 15px;
                border-bottom: 1px solid #e0e0e0;
                font-size: 13px;
            }}
            .test-results tr:hover {{
                background-color: #f8f9fa;
            }}
            .status-pass {{
                background-color: #d4edda;
                color: #155724;
                font-weight: bold;
                text-align: center;
                border-radius: 4px;
            }}
            .status-fail {{
                background-color: #f8d7da;
                color: #721c24;
                font-weight: bold;
                text-align: center;
                border-radius: 4px;
            }}
            .summary {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                font-size: 16px;
                font-weight: 600;
                text-align: center;
            }}
            .error-msg {{
                color: #dc3545;
                font-size: 11px;
                font-style: italic;
                margin-top: 4px;
            }}
        </style>
        
        <div class="summary">
            Test Results: {passed}/{total} passed ({percentage:.1f}%)
            {'🎉 All tests passed!' if passed == total else '⚠️ Some tests failed'}
        </div>
        
        <table class="test-results">
            <thead>
                <tr>
                    <th style="width: 10%;">Status</th>
                    <th style="width: 30%;">Test Name</th>
                    <th style="width: 60%;">Details</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for result in self.results:
            status_class = "status-pass" if result.passed else "status-fail"
            status_text = "✓ PASS" if result.passed else "✗ FAIL"
            
            error_html = ""
            if result.error:
                error_html = f'<div class="error-msg">Error: {result.error}</div>'
            
            html += f"""
                <tr>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result.test_name}</td>
                    <td>{result.message}{error_html}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        display(HTML(html))
