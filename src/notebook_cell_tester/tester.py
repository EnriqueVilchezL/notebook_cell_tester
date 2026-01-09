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
from dataclasses import dataclass
from IPython.display import HTML, display
import traceback


@dataclass
class TestCase:
    """A test case for validating student code.
    
    This class supports multiple test types including function tests, cell output tests,
    code pattern matching, and variable validation. The behavior changes based on which
    parameters are provided.
    
    Args:
        name: Display name for the test shown in the results table.
        test_type: Type of test to perform. Options are:
            - 'output': Test printed output (stdout)
            - 'return': Test function return value
            - 'exception': Test if function raises expected exception
            - 'regex': Test if code matches a regex pattern
            - 'variable': Test variable value using a validator function
        function_name: Name of the function to test. If None, tests entire cell execution.
            Required for function-level tests.
        variable_name: Name of the variable to validate. Required when test_type='variable'.
        inputs: List of arguments to pass to the function. Used for function tests.
        stdin_input: String to provide as standard input (simulates input() function).
            Can contain multiple lines separated by '\\n'.
        expected: Expected value for comparison:
            - For 'return' tests: Expected return value
            - For 'output' tests: Expected printed output string
            - For 'exception' tests: Expected exception type (e.g., ValueError)
            - For 'variable' tests: Optional, used in error messages
        validator: Lambda or function to validate variable value. Must return bool.
            Required when test_type='variable'.
        pattern: Regex pattern to match in code. Required when test_type='regex'.
        description: Additional description for the test (currently unused).
        error_message: Custom error message shown to students when test fails.
            For variable tests, use {value} placeholder for actual value.
    
    Examples:
        Test function return value::
        
            TestCase(
                name="Addition with positive numbers",
                test_type="return",
                function_name="add_numbers",
                inputs=[2, 3],
                expected=5
            )
        
        Test cell output with stdin::
        
            TestCase(
                name="Greet user",
                test_type="output",
                stdin_input="Alice",
                expected="Hello, Alice!"
            )
        
        Test code pattern::
        
            TestCase(
                name="Uses for loop",
                test_type="regex",
                pattern=r"for\s+\w+\s+in\s+",
                error_message="Your code must use a for loop"
            )
        
        Test variable validation::
        
            TestCase(
                name="Age is positive",
                test_type="variable",
                variable_name="age",
                validator=lambda x: x > 0,
                error_message="Variable 'age' must be positive, got {value}"
            )
    """
    name: str
    test_type: str
    function_name: Optional[str] = None
    variable_name: Optional[str] = None
    inputs: Optional[List[Any]] = None
    stdin_input: Optional[str] = None
    expected: Any = None
    validator: Optional[Callable] = None
    pattern: Optional[str] = None
    description: str = ""
    error_message: str = ""
    
    def __post_init__(self):
        """Initialize inputs to empty list if None."""
        if self.inputs is None:
            self.inputs = []


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
                return self.student_code
            
            # Method 2: Use _i variable
            last_input = ipython.user_ns.get('_i', '')
            if last_input:
                self.student_code = last_input
                return last_input
            
            # Method 3: Use history manager
            history = list(ipython.history_manager.get_range(output=False))
            if history and len(history) >= 2:
                # Get second to last entry
                self.student_code = history[-2][2]
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
                    # Create a fresh namespace for execution to avoid conflicts
                    exec_namespace = {}
                    # Execute the student code in isolated namespace
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
    
    def test_code_pattern(self, test_name: str, pattern: str, description: str, error_message: str = "") -> TestResult:
        """Test if code contains a specific regex pattern.
        
        Searches the student's code for a regex pattern match. Useful for verifying
        that students use specific language constructs (loops, conditionals, etc.).
        
        Args:
            test_name: Name of the test for display purposes.
            pattern: Regex pattern to search for in the code.
            description: Description of what the pattern checks (currently unused).
            error_message: Custom error message shown to students when pattern not found.
        
        Returns:
            TestResult object indicating if pattern was found.
        
        Examples:
            Check for for loop::
            
                result = tester.test_code_pattern(
                    test_name="Uses for loop",
                    pattern=r"for\s+\w+\s+in\s+",
                    description="Check for for loop",
                    error_message="Your code must use a for loop"
                )
            
            Check for function definition::
            
                result = tester.test_code_pattern(
                    test_name="Defines calculate function",
                    pattern=r"def\s+calculate\s*\(",
                    description="",
                    error_message="You must define a function called 'calculate'"
                )
        
        Note:
            Pattern matching uses re.MULTILINE and re.DOTALL flags.
        """
        try:
            match = re.search(pattern, self.student_code, re.MULTILINE | re.DOTALL)
            passed = match is not None
            
            # Use custom error message if provided, otherwise use default
            if not passed and error_message:
                message = error_message
            else:
                message = f"Pattern '{pattern}' {'found' if passed else 'not found'} in code"
            
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
        self.load_last_cell()
        
        for test in tests:
            if test.test_type == 'regex':
                # Code pattern test
                result = self.test_code_pattern(
                    test.name,
                    test.pattern,
                    test.description,
                    test.error_message
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
            elif test.function_name:
                # Function test
                result = self.test_function(
                    test.name,
                    test.function_name,
                    test.test_type,
                    test.inputs,
                    test.stdin_input or "",
                    test.expected
                )
            elif test.test_type == 'output':
                # Cell test
                result = self.test_cell_output(
                    test.name,
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
        passed = sum(1 for r in self.results if r.passed)
        
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
            Test Results: {passed}/{total} passed ({(passed/total*100):.1f}%)
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
