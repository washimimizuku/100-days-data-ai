# Day 55: Testing Data Pipelines with pytest - Quiz

Test your understanding of testing with pytest.

---

## Questions

### Question 1
What command runs all tests in the current directory?

A) python test.py  
B) pytest  
C) python -m unittest  
D) test --all

### Question 2
How do you test FastAPI endpoints?

A) Use requests library  
B) Use TestClient from fastapi.testclient  
C) Use urllib  
D) Use curl commands

### Question 3
What is a pytest fixture?

A) A broken test  
B) Reusable test setup code  
C) A test file  
D) A test assertion

### Question 4
How do you test that a function raises an exception?

A) try/except block  
B) pytest.raises()  
C) assert raises()  
D) expect_error()

### Question 5
What does mocking do?

A) Makes fun of your code  
B) Replaces real dependencies with fake ones  
C) Speeds up tests  
D) Generates test data

### Question 6
How do you mark a test to run with parameters?

A) @pytest.params  
B) @pytest.parametrize  
C) @pytest.multiple  
D) @pytest.repeat

### Question 7
What is the AAA pattern in testing?

A) Always Assert All  
B) Arrange-Act-Assert  
C) Automated API Analysis  
D) Async-Await-Assert

### Question 8
How do you measure test coverage?

A) pytest --coverage  
B) pytest --cov  
C) pytest --measure  
D) pytest --check

### Question 9
What should each test focus on?

A) Multiple behaviors  
B) Single behavior  
C) All edge cases  
D) Performance

### Question 10
Where should test files be located?

A) In the same directory as code  
B) In a tests/ directory  
C) Anywhere  
D) In src/

---

## Answers

### Answer 1
**B) pytest**

Simply run `pytest` in your project directory. pytest automatically discovers test files (test_*.py or *_test.py) and test functions (test_*).

### Answer 2
**B) Use TestClient from fastapi.testclient**

FastAPI provides TestClient for testing: `from fastapi.testclient import TestClient`. It simulates HTTP requests without running a server.

### Answer 3
**B) Reusable test setup code**

Fixtures provide reusable setup code using `@pytest.fixture`. They can set up test data, database connections, or any resources needed by tests.

### Answer 4
**B) pytest.raises()**

Use `with pytest.raises(ExceptionType):` to test that code raises an exception. Example: `with pytest.raises(ValueError): function_that_raises()`

### Answer 5
**B) Replaces real dependencies with fake ones**

Mocking replaces real dependencies (APIs, databases) with controlled fake versions. This isolates tests and makes them faster and more reliable.

### Answer 6
**B) @pytest.parametrize**

Use `@pytest.mark.parametrize("input,expected", [(1,2), (3,4)])` to run the same test with different inputs. Reduces code duplication.

### Answer 7
**B) Arrange-Act-Assert**

AAA is a test structure pattern: Arrange (setup), Act (execute), Assert (verify). Makes tests clear and organized.

### Answer 8
**B) pytest --cov**

Run `pytest --cov=app` to measure code coverage. Add `--cov-report=html` for a detailed HTML report showing which lines aren't tested.

### Answer 9
**B) Single behavior**

Each test should focus on one specific behavior. This makes tests easier to understand and maintain. If a test fails, you know exactly what broke.

### Answer 10
**B) In a tests/ directory**

Organize tests in a separate `tests/` directory. Mirror your source structure: `app/models.py` → `tests/test_models.py`.

---

## Scoring

- **10/10**: Perfect! You understand testing
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review fixtures and mocking
- **4-5/10**: Fair - Review core testing concepts
- **0-3/10**: Needs work - Review all sections

---

## Key Concepts to Remember

1. **pytest**: Automatic test discovery
2. **TestClient**: Test FastAPI without server
3. **Fixtures**: Reusable setup with @pytest.fixture
4. **pytest.raises()**: Test exceptions
5. **Mocking**: Replace dependencies
6. **@pytest.parametrize**: Run test with multiple inputs
7. **AAA Pattern**: Arrange-Act-Assert structure
8. **Coverage**: pytest --cov measures tested code
9. **Focus**: One behavior per test
10. **Organization**: tests/ directory structure
