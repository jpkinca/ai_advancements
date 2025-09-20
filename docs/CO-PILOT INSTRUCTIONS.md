CO-PILOT INSTRUCTIONS

ALL MARKET AND TRADING DATA NEEDS TO COME THROUGH THE IBKR IB gateway
NO MOCK DATA IS ALLOWED, REALTIME ONLINE only USING THE IBKR GATEWAY
FOR TECHNICAL ANALYSIS USE TA-LIB AT ALL TIMES WITH IBKR DATA RETRIEVED FROM THE GATEWAY, OR WHERE APPLICABLE ALREADY STORED IN THE POSTGRESQL DATABASE
NO FALLBACKS ARE ALLOWED IN THE CODE
ALL DATA NEEDS TO PERSIST IN THE POSTGRESQL database
NO DATA IS ALLOWED TO BE PASSED BETWEEN MODULES USING .json FILES
CONSISTENT LOGGING IS REQUIRED FOR DEBUGGING AND INFORMATIONAL PURPOSES
FOR ALL DEFECTS, PERFORM A ROOTCAUSE ANALYSIS, ASK THE 5-WHYS, DO NOT WORK AROUND DEFECTS BY SIMPLIFYING PROGRAMS REMOVING CRITICAL SOURCE CODE
ENSURE THE MAIN PROGRAM IS FIXED AND ALL PROGRAMS CREATED TO FIND A RESOLUTION REMOVED - IT NEEDS TO 1000% CLEAR WHAT THE PRODUCTION SOURCE CODE WILL BE
NO EMOJIS ARE ALLOWED IN THE SOURCE CODE EXCEPT FOR IN LOGS OUR UX PRESENTATION LAYER
IBKR MARKET DATA at lower intervals than daily need to include pre-market and afterhours data
DATE AND TIME OF TRADING DATA AND TRANSACTIONS MUST ADHERE TO THE US EASTERN TIMEZONE, THE TIMEZONE OF THE NASDAQ AND NYSE

PROGRAMS AND CODE
    - NEED TO BE FUNCTIONAL
    - AVOID OVERENGINEERING
    - KISS PRINCIPLE, MODULA, CLEAN, CRISP AND TIGHT CODING
    - ALWAYS DEVELOP WITH LOOSE COUPLING, REUSE AND BUILDING LIBRARIES IN MIND
    - AVOID BELLS AND WHISTLES
    - DO NOT ADD FEATURES OR FUNCTIONALITY I DID NOT ASK FOR
    - DO NOT REMOVE FUNCTIONALITY WITHOUT APPROVAL

DATABASE
    - DATABASE FOR THE PROJECT IS POSTGRESQL, NO SQLITE ALLOWED
    - DATABASE_URL = postgresql://postgres:TAqEkujnMknVURCcrYTIDOzQXbgBNtSX@turntable.proxy.rlwy.net:10410/railway
    - USER postgres
    - POST 5432

DOCUMENTATION
- SAVE FUNCTIONAL, USER, TECHNICAL AND IMPLEMENTATION DOCUMENTATION UNDER /DOCS
= SAVE REVIEWS , ASSESSMENTS , STATUS REPORTS, NEXT STEPS, PROPOSALS AND PLANS UNDER /PM_ORG


- SAVE REUSABLE PROGRAMS WRITTEN TO ASSIST IN DEVELOPMENT AND DEBUGGING UNDER /TOOLS

- DEVELOPMENT STATUS



- QA
    - FOLLOW AND APPLY THE QA CHECKLIST  UNDER THE /QA FOLDER


- DEVELOPMENT STANDARDS

MOSTLY FOLLOW Python Development Standards and Best Practices

## Table of Contents
1. [Code Style and Formatting](#code-style-and-formatting)
2. [Documentation](#documentation)
3. [Testing](#testing)
4. [Error Handling](#error-handling)
5. [Performance and Optimization](#performance-and-optimization)
6. [Security](#security)
7. [Version Control](#version-control)
8. [Dependency Management](#dependency-management)
9. [Project Structure](#project-structure)
10. [CI/CD](#cicd)

## Code Style and Formatting

### PEP 8 Compliance
```python
# Good
def calculate_area(radius: float) -> float:
    """Calculate the area of a circle."""
    return math.pi * radius ** 2

# Bad
def CalculateArea(r):
    return 3.14*r*r
```

### Naming Conventions
```python
# Variables and functions: snake_case
user_name = "john"
calculate_total_price()

# Classes: PascalCase
class DatabaseConnection:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CONNECTIONS = 100
DEFAULT_TIMEOUT = 30

# Private members: leading underscore
_private_variable = "internal"
```

### Type Hints
```python
from typing import List, Optional, Dict, Union

def process_data(
    data: List[Dict[str, Union[str, int]]],
    timeout: Optional[int] = None
) -> bool:
    """Process data with optional timeout."""
    # Function implementation
    return True
```

### Formatting Tools
```bash
# Use black for formatting
pip install black
black my_project/

# Use isort for import sorting
pip install isort
isort my_project/

# Use flake8 for linting
pip install flake8
flake8 my_project/
```

## Documentation

### Docstrings (Google Style)
```python
def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        x1: X coordinate of first point
        y1: Y coordinate of first point
        x2: X coordinate of second point
        y2: Y coordinate of second point

    Returns:
        Distance between the two points

    Raises:
        ValueError: If coordinates are not finite numbers

    Examples:
        >>> calculate_distance(0, 0, 3, 4)
        5.0
    """
    if not all(math.isfinite(coord) for coord in [x1, y1, x2, y2]):
        raise ValueError("Coordinates must be finite numbers")

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
```

### Module-Level Documentation
```python
"""
Data processing utilities for customer analytics.

This module provides functions for cleaning, transforming, and analyzing
customer data from various sources.

Key Features:
- Data validation and cleaning
- Statistical analysis
- Report generation

Dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
"""
```

## Testing

### Test Structure
```python
# tests/test_calculations.py
import pytest
from my_project.calculations import calculate_distance

class TestDistanceCalculation:
    """Test suite for distance calculation function."""

    def test_distance_positive_coordinates(self):
        """Test distance calculation with positive coordinates."""
        result = calculate_distance(0, 0, 3, 4)
        assert result == 5.0

    def test_distance_negative_coordinates(self):
        """Test distance calculation with negative coordinates."""
        result = calculate_distance(-1, -1, 2, 3)
        assert pytest.approx(result, 0.001) == 5.0

    def test_distance_invalid_coordinates(self):
        """Test that invalid coordinates raise ValueError."""
        with pytest.raises(ValueError):
            calculate_distance(0, 0, float('inf'), 4)
```

### Test Organization
```
project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── module1.py
│       └── module2.py
└── tests/
    ├── __init__.py
    ├── test_module1.py
    └── test_module2.py
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=my_project

# Run specific test file
pytest tests/test_module1.py

# Run tests with verbose output
pytest -v
```

## Error Handling

### Specific Exceptions
```python
# Good
try:
    with open(filename, 'r') as file:
        data = file.read()
except FileNotFoundError:
    logger.error("File %s not found", filename)
    raise
except PermissionError:
    logger.error("Permission denied for file %s", filename)
    raise

# Bad
try:
    with open(filename, 'r') as file:
        data = file.read()
except Exception:  # Too broad
    pass
```

### Custom Exceptions
```python
class DataValidationError(Exception):
    """Exception raised for data validation errors."""

    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)

def validate_user_data(user_data: dict):
    if 'email' not in user_data:
        raise DataValidationError("Email is required", "email")
```

### Context Managers for Resource Management
```python
class DatabaseConnection:
    def __enter__(self):
        self.connection = create_connection()
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()

# Usage
with DatabaseConnection() as conn:
    result = conn.execute_query("SELECT * FROM users")
```

## Performance and Optimization

### Efficient Data Structures
```python
# Use sets for membership testing
valid_users = {'user1', 'user2', 'user3'}
if username in valid_users:  # O(1) operation
    pass

# Use generators for large datasets
def process_large_file(filename):
    with open(filename, 'r') as file:
        for line in file:  # Memory efficient
            yield process_line(line)
```

### Caching and Memoization
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(x: int, y: int) -> int:
    """Perform expensive calculation with caching."""
    # Complex computation
    return result
```

### Async/Await for I/O Operations
```python
import aiohttp
import asyncio

async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def main():
    tasks = [fetch_data(url) for url in url_list]
    results = await asyncio.gather(*tasks)
```

## Security

### Input Validation
```python
import re
from typing import Optional

def sanitize_input(input_string: str, max_length: int = 100) -> Optional[str]:
    """Sanitize and validate user input."""
    if not input_string or len(input_string) > max_length:
        return None

    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>{}[\]\\]', '', input_string.strip())
    return sanitized if sanitized else None
```

### Secure Configuration
```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class Config:
    """Application configuration with secure defaults."""

    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-default-change-in-production')
    DATABASE_URL = os.getenv('DATABASE_URL')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.SECRET_KEY or cls.SECRET_KEY == 'dev-default-change-in-production':
            raise ValueError("SECRET_KEY must be set in production")
```

## Version Control

### Git Commit Messages
```
feat: add user authentication system

- Implement JWT-based authentication
- Add user registration and login endpoints
- Include password hashing with bcrypt

Resolves: #123
```

### .gitignore Example
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
```

## Dependency Management

### requirements.txt
```
# Production dependencies
Django==4.2.0
psycopg2-binary==2.9.6
redis==4.5.5
celery==5.2.7

# Development dependencies
black==23.3.0
flake8==6.0.0
pytest==7.3.1
pytest-cov==4.0.0
```

### pyproject.toml (Modern Approach)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-project"
version = "0.1.0"
dependencies = [
    "Django>=4.2,<5.0",
    "psycopg2-binary>=2.9.6",
    "redis>=4.5.5",
]

[project.optional-dependencies]
dev = [
    "black>=23.3.0",
    "flake8>=6.0.0",
    "pytest>=7.3.1",
]

[tool.black]
line-length = 88
target-version = ['py310']
```

## Project Structure

### Recommended Layout
```
my_project/
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   └── services.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── validators.py
│       │   └── helpers.py
│       └── api/
│           ├── __init__.py
│           ├── routes.py
│           └── middleware.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core/
│   └── test_api/
├── docs/
│   ├── conf.py
│   ├── index.rst
│   └── make.bat
├── scripts/
│   ├── deploy.sh
│   └── setup.py
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
```

## CI/CD

### GitHub Actions Example
```yaml
name: Python CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Run linting
      run: |
        black --check .
        flake8 .

    - name: Run tests
      run: |
        pytest --cov=my_package --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Additional Best Practices

### Logging
```python
import logging
import logging.config

logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s %(name)-15s %(levelname)-8s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'detailed'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console']
    }
})

logger = logging.getLogger(__name__)
logger.info("Application started")
```

### Configuration Management
```python
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings with validation."""

    database_url: str = Field(..., env="DATABASE_URL")
    secret_key: str = Field(..., env="SECRET_KEY")
    debug: bool = Field(False, env="DEBUG")

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Monitoring and Observability
```python
from prometheus_client import Counter, Histogram
import time

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

def track_request(func):
    def wrapper(*args, **kwargs):
        REQUEST_COUNT.inc()
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    return wrapper
```

These standards and best practices provide a solid foundation for developing maintainable, scalable, and secure Python applications. Adapt them to your specific project needs and team preferences.
