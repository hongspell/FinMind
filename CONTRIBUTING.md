# Contributing to FinMind

Thank you for your interest in contributing to FinMind! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We expect all contributors to:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Accept responsibility for mistakes and learn from them

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Basic understanding of financial concepts (helpful but not required)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/FinMind.git
   cd FinMind
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/hongspell/FinMind.git
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. Copy environment configuration:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run tests to verify setup:
   ```bash
   pytest tests/ -v
   ```

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
1. Check existing issues to avoid duplicates
2. Collect information about the bug (error messages, steps to reproduce)

When submitting a bug report, include:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Environment details (OS, Python version)
- Relevant log output or error messages

### Suggesting Features

Feature suggestions are welcome! Please:
1. Check if the feature has already been suggested
2. Provide a clear use case and description
3. Explain how it benefits users

### Contributing Code

1. **Find an issue** to work on, or create one for discussion
2. **Comment** on the issue to let others know you're working on it
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

4. **Make your changes** following our coding standards
5. **Write tests** for new functionality
6. **Submit a pull request**

## Pull Request Process

1. **Update documentation** if needed
2. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v
   ```

3. **Follow the PR template** (if available)
4. **Request review** from maintainers
5. **Address feedback** promptly

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added for new functionality
- [ ] All tests passing
- [ ] Documentation updated
- [ ] No sensitive data (API keys, credentials) included

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add type hints where possible
- Maximum line length: 100 characters

### Code Structure

```python
# Good: Clear function with docstring
async def analyze_stock(symbol: str, chain: str = "full_analysis") -> AnalysisResult:
    """
    Analyze a stock using the specified analysis chain.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        chain: Analysis chain to use

    Returns:
        AnalysisResult containing all analysis data

    Raises:
        ValueError: If symbol is invalid
    """
    # Implementation
```

### Commit Messages

Follow conventional commits format:
- `feat: Add new valuation method`
- `fix: Correct RSI calculation`
- `docs: Update API documentation`
- `test: Add tests for technical agent`
- `refactor: Simplify data extraction logic`

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_technical_agent.py -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure (e.g., `tests/test_agents/test_technical.py`)
- Use descriptive test names
- Test both success and failure cases

```python
# Example test
def test_calculate_rsi_normal_values():
    """Test RSI calculation with typical price data."""
    prices = [44, 44.25, 44.5, 43.75, 44.5, 44.25, 44.5, 44.5]
    result = calculate_rsi(prices, period=14)
    assert 0 <= result <= 100
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Include examples in docstrings where helpful
- Keep comments up to date with code changes

### README Updates

When adding features, update the README.md to reflect:
- New commands or options
- Configuration changes
- New dependencies

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue` - these are suitable for newcomers.

### Priority Areas

- **Data Sources**: Adding new financial data providers
- **Agents**: Improving analysis agents or adding new ones
- **Testing**: Increasing test coverage
- **Documentation**: Improving docs and examples
- **Localization**: Adding new language support

## Questions?

- Open a GitHub issue for project-related questions
- Check existing issues and documentation first
- Be patient - maintainers are volunteers

---

Thank you for contributing to FinMind! Your efforts help make financial analysis more accessible to everyone.
