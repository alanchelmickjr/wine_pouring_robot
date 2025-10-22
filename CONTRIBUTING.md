# Contributing to Wine Pouring Robot

Thank you for your interest in contributing to the Wine Pouring Robot project! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## 🤝 Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences
- Accept responsibility and apologize for mistakes

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of robotics and machine learning
- Familiarity with PyTorch (for ML contributions)

### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/wine-pouring-robot.git
   cd wine-pouring-robot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Dependencies

```bash
# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0

# Code quality
black>=23.7.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.5.0
pylint>=2.17.0

# Documentation
sphinx>=7.1.0
sphinx-rtd-theme>=1.3.0
```

## 🔄 Development Process

### Workflow

1. **Check existing issues** - Look for related issues or create a new one
2. **Discuss major changes** - Open an issue for discussion before starting
3. **Write code** - Follow our coding standards
4. **Add tests** - Ensure your code is well-tested
5. **Update documentation** - Document new features and changes
6. **Submit PR** - Create a pull request with a clear description

### Branch Naming Convention

- `feature/` - New features (e.g., `feature/add-splash-detection`)
- `bugfix/` - Bug fixes (e.g., `bugfix/fix-iou-calculation`)
- `docs/` - Documentation updates (e.g., `docs/update-readme`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-vision-pipeline`)
- `test/` - Test additions/improvements (e.g., `test/add-fluid-dynamics-tests`)

## 💻 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized with `isort`

### Code Formatting

```bash
# Format code with black
black .

# Sort imports
isort .

# Check style
flake8 .

# Type checking
mypy .
```

### Naming Conventions

```python
# Classes: PascalCase
class DiffusionPolicy:
    pass

# Functions/methods: snake_case
def calculate_iou(circle1, circle2):
    pass

# Constants: UPPER_SNAKE_CASE
MAX_FLOW_RATE = 0.5

# Private methods: _leading_underscore
def _internal_helper(self):
    pass
```

### Documentation Strings

Use Google-style docstrings:

```python
def calculate_optimal_tilt(bottle_pos: np.ndarray, cup_pos: np.ndarray) -> float:
    """Calculate optimal bottle tilt angle to hit the cup.
    
    Uses projectile motion equations to determine the angle that will
    make the wine stream land in the target cup.
    
    Args:
        bottle_pos: 3D position of bottle [x, y, z] in meters
        cup_pos: 3D position of cup center [x, y, z] in meters
        
    Returns:
        Optimal tilt angle in radians
        
    Raises:
        ValueError: If cup is unreachable with current flow velocity
        
    Example:
        >>> bottle = np.array([0.5, 0.0, 0.7])
        >>> cup = np.array([0.5, 0.2, 0.45])
        >>> angle = calculate_optimal_tilt(bottle, cup)
        >>> print(f"Tilt: {np.degrees(angle):.1f}°")
        Tilt: 35.2°
    """
    pass
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_fluid_dynamics.py

# Run with verbose output
pytest -v
```

### Writing Tests

```python
import pytest
import numpy as np
from fluid_dynamics import LiquidStreamSimulator

class TestLiquidStreamSimulator:
    """Test suite for liquid stream simulation."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance for testing."""
        return LiquidStreamSimulator()
    
    def test_flow_rate_calculation(self, simulator):
        """Test that flow rate increases with tilt angle."""
        bottle = BottleState(
            position=np.array([0.5, 0.0, 0.7]),
            tilt_angle=np.radians(30),
            fill_level=0.8
        )
        
        flow_rate = simulator.calculate_flow_rate(bottle)
        
        assert flow_rate > 0, "Flow rate should be positive"
        assert flow_rate < 0.001, "Flow rate should be realistic"
    
    def test_trajectory_lands_in_cup(self, simulator):
        """Test that optimal tilt makes stream land in cup."""
        # Test implementation
        pass
```

### Test Coverage Requirements

- **Minimum coverage**: 80% for new code
- **Critical paths**: 100% coverage required
- **Integration tests**: Required for major features

## 📚 Documentation

### Code Documentation

- All public functions/classes must have docstrings
- Complex algorithms should have inline comments
- Use type hints for function signatures

### README Updates

When adding features, update:
- Feature list
- Installation instructions (if needed)
- Usage examples
- API documentation

### Changelog

Update `CHANGELOG.md` with:
- Version number
- Date
- Added/Changed/Fixed/Removed sections

## 🔀 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] No merge conflicts with main branch

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog updated

## Screenshots (if applicable)
Add screenshots for visual changes
```

### Review Process

1. **Automated checks** - CI/CD pipeline runs tests
2. **Code review** - At least one maintainer reviews
3. **Discussion** - Address feedback and questions
4. **Approval** - Maintainer approves changes
5. **Merge** - Squash and merge to main

## 🐛 Issue Guidelines

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. Observe error '...'

**Expected behavior**
What should happen

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.5]
- PyTorch version: [e.g., 2.0.1]

**Additional context**
Any other relevant information
```

### Feature Requests

```markdown
**Feature description**
Clear description of the proposed feature

**Motivation**
Why is this feature needed?

**Proposed solution**
How should it work?

**Alternatives considered**
Other approaches you've thought about

**Additional context**
Any other relevant information
```

## 🎯 Areas for Contribution

We especially welcome contributions in:

### High Priority
- **Performance optimization** - Speed up inference/training
- **Additional robot support** - Integrate new robot arms
- **Improved vision** - Better cup detection algorithms
- **Safety features** - Enhanced collision detection

### Medium Priority
- **Documentation** - Tutorials, examples, API docs
- **Testing** - Increase test coverage
- **Datasets** - Contribute demonstration data
- **Visualization** - Better debugging tools

### Good First Issues
Look for issues labeled `good-first-issue` - these are great for newcomers!

## 📞 Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/yourusername/wine-pouring-robot/discussions)
- **Bugs**: Open a [GitHub Issue](https://github.com/yourusername/wine-pouring-robot/issues)
- **Chat**: Join our [Discord server](https://discord.gg/your-invite)

## 🏆 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in papers (for significant contributions)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Wine Pouring Robot! 🍷🤖