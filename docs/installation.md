# Installation

## Requirements

All of package requirements will be installed automatically if STELA is installed with pip (see below)

STELA Toolkit requires:

- **Python 3.8 or newer**
- A working C++ compiler (required by PyTorch and GPyTorch, again, pip will handle this)
- The following Python packages:

| Package       | Minimum Version |
|---------------|-----------------|
| numpy         | 1.21            |
| scipy         | 1.7             |
| matplotlib    | 3.5             |
| astropy       | 5.0             |
| torch         | 1.10            |
| gpytorch      | 1.9             |
| statsmodels   | 0.13            |

Optionally, you may install Jupyter to run the interactive tutorial

---

## Installing STELA Toolkit

You can install the package using pip:

```bash
pip install stela-toolkit
```

If you are installing directly from the GitHub repository:

```bash
git clone https://github.com/collinlewin/STELA-Toolkit.git
cd STELA-Toolkit
pip install .
```

---

## Verifying Your Installation

You can verify the installation by running:

```python
import stela_toolkit
print(stela_toolkit.__version__)
```

If this runs without an error, you're good to go!