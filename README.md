# Rocketry Formulas

Rocketry Formulas is a small Python library of rocket propulsion equations and
helpers for processing motor test data.

## Installation

```bash
pip install rocketry-formulas
```

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/Clopeq/Rocketry_formulas.git
```

## Usage

```python
from Rocketry_formulas import calculate_cstar_ideal, calculate_impulse

cstar = calculate_cstar_ideal(
	k=1.2,
	Rs=355.0,
	temperature_combustion=3200.0,
)
impulse = calculate_impulse(
	thrust=[0.0, 100.0, 0.0],
	time=[0.0, 1.0, 2.0],
)
```

The `Rocketry_formulas.rocketry` module contains propulsion and nozzle
equations. The `Rocketry_formulas.analysis` module contains CSV and test-data
helpers.

## Development

Install the package in editable mode:

```bash
pip install -e .
```

## License

This project is released under the MIT License.
