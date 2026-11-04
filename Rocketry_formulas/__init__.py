# Rocketry_formulas/__init__.py

__version__ = "0.1.0"

from .analysis import (
    average_smoothing,
    detect_start,
    detect_stop,
    downsample,
    open_csv,
)
from .rocketry import (
    calculate_Cf,
    calculate_Cf_fixed_nozzle,
    calculate_Cf_ideal,
    calculate_cstar,
    calculate_cstar_ideal,
    calculate_impulse,
    calculate_isp,
    calculate_isp_cstar_Cf,
    calculate_isp_ideal,
)

__all__ = [
    "__version__",
    "average_smoothing",
    "detect_start",
    "detect_stop",
    "downsample",
    "open_csv",
    "calculate_Cf",
    "calculate_Cf_fixed_nozzle",
    "calculate_Cf_ideal",
    "calculate_cstar",
    "calculate_cstar_ideal",
    "calculate_impulse",
    "calculate_isp",
    "calculate_isp_cstar_Cf",
    "calculate_isp_ideal",
]