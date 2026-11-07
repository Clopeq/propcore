# propcore/__init__.py
from importlib.metadata import version

__version__ = version("propcore")

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

from .N2O import (
    n2o_sat_p_from_t,
    n2o_sat_t_from_p,
    n2o_gas_density_from_t,
    n2o_gas_density_from_p,
    n2o_liquid_density_from_t,
    n2o_liquid_density_from_p
)

from .flow import (
    SPI,
    SPC,
    dyer_raw,
    dyer_raw_compressible,
    dyer,
    dyer_compressible,
    HEM_raw,
    HEM,
    FML_raw,
    FML
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
    "n2o_sat_p_from_t",
    "n2o_sat_t_from_p",
    "n2o_gas_density_from_t",
    "n2o_gas_density_from_p",
    "n2o_liquid_density_from_t",
    "n2o_liquid_density_from_p",
    "SPI",
    "SPC",
    "dyer_raw",
    "dyer_raw_compressible",
    "dyer",
    "dyer_compressible",
    "HEM_raw",
    "HEM",
    "FML_raw",
    "FML"
]