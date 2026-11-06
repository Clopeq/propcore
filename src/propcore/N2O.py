import CoolProp.CoolProp as cp  # type: ignore[import-not-found]


def n2o_sat_p_from_t(temp: float):
    """Return the saturation pressure of nitrous oxide at a temperature."""
    return cp.PropsSI("P", "T", temp, "Q", 0, "N2O")

def n2o_sat_t_from_p(pressure: float):
    """Return the saturation temperature of nitrous oxide at a pressure."""
    return cp.PropsSI("T", "P", pressure, "Q", 0, "N2O")

def n2o_gas_density_from_t(temp: float):
    """Return the density of nitrous oxide gas at a temperature."""
    return cp.PropsSI("D", "T", temp, "Q", 1, "N2O")

def n2o_gas_density_from_p(pressure: float):
    """Return the density of nitrous oxide gas at a pressure."""
    return cp.PropsSI("D", "P", pressure, "Q", 1, "N2O")

def n2o_liquid_density_from_t(temp: float):
    """Return the density of nitrous oxide liquid at a temperature."""
    return cp.PropsSI("D", "T", temp, "Q", 0, "N2O")

def n2o_liquid_density_from_p(pressure: float):
    """Return the density of nitrous oxide liquid at a pressure."""
    return cp.PropsSI("D", "P", pressure, "Q", 0, "N2O")
