
from math import sqrt
import CoolProp.CoolProp as cp
from scipy.optimize import minimize_scalar


def SPI(upstream_pressure: float, downstream_pressure: float, orifice_area: float, discharge_coefficient: float, density: float) -> float:
    """
    Calculate the mass flow rate through an orifice using the SPI (Single Phase Incompressible) model.
    """
    return discharge_coefficient * orifice_area * sqrt(2*(upstream_pressure-downstream_pressure)*density)

def SPC(upstream_pressure: float, downstream_pressure: float, orifice_area: float, discharge_coefficient: float, density: float) -> float:
    """
    Calculate the mass flow rate through an orifice using the SPC (Single Phase Compressible) model.
    """
    raise NotImplementedError("SPC model is not implemented yet.")

def dyer_raw(
        upstream_pressure: float, 
        downstream_pressure: float, 
        orifice_area: float, 
        discharge_coefficient: float, 
        upstream_temperature: float | str = "SAT", 
        upstream_vapor_quality: float = 0.0,
        fluid: str = "N2O"
    ) -> float:
    """
    Calculate the mass flow rate through an orifice using the Dyer model.
    """
    if upstream_temperature == "SAT" or abs(upstream_temperature - cp.PropsSI("T", "P", upstream_pressure, "Q", 0 , fluid)) < 1e-3:
        # Upstream conditions are saturated
        p_vap = upstream_pressure
        #s_up = cp.PropsSI("S", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)
        rho_up = cp.PropsSI("D", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)
    else:
        p_vap = cp.PropsSI("P", "T", upstream_temperature, "Q", 0, fluid)
        #s_up = cp.PropsSI("S", "P", upstream_pressure, "T", upstream_temperature, fluid)
        rho_up = cp.PropsSI("D", "P", upstream_pressure, "T", upstream_temperature, fluid)
        if upstream_pressure < p_vap:   # if this check fails the fluid is supercharged
            # "Subcharged" upstream conditions (gas only)
            raise ValueError("Upstream pressure is below the vapor pressure at the given upstream temperature. The Dyer model is not valid for this case.")
        
    
    k = sqrt((upstream_pressure-downstream_pressure)/(p_vap-downstream_pressure))

    mdot_SPI = SPI(
        upstream_pressure, 
        downstream_pressure, 
        orifice_area, 
        discharge_coefficient, 
        rho_up
    )
    
    mdot_HEM = HEM_raw(
        upstream_pressure, 
        downstream_pressure, 
        orifice_area, 
        discharge_coefficient, 
        upstream_temperature, 
        upstream_vapor_quality,
        fluid
    )


    return (1/(1+k))*mdot_SPI + (1 - 1/(1+k))*mdot_HEM
    
    raise NotImplementedError("Dyer model is not implemented yet.")

def dyer(
        upstream_pressure: float, 
        downstream_pressure: float, 
        orifice_area: float, 
        discharge_coefficient: float, 
        upstream_temperature: float | str = "SAT", 
        upstream_vapor_quality: float  = 0.0,
        fluid: str = "N2O"
    ) -> float:
    """
    Calculate the mass flow rate through an orifice using the Dyer model.
    """
    result = minimize_scalar(
        lambda p: -dyer_raw(
            upstream_pressure, 
            p, 
            orifice_area, 
            discharge_coefficient, 
            upstream_temperature, 
            upstream_vapor_quality,
            fluid
        ), 
        bounds = (1e5, upstream_pressure), 
        method="bounded"
    )

    if downstream_pressure > result.x:
        return dyer_raw(
            upstream_pressure, 
            downstream_pressure, 
            orifice_area, 
            discharge_coefficient, 
            upstream_temperature, 
            upstream_vapor_quality,
            fluid
        )
    else:
        return dyer_raw(
            upstream_pressure, 
            result.x, 
            orifice_area, 
            discharge_coefficient, 
            upstream_temperature, 
            upstream_vapor_quality,
            fluid
        )

def HEM_raw(
        upstream_pressure: float,
        downstream_pressure: float,
        orifice_area: float = 1.0,
        discharge_coefficient: float = 1.0, 
        upstream_temperature: float | str = "SAT", 
        upstream_vapor_quality: float  = 0.0,
        fluid: str = "N2O"
    ) -> float:
    """
    Calculate the mass flow rate through an orifice using the HEM (Homogeneous Equilibrium Model) model.
    To calculate the mass flux G, keep orifice_area and discharge_coefficient as 1.0.
    To calculate actual mass flow rate set the orifice_area and discharge_coefficient to the actual values.

    TODO:
    - What happens when upstream quality is 1.0?
    - quality has no meaning when the fluid is not saturated. Should we raise an error in that case?
    """
    if upstream_vapor_quality < 0.0 or upstream_vapor_quality > 1.0:
        raise ValueError("Upstream vapor quality must be between 0 and 1.")

    if upstream_temperature == "SAT":   
        # Get the saturation temperature of the fluid at the upstream pressure
        upstream_temperature = cp.PropsSI("T", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)

    if abs(upstream_temperature - cp.PropsSI("T", "P", upstream_pressure, "Q", 0, fluid)) > 1e-3:
        # Upstream conditions are not satureated
        h_up = cp.PropsSI("H", "P", upstream_pressure, "T", upstream_temperature, fluid)   # upstream enthalpy
        s_up = cp.PropsSI("S", "P", upstream_pressure, "T", upstream_temperature, fluid)   # downstream entropy
    else:
        # The fluid is satureated at the upstream conditions
        h_up = cp.PropsSI("H", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)   # upstream enthalpy
        s_up = cp.PropsSI("S", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)   


    # assume isentropic expansion to find the downstream enthalpy and density
    h_down = cp.PropsSI("H", "P", downstream_pressure, "S", s_up, fluid)
    rho_down = cp.PropsSI("D", "P", downstream_pressure, "S", s_up, fluid)

    return orifice_area * discharge_coefficient * rho_down * sqrt(2 * (h_up-h_down))

def HEM(
        upstream_pressure: float, 
        downstream_pressure: float, 
        orifice_area: float, 
        discharge_coefficient: float, 
        upstream_temperature: float | str = "SAT", 
        upstream_vapor_quality: float  = 0.0,
        fluid: str = "N2O"
    ) -> float:
    """
    Calculate the mass flow rate through an orifice using the HEM (Homogeneous Equilibrium Model) model.
    """

    result = minimize_scalar(
        lambda p: -HEM_raw(
            upstream_pressure, 
            p, 
            orifice_area, 
            discharge_coefficient, 
            upstream_temperature, 
            upstream_vapor_quality, 
            fluid
        ), 
        bounds = (1e5, upstream_pressure), 
        method="bounded"
    )

    if downstream_pressure > result.x:
        return HEM_raw(
            upstream_pressure, 
            downstream_pressure, 
            orifice_area, 
            discharge_coefficient, 
            upstream_temperature, 
            upstream_vapor_quality, 
            fluid
        )
    else:
        return HEM_raw(
            upstream_pressure, 
            result.x, 
            orifice_area, 
            discharge_coefficient, 
            upstream_temperature, 
            upstream_vapor_quality, 
            fluid
        )

def FML_raw(
        upstream_pressure: float, 
        downstream_pressure: float, 
        orifice_area: float, 
        discharge_coefficient: float, 
        upstream_temperature: float | str = "SAT", 
        upstream_vapor_quality: float = 0.0,
        fluid: str = "N2O"
    ) -> float:
    """
    Calculate the mass flow rate through an orifice using the FML (Foletti–Magni–La Luna) model.
    """

    if (upstream_temperature == "SAT"):
        upstream_temperature = cp.PropsSI("T", "P", upstream_pressure, "Q", 0, fluid)
        SATURATED = True
    else:
        SATURATED = False


    if SATURATED:
        # upstream conditions are saturated
        s_up = cp.PropsSI("S", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)
    else:
        # supercharged upstream conditions
        s_up = cp.PropsSI("S", "P", upstream_pressure, "T", upstream_temperature, fluid)

    T_down = cp.PropsSI("T", "P", downstream_pressure, "S", s_up, fluid)
    s_v_down = cp.PropsSI("S", "P", downstream_pressure, "Q", 1, fluid)
    s_l_down = cp.PropsSI("S", "P", downstream_pressure, "Q", 0, fluid)
    rho_v_down = cp.PropsSI("D", "P", downstream_pressure, "Q", 1, fluid)
    rho_l_down = cp.PropsSI("D", "P", downstream_pressure, "Q", 0, fluid)

    x = (s_up - s_l_down)/(s_v_down - s_l_down)
    S = (rho_l_down/rho_v_down)**(1/3) # slip ratio
    alpha = 1 / (1 + (1-x)/x * S * rho_v_down/rho_l_down)

    mdot_SPI = SPI(
        upstream_pressure, 
        downstream_pressure, 
        orifice_area, 
        discharge_coefficient, 
        cp.PropsSI("D", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)
    )

    mdot_HEM = HEM(
        upstream_pressure, 
        downstream_pressure, 
        orifice_area, 
        discharge_coefficient, 
        upstream_temperature, 
        upstream_vapor_quality,
        fluid
    )
    
    mdot_l = (1-alpha)*mdot_SPI + alpha*mdot_HEM
    mdot_v = alpha*mdot_SPI + (1-alpha)*mdot_HEM
    return upstream_vapor_quality*mdot_v + (1-upstream_vapor_quality)*mdot_l

def FML(
        upstream_pressure: float, 
        downstream_pressure: float, 
        orifice_area: float, 
        discharge_coefficient: float, 
        upstream_temperature: float | str = "SAT", 
        upstream_vapor_quality: float = 0.0,
        fluid: str = "N2O"
    ) -> float:
    """
    Calculate the mass flow rate through an orifice using the FML (Foletti–Magni–La Luna) model.
    """
    result = minimize_scalar(
        lambda p: -FML_raw(
            upstream_pressure, 
            p, 
            orifice_area, 
            discharge_coefficient, 
            upstream_temperature, 
            upstream_vapor_quality, 
            fluid
        ), 
        bounds = (1e5, upstream_pressure), 
        method="bounded"
    )

    if downstream_pressure > result.x:
        return FML_raw(
            upstream_pressure, 
            downstream_pressure, 
            orifice_area, 
            discharge_coefficient, 
            upstream_temperature, 
            upstream_vapor_quality, 
            fluid
        )
    else:
        return FML_raw(
            upstream_pressure, 
            result.x, 
            orifice_area, 
            discharge_coefficient, 
            upstream_temperature, 
            upstream_vapor_quality, 
            fluid
        )