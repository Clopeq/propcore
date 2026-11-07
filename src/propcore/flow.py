
from math import isfinite, sqrt
import CoolProp.CoolProp as cp
from scipy.optimize import minimize_scalar
from scipy.constants import R


def _validate_pressure_inputs(upstream_pressure: float, downstream_pressure: float) -> None:
    if not isfinite(upstream_pressure) or not isfinite(downstream_pressure):
        raise ValueError("Upstream and downstream pressures must be finite numbers.")
    if upstream_pressure <= 0.0 or downstream_pressure <= 0.0:
        raise ValueError("Pressure inputs must be strictly positive.")
    if downstream_pressure >= upstream_pressure:
        raise ValueError("Downstream pressure must be strictly lower than upstream pressure.")


def _validate_orifice_inputs(orifice_area: float, discharge_coefficient: float, density: float | None = None) -> None:
    if not isfinite(orifice_area) or not isfinite(discharge_coefficient):
        raise ValueError("Orifice area and discharge coefficient must be finite numbers.")
    if orifice_area <= 0.0:
        raise ValueError("Orifice area must be strictly positive.")
    if discharge_coefficient <= 0.0:
        raise ValueError("Discharge coefficient must be strictly positive.")
    if density is not None and (not isfinite(density) or density <= 0.0):
        raise ValueError("Density must be a finite, strictly positive value.")


def _bounded_downstream_pressure(upstream_pressure: float) -> tuple[float, float]:
    if not isfinite(upstream_pressure) or upstream_pressure <= 0.0:
        raise ValueError("Upstream pressure must be a finite, strictly positive value.")
    lower = max(1e-12, 1e-6 * upstream_pressure)
    if lower >= upstream_pressure:
        lower = upstream_pressure * 0.5
    return lower, upstream_pressure


def SPI(upstream_pressure: float, downstream_pressure: float, orifice_area: float, discharge_coefficient: float, density: float) -> float:
    """
    Calculate the mass flow rate through an orifice using the SPI (Single Phase Incompressible) model.
    """
    _validate_pressure_inputs(upstream_pressure, downstream_pressure)
    _validate_orifice_inputs(orifice_area, discharge_coefficient, density)
    return discharge_coefficient * orifice_area * sqrt(2 * (upstream_pressure - downstream_pressure) * density)

def SPC(
        upstream_pressure: float, 
        downstream_pressure: float, 
        orifice_area: float, 
        discharge_coefficient: float, 
        upstream_temperature: float | str = "SAT", 
        upstream_vapor_quality: float = 0.0,
        fluid: str = "N2O"
    ) -> float:
    """
    Calculate the mass flow rate through an orifice using the SPC (Single Phase Compressible) model.
    """
    _validate_pressure_inputs(upstream_pressure, downstream_pressure)
    _validate_orifice_inputs(orifice_area, discharge_coefficient)

    Tsat = cp.PropsSI("T", "P", upstream_pressure, "Q", 0, fluid)
    if upstream_temperature == "SAT":
        upstream_temperature = cp.PropsSI("T", "P", upstream_pressure, "Q", 0, fluid)
        SATURATED = True
    elif abs(upstream_temperature - Tsat) < 1e-3:
        SATURATED = True
    else:
        SATURATED = False

    if SATURATED:
        # upstream conditions are saturated
        rho = cp.PropsSI("D", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)
        V = 1/rho
        drho_dt = cp.PropsSI("d(D)/d(T)|P", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)
        dvdt = -drho_dt / rho**2 
        dpdt = cp.PropsSI("d(P)/d(T)|D", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)
        c_p = cp.PropsSI("CPMASS", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)
        c_v = cp.PropsSI("CVMASS", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)
        Z = cp.PropsSI("Z", "P", upstream_pressure, "Q", upstream_vapor_quality, fluid)
    else:
        # upstream conditions are supercharged
        rho = cp.PropsSI("D", "P", upstream_pressure, "T", upstream_temperature, fluid)
        V = 1/rho
        drho_dt = cp.PropsSI("d(D)/d(T)|P", "P", upstream_pressure, "T", upstream_temperature, fluid)
        dvdt = -drho_dt / rho**2  
        dpdt = cp.PropsSI("d(P)/d(T)|D", "P", upstream_pressure, "T", upstream_temperature, fluid)
        c_p = cp.PropsSI("CPMASS", "P", upstream_pressure, "T", upstream_temperature, fluid)
        c_v = cp.PropsSI("CVMASS", "P", upstream_pressure, "T", upstream_temperature, fluid)
        Z = cp.PropsSI("Z", "P", upstream_pressure, "T", upstream_temperature, fluid)

    k = c_p/c_v    
    M = cp.PropsSI("M", fluid)
    Rs = R/M
    dzdt_p = upstream_pressure/Rs * (1/upstream_temperature * dvdt - V/upstream_temperature**2)
    dzdt_rho = V/Rs * (1/upstream_temperature * dpdt - upstream_pressure/upstream_temperature**2)

    n = k * ((Z + upstream_temperature*dzdt_rho) / (Z + upstream_temperature*dzdt_p))

    if n <= 1:
        raise ValueError("The calculated polytropic index n is less than or equal to 1. This indicates that the flow is not compressible, and the SPC model is not applicable.")

    dP = upstream_pressure - downstream_pressure
    P1 = upstream_pressure

    p_crit =  downstream_pressure / ( (2/(n+1))**(n/(n-1)) )

    if upstream_pressure > p_crit:
        # choked flow
        mdot_SPC = discharge_coefficient * orifice_area * sqrt(n*rho*P1*(2/(n+1))**((n+1)/(n-1)))
    else:
        # Compressibility correction factor
        #Y = sqrt(P1/2/dP * (2*n)/(n-1) * (1 - dP/P1)**(2/n) * (1 - (1 - dP/P1)**((n-1)/n)))
        #mdot_SPC = discharge_coefficient * orifice_area * Y * sqrt(2*rho*dP)

        mdot_SPC = discharge_coefficient * orifice_area * sqrt(2 * rho * P1 * (n/(n-1)) * ((downstream_pressure/P1)**(2/n) - (downstream_pressure/P1)**((n+1)/n)))
    return mdot_SPC

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
    _validate_pressure_inputs(upstream_pressure, downstream_pressure)
    _validate_orifice_inputs(orifice_area, discharge_coefficient)

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

    if downstream_pressure >= p_vap:
        raise ValueError("The Dyer model requires downstream pressure to be lower than the saturation pressure of the upstream state.")

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

def dyer_raw_compressible(
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
    _validate_pressure_inputs(upstream_pressure, downstream_pressure)
    _validate_orifice_inputs(orifice_area, discharge_coefficient)

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

    if downstream_pressure >= p_vap:
        raise ValueError("The Dyer model requires downstream pressure to be lower than the saturation pressure of the upstream state.")

    k = sqrt((upstream_pressure-downstream_pressure)/(p_vap-downstream_pressure))

    mdot_SPC = SPC(
        upstream_pressure, 
        downstream_pressure, 
        orifice_area, 
        discharge_coefficient, 
        upstream_temperature, 
        upstream_vapor_quality,
        fluid
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


    return (1/(1+k))*mdot_SPC + (1 - 1/(1+k))*mdot_HEM

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
    lower, upper = _bounded_downstream_pressure(upstream_pressure)
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
        bounds = (lower, upper), 
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

def dyer_compressible(
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
    lower, upper = _bounded_downstream_pressure(upstream_pressure)
    result = minimize_scalar(
        lambda p: -dyer_raw_compressible(
            upstream_pressure, 
            p, 
            orifice_area, 
            discharge_coefficient, 
            upstream_temperature, 
            upstream_vapor_quality,
            fluid
        ), 
        bounds = (lower, upper), 
        method="bounded"
    )

    if downstream_pressure > result.x:
        return dyer_raw_compressible(
            upstream_pressure, 
            downstream_pressure, 
            orifice_area, 
            discharge_coefficient, 
            upstream_temperature, 
            upstream_vapor_quality,
            fluid
        )
    else:
        return dyer_raw_compressible(
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
    _validate_pressure_inputs(upstream_pressure, downstream_pressure)
    _validate_orifice_inputs(orifice_area, discharge_coefficient)
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

    lower, upper = _bounded_downstream_pressure(upstream_pressure)
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
        bounds = (lower, upper), 
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
    _validate_pressure_inputs(upstream_pressure, downstream_pressure)
    _validate_orifice_inputs(orifice_area, discharge_coefficient)

    if upstream_vapor_quality < 0.0 or upstream_vapor_quality > 1.0:
        raise ValueError("Upstream vapor quality must be between 0 and 1.")

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

    mdot_SPI = SPC(
        upstream_pressure, 
        downstream_pressure, 
        orifice_area, 
        discharge_coefficient, 
        upstream_temperature, 
        upstream_vapor_quality,
        fluid
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
    lower, upper = _bounded_downstream_pressure(upstream_pressure)
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
        bounds = (lower, upper), 
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