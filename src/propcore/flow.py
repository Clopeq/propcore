from collections.abc import Callable
from math import isfinite, sqrt

import CoolProp.CoolProp as cp
from scipy.constants import R
from scipy.optimize import minimize_scalar


def _validate_pressure_inputs(p_up: float, p_down: float) -> None:
    """Validate upstream and downstream pressures.

    Sources
    -------
    Not applicable; this is an internal validation helper.

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Downstream absolute pressure in pascals.

    Returns
    -------
    None
        Raises ``ValueError`` if either pressure is invalid or p_down >= p_up.
    """
    if not isfinite(p_up) or not isfinite(p_down):
        raise ValueError("Upstream and downstream pressures must be finite numbers.")
    if p_up <= 0.0 or p_down <= 0.0:
        raise ValueError("Pressure inputs must be strictly positive.")
    if p_down >= p_up:
        raise ValueError("Downstream pressure must be strictly lower than upstream pressure.")


def _validate_orifice_inputs(
    A_orif: float,
    C_d: float,
    rho: float | None = None,
) -> None:
    """Validate orifice area, discharge coefficient, and optional density.

    Sources
    -------
    Not applicable; this is an internal validation helper.

    Arguments
    ---------
    A_orif : float
        Orifice area in square metres.
    C_d : float
        Dimensionless discharge coefficient.
    rho : float or None
        Optional fluid density in kilograms per cubic metre.

    Returns
    -------
    None
        Raises ``ValueError`` if any supplied value is invalid.
    """
    if not isfinite(A_orif) or not isfinite(C_d):
        raise ValueError("Orifice area and discharge coefficient must be finite numbers.")
    if A_orif <= 0.0:
        raise ValueError("Orifice area must be strictly positive.")
    if C_d <= 0.0:
        raise ValueError("Discharge coefficient must be strictly positive.")
    if rho is not None and (not isfinite(rho) or rho <= 0.0):
        raise ValueError("Density must be a finite, strictly positive value.")


def _bounded_downstream_pressure(p_up: float) -> tuple[float, float]:
    """Return a positive pressure interval for critical-flow optimization.

    Sources
    -------
    Not applicable; this is an internal numerical helper.

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.

    Returns
    -------
    tuple[float, float]
        Lower and upper downstream-pressure bounds in pascals.
    """
    if not isfinite(p_up) or p_up <= 0.0:
        raise ValueError("Upstream pressure must be a finite, strictly positive value.")
    p_min = max(1e-12, 1e-4 * p_up)
    if p_min >= p_up:
        p_min = p_up * 0.5
    return p_min, p_up


def _critical_flow_rate(
    raw_model: Callable[..., float],
    p_up: float,
    p_down: float,
    A_orif: float,
    C_d: float,
    T_up: float | str,
    Q_up: float,
    fluid: str,
) -> float:
    """Find a model's critical flow and evaluate it at the requested pressure.

    Sources
    -------
    Not applicable; this is an internal numerical helper.

    Arguments
    ---------
    raw_model : callable
        Raw mass-flow model accepting pressure, geometry, state, and fluid inputs.
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Requested downstream absolute pressure in pascals.
    A_orif : float
        Orifice area in square metres.
    C_d : float
        Dimensionless discharge coefficient.
    T_up : float or str
        Upstream temperature in kelvins, or ``"SAT"`` for saturation conditions.
    Q_up : float
        Upstream vapor quality, dimensionless and between 0 and 1.
    fluid : str
        CoolProp fluid identifier.

    Returns
    -------
    float
        Mass flow rate in kilograms per second, limited to the critical-flow value.
    """
    p_min, p_max = _bounded_downstream_pressure(p_up)
    result = minimize_scalar(
        lambda p: -raw_model(
            p_up,
            p,
            A_orif,
            C_d,
            T_up,
            Q_up,
            fluid,
        ),
        bounds=(p_min, p_max),
        method="bounded",
    )

    # Keep requested pressure above critical; otherwise evaluate at the critical pressure.
    p_eval = p_down if p_down > result.x else result.x
    return raw_model(
        p_up,
        p_eval,
        A_orif,
        C_d,
        T_up,
        Q_up,
        fluid,
    )


def SPI(
    p_up: float,
    p_down: float,
    A_orif: float,
    C_d: float,
    rho_up: float,
) -> float:
    """
    Calculate orifice mass flow using the single-phase incompressible (SPI) model.

    Sources
    -------
    Bernoulli principle and continuity equation.

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Downstream absolute pressure in pascals.
    A_orif : float
        Orifice area in square metres.
    C_d : float
        Dimensionless discharge coefficient.
    rho_up : float
        Upstream fluid density in kilograms per cubic metre.

    Returns
    -------
    float
        Mass flow rate through the orifice in kilograms per second.
    """
    _validate_pressure_inputs(p_up, p_down)
    _validate_orifice_inputs(A_orif, C_d, rho_up)
    # Bernoulli velocity times density and area gives the incompressible mass flow.
    return C_d * A_orif * sqrt(2 * (p_up - p_down) * rho_up)


def SPC(
    p_up: float,
    p_down: float,
    A_orif: float,
    C_d: float,
    T_up: float | str = "SAT",
    Q_up: float = 0.0,
    fluid: str = "N2O",
) -> float:
    """
    Calculate orifice mass flow using the single-phase compressible (SPC) model.

    Sources
    -------
    La Luna, S.; Foletti, N.; Magni, L.; Zuin, D.; Maggi, F. "A Two-Phase
    Mass Flow Rate Model for Nitrous Oxide Based on Void Fraction." Aerospace
    2022, 9(12), 828. https://doi.org/10.3390/aerospace9120828

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Downstream absolute pressure in pascals.
    A_orif : float
        Orifice area in square metres.
    C_d : float
        Dimensionless discharge coefficient.
    T_up : float or str
        Upstream temperature in kelvins, or ``"SAT"`` for saturation conditions.
    Q_up : float
        Upstream vapor quality, dimensionless and between 0 and 1.
    fluid : str
        CoolProp fluid identifier.

    Returns
    -------
    float
        Mass flow rate through the orifice in kilograms per second.
    """
    _validate_pressure_inputs(p_up, p_down)
    _validate_orifice_inputs(A_orif, C_d)

    T_sat = cp.PropsSI("T", "P", p_up, "Q", 0, fluid)  # Saturation temperature at p_up [K].
    if T_up == "SAT":
        T_up = T_sat
        is_sat = True  # Indicates whether the upstream state uses quality as its state pair.
    else:
        is_sat = abs(T_up - T_sat) < 1e-3

    # Select quality for saturated states and temperature for superheated states.
    state_key, state_val = ("Q", Q_up) if is_sat else ("T", T_up)
    rho_up = cp.PropsSI(
        "D", "P", p_up, state_key, state_val, fluid
    )  # Upstream density [kg/m^3].
    V_up = 1 / rho_up  # Upstream specific volume [m^3/kg].
    drho_dT_p = cp.PropsSI(
        "d(D)/d(T)|P", "P", p_up, state_key, state_val, fluid
    )  # Density derivative with respect to temperature at constant pressure.
    dV_dT_p = -drho_dT_p / rho_up**2  # Specific-volume derivative at constant pressure.
    dp_dT_rho = cp.PropsSI(
        "d(P)/d(T)|D", "P", p_up, state_key, state_val, fluid
    )  # Pressure derivative with respect to temperature at constant density.
    c_p = cp.PropsSI(
        "CPMASS", "P", p_up, state_key, state_val, fluid
    )  # Specific heat capacity at constant pressure [J/(kg K)].
    c_v = cp.PropsSI(
        "CVMASS", "P", p_up, state_key, state_val, fluid
    )  # Specific heat capacity at constant volume [J/(kg K)].
    Z_up = cp.PropsSI(
        "Z", "P", p_up, state_key, state_val, fluid
    )  # Upstream compressibility factor.

    k = c_p / c_v  # Heat-capacity ratio.
    M = cp.PropsSI("M", fluid)  # Fluid molar mass [kg/mol].
    R_s = R / M  # Specific gas constant [J/(kg K)].
    dZ_dT_p = p_up / R_s * (
        1 / T_up * dV_dT_p
        - V_up / T_up**2
    )  # Compressibility derivative with respect to temperature at constant pressure.
    dZ_dT_rho = V_up / R_s * (
        1 / T_up * dp_dT_rho
        - p_up / T_up**2
    )  # Compressibility derivative with respect to temperature at constant density.

    n = k * (
        (Z_up + T_up * dZ_dT_rho)
        / (Z_up + T_up * dZ_dT_p)
    )

    if n <= 1:
        raise ValueError(
            "The calculated polytropic index n is less than or equal to 1. "
            "This indicates that the flow is not compressible, and the SPC model is not applicable."
        )

    p_crit = p_down / ((2 / (n + 1)) ** (n / (n - 1)))  # Critical pressure threshold [Pa].

    if p_up > p_crit:
        # Choked mass flow evaluated at the critical pressure.
        return C_d * A_orif * sqrt(
            n * rho_up * p_up * (2 / (n + 1)) ** ((n + 1) / (n - 1))
        )
    else:
        p_ratio = p_down / p_up  # Downstream-to-upstream absolute pressure ratio.
        return C_d * A_orif * sqrt(
            2
            * rho_up
            * p_up
            * (n / (n - 1))
            * (
                p_ratio ** (2 / n)
                - p_ratio ** ((n + 1) / n)
            )
        )


def _dyer_upstream_state(
    p_up: float,
    T_up: float | str,
    Q_up: float,
    fluid: str,
) -> tuple[float, float]:
    """Return the vapor pressure and density for Dyer's upstream state.

    Sources
    -------
    Not applicable; this is an internal state-property helper.

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    T_up : float or str
        Upstream temperature in kelvins, or ``\"SAT\"`` for saturation conditions.
    Q_up : float
        Upstream vapor quality, dimensionless and between 0 and 1.
    fluid : str
        CoolProp fluid identifier.

    Returns
    -------
    tuple[float, float]
        Saturation pressure in pascals and upstream density in kilograms per cubic metre.
    """
    T_sat = cp.PropsSI("T", "P", p_up, "Q", 0, fluid)  # Saturation temperature at p_up [K].
    if T_up == "SAT" or abs(
        T_up - T_sat
    ) < 1e-3:
        p_vap = p_up  # Saturation pressure of a saturated upstream state [Pa].
        rho_up = cp.PropsSI("D", "P", p_up, "Q", Q_up, fluid)  # Upstream density [kg/m^3].
    else:
        p_vap = cp.PropsSI("P", "T", T_up, "Q", 0, fluid)  # Saturation pressure at T_up [Pa].
        rho_up = cp.PropsSI("D", "P", p_up, "T", T_up, fluid)  # Upstream density [kg/m^3].
        if p_up < p_vap:
            raise ValueError(
                "Upstream pressure is below the vapor pressure at the given upstream temperature. "
                "The Dyer model is not valid for this case."
            )
    return p_vap, rho_up


def dyer_raw(
    p_up: float,
    p_down: float,
    A_orif: float,
    C_d: float,
    T_up: float | str = "SAT",
    Q_up: float = 0.0,
    fluid: str = "N2O",
) -> float:
    """
    Calculate raw Dyer-model mass flow using the SPI incompressible branch.

    Sources
    -------
    Dyer model notes: https://drive.proton.me/u/1/RlBXyiB0sU0wGFCPvRY1wA/file/Ip2cezA0TTTDBmD6Eq_0ZA
    Dyer, J.; Zilliac, G.; Sadhwani, A.; Karabeyoglu, A.; Cantwell, B.
    "Modeling Feed System Flow Physics for Self-Pressurizing Propellants."
    AIAA 2007-5702. https://doi.org/10.2514/6.2007-5702

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Downstream absolute pressure in pascals.
    A_orif : float
        Orifice area in square metres.
    C_d : float
        Dimensionless discharge coefficient.
    T_up : float or str
        Upstream temperature in kelvins, or ``\"SAT\"`` for saturation conditions.
    Q_up : float
        Upstream vapor quality, dimensionless and between 0 and 1.
    fluid : str
        CoolProp fluid identifier.

    Returns
    -------
    float
        Dyer-model mass flow rate in kilograms per second.
    """
    _validate_pressure_inputs(p_up, p_down)
    _validate_orifice_inputs(A_orif, C_d)
    p_vap, rho_up = _dyer_upstream_state(p_up, T_up, Q_up, fluid)

    if p_down >= p_vap:
        raise ValueError(
            "The Dyer model requires downstream pressure to be lower than the saturation "
            "pressure of the upstream state."
        )

    K_blend = sqrt((p_up - p_down) / (p_vap - p_down))  # Dyer pressure-based blend parameter.
    mdot_SPI = SPI(p_up, p_down, A_orif, C_d, rho_up)  # Incompressible branch mass flow [kg/s].
    mdot_HEM = HEM_raw(p_up, p_down, A_orif, C_d, T_up, Q_up, fluid)  # HEM branch mass flow [kg/s].
    w_SPI = 1 / (1 + K_blend)  # Weight assigned to the SPI branch.
    return w_SPI * mdot_SPI + (1 - w_SPI) * mdot_HEM


def dyer_raw_compressible(
    p_up: float,
    p_down: float,
    A_orif: float,
    C_d: float,
    T_up: float | str = "SAT",
    Q_up: float = 0.0,
    fluid: str = "N2O",
) -> float:
    """
    Calculate raw Dyer-variant mass flow using the SPC compressible branch.

    Sources
    -------
    This compressible variation is an original implementation with no separate
    reference. The Dyer model notes are available at:
    https://drive.proton.me/u/1/RlBXyiB0sU0wGFCPvRY1wA/file/Ip2cezA0TTTDBmD6Eq_0ZA

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Downstream absolute pressure in pascals.
    A_orif : float
        Orifice area in square metres.
    C_d : float
        Dimensionless discharge coefficient.
    T_up : float or str
        Upstream temperature in kelvins, or ``\"SAT\"`` for saturation conditions.
    Q_up : float
        Upstream vapor quality, dimensionless and between 0 and 1.
    fluid : str
        CoolProp fluid identifier.

    Returns
    -------
    float
        Compressible Dyer-variant mass flow rate in kilograms per second.
    """
    _validate_pressure_inputs(p_up, p_down)
    _validate_orifice_inputs(A_orif, C_d)
    p_vap, _ = _dyer_upstream_state(p_up, T_up, Q_up, fluid)

    if p_down >= p_vap:
        raise ValueError(
            "The Dyer model requires downstream pressure to be lower than the saturation "
            "pressure of the upstream state."
        )

    K_blend = sqrt((p_up - p_down) / (p_vap - p_down))  # Dyer pressure-based blend parameter.
    mdot_SPC = SPC(p_up, p_down, A_orif, C_d, T_up, Q_up, fluid)  # Compressible branch [kg/s].
    mdot_HEM = HEM_raw(p_up, p_down, A_orif, C_d, T_up, Q_up, fluid)  # HEM branch [kg/s].
    w_SPC = 1 / (1 + K_blend)  # Weight assigned to the SPC branch.
    return w_SPC * mdot_SPC + (1 - w_SPC) * mdot_HEM

def dyer(
    p_up: float,
    p_down: float,
    A_orif: float,
    C_d: float,
    T_up: float | str = "SAT",
    Q_up: float = 0.0,
    fluid: str = "N2O",
) -> float:
    """
    Calculate Dyer-model mass flow, including its critical-flow limit.

    Sources
    -------
    Dyer model notes: https://drive.proton.me/u/1/RlBXyiB0sU0wGFCPvRY1wA/file/Ip2cezA0TTTDBmD6Eq_0ZA
    Dyer, J.; Zilliac, G.; Sadhwani, A.; Karabeyoglu, A.; Cantwell, B.
    "Modeling Feed System Flow Physics for Self-Pressurizing Propellants."
    AIAA 2007-5702. https://doi.org/10.2514/6.2007-5702

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Downstream absolute pressure in pascals.
    A_orif : float
        Orifice area in square metres.
    C_d : float
        Dimensionless discharge coefficient.
    T_up : float or str
        Upstream temperature in kelvins, or ``\"SAT\"`` for saturation conditions.
    Q_up : float
        Upstream vapor quality, dimensionless and between 0 and 1.
    fluid : str
        CoolProp fluid identifier.

    Returns
    -------
    float
        Critical-limited Dyer mass flow rate in kilograms per second.
    """
    return _critical_flow_rate(dyer_raw, p_up, p_down, A_orif, C_d, T_up, Q_up, fluid)


def dyer_compressible(
    p_up: float,
    p_down: float,
    A_orif: float,
    C_d: float,
    T_up: float | str = "SAT",
    Q_up: float = 0.0,
    fluid: str = "N2O",
) -> float:
    """
    Calculate compressible Dyer-variant mass flow, including critical flow.

    Sources
    -------
    This compressible variation is an original implementation with no separate
    reference. The Dyer model notes are available at:
    https://drive.proton.me/u/1/RlBXyiB0sU0wGFCPvRY1wA/file/Ip2cezA0TTTDBmD6Eq_0ZA

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Downstream absolute pressure in pascals.
    A_orif : float
        Orifice area in square metres.
    C_d : float
        Dimensionless discharge coefficient.
    T_up : float or str
        Upstream temperature in kelvins, or ``\"SAT\"`` for saturation conditions.
    Q_up : float
        Upstream vapor quality, dimensionless and between 0 and 1.
    fluid : str
        CoolProp fluid identifier.

    Returns
    -------
    float
        Critical-limited compressible Dyer-variant mass flow in kilograms per second.
    """
    return _critical_flow_rate(
        dyer_raw_compressible, p_up, p_down, A_orif, C_d, T_up, Q_up, fluid
    )


def HEM_raw(
    p_up: float,
    p_down: float,
    A_orif: float = 1.0,
    C_d: float = 1.0,
    T_up: float | str = "SAT",
    Q_up: float = 0.0,
    fluid: str = "N2O",
) -> float:
    """
    Calculate raw mass flow with the homogeneous equilibrium model (HEM).

    Assumes a homogeneous two-phase mixture, thermal equilibrium, no slip, and
    isentropic flow. With A_orif=1 m^2 and C_d=1, the result numerically equals
    mass flux in kg/(m^2 s).

    Sources
    -------
    La Luna, S.; Foletti, N.; Magni, L.; Zuin, D.; Maggi, F. "A Two-Phase
    Mass Flow Rate Model for Nitrous Oxide Based on Void Fraction." Aerospace
    2022, 9(12), 828, Section 4.3, equations 13-16.
    https://doi.org/10.3390/aerospace9120828
    Dyer, J.; Zilliac, G.; Sadhwani, A.; Karabeyoglu, A.; Cantwell, B.
    "Modeling Feed System Flow Physics for Self-Pressurizing Propellants."
    AIAA 2007-5702. https://doi.org/10.2514/6.2007-5702
    Waxman, B.S. "An Investigation of Injectors for Use with High Vapor
    Pressure Propellants with Applications to Hybrid Rockets." Ph.D. thesis,
    Stanford University, 2014.

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Downstream absolute pressure in pascals.
    A_orif : float
        Orifice area in square metres; defaults to 1 m^2.
    C_d : float
        Dimensionless discharge coefficient; defaults to 1.
    T_up : float or str
        Upstream temperature in kelvins, or ``\"SAT\"`` for saturation conditions.
    Q_up : float
        Upstream vapor quality, dimensionless and between 0 and 1.
    fluid : str
        CoolProp fluid identifier.

    Returns
    -------
    float
        HEM mass flow rate in kilograms per second.
    """
    _validate_pressure_inputs(p_up, p_down)
    _validate_orifice_inputs(A_orif, C_d)
    if Q_up < 0.0 or Q_up > 1.0:
        raise ValueError("Upstream vapor quality must be between 0 and 1.")

    if T_up == "SAT":
        T_up = cp.PropsSI("T", "P", p_up, "Q", Q_up, fluid)  # Saturation temperature [K].

    T_sat_liq = cp.PropsSI("T", "P", p_up, "Q", 0, fluid)  # Saturation temperature [K].
    if abs(T_up - T_sat_liq) > 1e-3:
        # Temperature identifies the upstream state when it is not saturated.
        H_up = cp.PropsSI("H", "P", p_up, "T", T_up, fluid)  # Upstream enthalpy [J/kg].
        S_up = cp.PropsSI("S", "P", p_up, "T", T_up, fluid)  # Upstream entropy [J/(kg K)].
    else:
        # Quality identifies the upstream state on the saturation curve.
        H_up = cp.PropsSI("H", "P", p_up, "Q", Q_up, fluid)  # Upstream enthalpy [J/kg].
        S_up = cp.PropsSI("S", "P", p_up, "Q", Q_up, fluid)  # Upstream entropy [J/(kg K)].

    # Isentropic expansion fixes downstream entropy to the upstream entropy.
    H_down = cp.PropsSI("H", "P", p_down, "S", S_up, fluid)  # Downstream enthalpy [J/kg].
    rho_down = cp.PropsSI("D", "P", p_down, "S", S_up, fluid)  # Downstream density [kg/m^3].
    v_down = sqrt(2 * (H_up - H_down))  # Downstream velocity from enthalpy change [m/s].
    return C_d * A_orif * rho_down * v_down


def HEM(
    p_up: float,
    p_down: float,
    A_orif: float,
    C_d: float,
    T_up: float | str = "SAT",
    Q_up: float = 0.0,
    fluid: str = "N2O",
) -> float:
    """
    Calculate HEM mass flow, including its critical-flow limit.

    Sources
    -------
    La Luna, S.; Foletti, N.; Magni, L.; Zuin, D.; Maggi, F. "A Two-Phase
    Mass Flow Rate Model for Nitrous Oxide Based on Void Fraction." Aerospace
    2022, 9(12), 828, Section 4.3, equations 13-16.
    https://doi.org/10.3390/aerospace9120828
    Dyer, J.; Zilliac, G.; Sadhwani, A.; Karabeyoglu, A.; Cantwell, B.
    "Modeling Feed System Flow Physics for Self-Pressurizing Propellants."
    AIAA 2007-5702. https://doi.org/10.2514/6.2007-5702
    Waxman, B.S. "An Investigation of Injectors for Use with High Vapor
    Pressure Propellants with Applications to Hybrid Rockets." Ph.D. thesis,
    Stanford University, 2014.

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Downstream absolute pressure in pascals.
    A_orif : float
        Orifice area in square metres.
    C_d : float
        Dimensionless discharge coefficient.
    T_up : float or str
        Upstream temperature in kelvins, or ``\"SAT\"`` for saturation conditions.
    Q_up : float
        Upstream vapor quality, dimensionless and between 0 and 1.
    fluid : str
        CoolProp fluid identifier.

    Returns
    -------
    float
        Critical-limited HEM mass flow rate in kilograms per second.
    """
    return _critical_flow_rate(HEM_raw, p_up, p_down, A_orif, C_d, T_up, Q_up, fluid)


def FML_raw(
    p_up: float,
    p_down: float,
    A_orif: float,
    C_d: float,
    T_up: float | str = "SAT",
    Q_up: float = 0.0,
    fluid: str = "N2O",
) -> float:
    """
    Calculate raw mass flow using the Foletti–Magni–La Luna (FML) model.

    The model combines liquid- and vapor-phase estimates using downstream
    equilibrium quality and a slip-corrected void fraction.

    Sources
    -------
    La Luna, S.; Foletti, N.; Magni, L.; Zuin, D.; Maggi, F. "A Two-Phase
    Mass Flow Rate Model for Nitrous Oxide Based on Void Fraction." Aerospace
    2022, 9(12), 828. https://doi.org/10.3390/aerospace9120828

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Downstream absolute pressure in pascals.
    A_orif : float
        Orifice area in square metres.
    C_d : float
        Dimensionless discharge coefficient.
    T_up : float or str
        Upstream temperature in kelvins, or ``\"SAT\"`` for saturation conditions.
    Q_up : float
        Upstream vapor quality, dimensionless and between 0 and 1.
    fluid : str
        CoolProp fluid identifier.

    Returns
    -------
    float
        FML mass flow rate in kilograms per second.
    """
    _validate_pressure_inputs(p_up, p_down)
    _validate_orifice_inputs(A_orif, C_d)

    if Q_up < 0.0 or Q_up > 1.0:
        raise ValueError("Upstream vapor quality must be between 0 and 1.")

    if T_up == "SAT":
        T_up = cp.PropsSI("T", "P", p_up, "Q", 0, fluid)  # Upstream saturation temperature [K].
        is_sat = True  # Selects quality as the CoolProp upstream state pair.
    else:
        is_sat = False

    if is_sat:
        S_up = cp.PropsSI("S", "P", p_up, "Q", Q_up, fluid)  # Upstream entropy [J/(kg K)].
    else:
        S_up = cp.PropsSI("S", "P", p_up, "T", T_up, fluid)  # Upstream entropy [J/(kg K)].

    T_down = cp.PropsSI(
        "T", "P", p_down, "S", S_up, fluid
    )  # Isentropic downstream temperature [K].
    S_v_down = cp.PropsSI(
        "S", "P", p_down, "Q", 1, fluid
    )  # Saturated-vapor entropy downstream [J/(kg K)].
    S_l_down = cp.PropsSI(
        "S", "P", p_down, "Q", 0, fluid
    )  # Saturated-liquid entropy downstream [J/(kg K)].
    rho_v_down = cp.PropsSI(
        "D", "P", p_down, "Q", 1, fluid
    )  # Saturated-vapor density downstream [kg/m^3].
    rho_l_down = cp.PropsSI(
        "D", "P", p_down, "Q", 0, fluid
    )  # Saturated-liquid density downstream [kg/m^3].

    Q_down = (S_up - S_l_down) / (S_v_down - S_l_down)  # Isentropic downstream vapor quality.
    slip_ratio = (rho_l_down / rho_v_down) ** (1 / 3)  # Slip ratio from phase density ratio.
    Alpha_down = 1 / (
        1
        + (1 - Q_down) / Q_down * slip_ratio * rho_v_down / rho_l_down
    )

    mdot_SPC = SPC(p_up, p_down, A_orif, C_d, T_up, Q_up, fluid)  # SPC mass-flow estimate [kg/s].
    mdot_HEM = HEM(p_up, p_down, A_orif, C_d, T_up, Q_up, fluid)  # HEM mass-flow estimate [kg/s].
    mdot_l = (1 - Alpha_down) * mdot_SPC + Alpha_down * mdot_HEM  # Liquid-phase estimate [kg/s].
    mdot_v = Alpha_down * mdot_SPC + (1 - Alpha_down) * mdot_HEM  # Vapor-phase estimate [kg/s].
    return Q_up * mdot_v + (1 - Q_up) * mdot_l


def FML(
    p_up: float,
    p_down: float,
    A_orif: float,
    C_d: float,
    T_up: float | str = "SAT",
    Q_up: float = 0.0,
    fluid: str = "N2O",
) -> float:
    """
    Calculate FML mass flow, including its critical-flow limit.

    Sources
    -------
    La Luna, S.; Foletti, N.; Magni, L.; Zuin, D.; Maggi, F. "A Two-Phase
    Mass Flow Rate Model for Nitrous Oxide Based on Void Fraction." Aerospace
    2022, 9(12), 828. https://doi.org/10.3390/aerospace9120828

    Arguments
    ---------
    p_up : float
        Upstream absolute pressure in pascals.
    p_down : float
        Downstream absolute pressure in pascals.
    A_orif : float
        Orifice area in square metres.
    C_d : float
        Dimensionless discharge coefficient.
    T_up : float or str
        Upstream temperature in kelvins, or ``\"SAT\"`` for saturation conditions.
    Q_up : float
        Upstream vapor quality, dimensionless and between 0 and 1.
    fluid : str
        CoolProp fluid identifier.

    Returns
    -------
    float
        Critical-limited FML mass flow rate in kilograms per second.
    """
    return _critical_flow_rate(FML_raw, p_up, p_down, A_orif, C_d, T_up, Q_up, fluid)
