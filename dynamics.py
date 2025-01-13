import numpy as np
from scipy.spatial.transform import Rotation

# Physical constants
MU_EARTH = 398600.4418  # Earth's gravitational parameter (km³/s²)
MU_MOON = 4902.8000  # Moon's gravitational parameter (km³/s²)
MU_SUN = 132712440018  # Sun's gravitational parameter (km³/s²)
R_EARTH = 6378.137  # Earth's radius (km)
J2 = 0.00108263  # Earth's J2 coefficient
J3 = -2.54e-6    # Earth's J3 coefficient
J4 = -1.61e-6    # Earth's J4 coefficient
C_SOLAR = 299792.458  # Speed of light (km/s)
P_SOLAR = 4.56e-6  # Solar radiation pressure at 1 AU (N/m²)
AU = 149597870.700  # Astronomical Unit (km)

def two_body_acceleration(r):
    """Compute two-body acceleration."""
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    return -MU_EARTH * r / (r_norm**3)

def zonal_harmonics_acceleration(r, order=4):
    """Compute acceleration due to zonal harmonics (J2, J3, J4)."""
    x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    r_sqr = r_norm**2
    z_sqr = z**2
    
    # Pre-compute common terms
    Re_r = R_EARTH / r_norm
    z_r = z / r_norm
    
    # J2 terms
    j2_factor = 1.5 * J2 * MU_EARTH * (R_EARTH**2) / r_sqr**2
    j2_z_term = 5 * z_sqr / r_sqr - 1
    
    ax = j2_factor * x * j2_z_term
    ay = j2_factor * y * j2_z_term
    az = j2_factor * z * (5 * z_sqr / r_sqr - 3)
    
    if order >= 3:
        # J3 terms
        j3_factor = 2.5 * J3 * MU_EARTH * (R_EARTH**3) / r_sqr**3
        j3_z_term = 7 * z_sqr / r_sqr - 3
        
        ax += j3_factor * x * z * j3_z_term
        ay += j3_factor * y * z * j3_z_term
        az += j3_factor * (j3_z_term * z_sqr / r_sqr - 6 * z_sqr / r_sqr + 2)
    
    if order >= 4:
        # J4 terms
        j4_factor = 1.875 * J4 * MU_EARTH * (R_EARTH**4) / r_sqr**4
        j4_z_term = 9 * z_sqr / r_sqr - 3
        
        ax += j4_factor * x * (35 * z_sqr**2 / r_sqr**2 - 30 * z_sqr / r_sqr + 3)
        ay += j4_factor * y * (35 * z_sqr**2 / r_sqr**2 - 30 * z_sqr / r_sqr + 3)
        az += j4_factor * z * (63 * z_sqr**2 / r_sqr**2 - 70 * z_sqr / r_sqr + 15)
    
    return np.concatenate([ax, ay, az], axis=1)

def third_body_acceleration(r, r_third, mu_third):
    """Compute third-body perturbation acceleration."""
    # Vector from spacecraft to third body
    r_sc_third = r_third - r
    
    # Norms
    r_sc_third_norm = np.linalg.norm(r_sc_third, axis=1, keepdims=True)
    r_third_norm = np.linalg.norm(r_third, axis=1, keepdims=True)
    
    # Compute acceleration
    acc = mu_third * (
        r_sc_third / r_sc_third_norm**3 - 
        r_third / r_third_norm**3
    )
    return acc

def solar_radiation_pressure(r, r_sun, area_mass_ratio=0.01, cr=1.8):
    """Compute solar radiation pressure acceleration."""
    # Vector from spacecraft to Sun
    r_sc_sun = r_sun - r
    r_sc_sun_norm = np.linalg.norm(r_sc_sun, axis=1, keepdims=True)
    
    # Check if spacecraft is in Earth's shadow
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    cos_angle = np.sum(r * r_sc_sun, axis=1, keepdims=True) / (r_norm * r_sc_sun_norm)
    
    # Shadow function (umbra only for simplicity)
    in_shadow = (cos_angle > 0) & (r_norm * np.sqrt(1 - cos_angle**2) < R_EARTH)
    
    # Compute acceleration magnitude
    acc_mag = P_SOLAR * cr * area_mass_ratio * (AU / r_sc_sun_norm)**2
    acc_mag[in_shadow] = 0
    
    # Direction is away from Sun
    return acc_mag * r_sc_sun / r_sc_sun_norm

def atmospheric_drag(r, v, area_mass_ratio=0.01, cd=2.2):
    """Compute atmospheric drag acceleration."""
    # Simple exponential atmosphere model
    h = np.linalg.norm(r, axis=1, keepdims=True) - R_EARTH
    rho = 1.225 * np.exp(-h / 7.249)  # kg/km³
    
    # Relative velocity (assuming rotating atmosphere)
    omega_earth = np.array([0, 0, 7.2921159e-5])  # Earth's rotation rate (rad/s)
    v_rel = v - np.cross(omega_earth, r)
    v_rel_norm = np.linalg.norm(v_rel, axis=1, keepdims=True)
    
    # Compute acceleration
    acc_mag = -0.5 * cd * area_mass_ratio * rho * v_rel_norm
    return acc_mag * v_rel / v_rel_norm

def total_acceleration(r, v=None, r_moon=None, r_sun=None, area_mass_ratio=0.01):
    """Compute total acceleration including all perturbations."""
    # Initialize with two-body and zonal harmonics
    acc = two_body_acceleration(r) + zonal_harmonics_acceleration(r)
    
    # Add third-body perturbations if positions are provided
    if r_moon is not None:
        acc += third_body_acceleration(r, r_moon, MU_MOON)
    if r_sun is not None:
        acc += third_body_acceleration(r, r_sun, MU_SUN)
        acc += solar_radiation_pressure(r, r_sun, area_mass_ratio)
    
    # Add atmospheric drag if velocity is provided
    if v is not None:
        acc += atmospheric_drag(r, v, area_mass_ratio)
    
    return acc

def state_transition_matrix(r, v, dt):
    """Compute state transition matrix for linearized dynamics."""
    mu = MU_EARTH
    r_norm = np.linalg.norm(r)
    
    # Compute gradient of gravitational acceleration
    grad_g = mu * (
        3 * np.outer(r, r) / r_norm**5 -
        np.eye(3) / r_norm**3
    )
    
    # Build state transition matrix
    phi = np.zeros((6, 6))
    phi[:3, :3] = np.eye(3)
    phi[:3, 3:] = np.eye(3) * dt
    phi[3:, :3] = grad_g * dt
    phi[3:, 3:] = np.eye(3)
    
    return phi