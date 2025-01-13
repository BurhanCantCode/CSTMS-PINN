import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u

def parse_mpc_line(line):
    """Parse a single line of MPC 80-column format data."""
    try:
        # Extract fields from MPC format
        year = int(line[15:19])
        month = int(line[20:22])
        day = float(line[23:32])
        
        ra_h = int(line[32:34])
        ra_m = int(line[35:37])
        ra_s = float(line[38:44])
        
        dec_d = int(line[44:47])
        dec_m = int(line[48:50])
        dec_s = float(line[51:56])
        
        # Convert time to Julian Date
        dt = datetime(year, month, int(day))
        frac_day = day - int(day)
        dt = dt + timedelta(days=frac_day)
        t = Time(dt)
        
        # Convert RA/DEC to degrees
        ra = (ra_h + ra_m/60 + ra_s/3600) * 15  # Convert hours to degrees
        dec = dec_d + np.sign(dec_d)*dec_m/60 + np.sign(dec_d)*dec_s/3600
        
        return t.jd, ra, dec
    except:
        return None

def load_data(file_path, format='auto'):
    """Load observation data from various formats."""
    if format == 'auto':
        # Try to detect format from file extension
        if file_path.endswith('.txt'):
            format = 'mpc'
        elif file_path.endswith('.csv'):
            format = 'csv'
        elif file_path.endswith('.npy'):
            format = 'numpy'
        else:
            format = 'mpc'  # Default to MPC format
    
    if format == 'mpc':
        observations = []
        with open(file_path, 'r') as f:
            for line in f:
                if len(line) >= 80:  # MPC 80-column format
                    result = parse_mpc_line(line)
                    if result is not None:
                        observations.append(result)
        return np.array(observations)
    
    elif format == 'csv':
        # Assume CSV has columns: time, ra, dec
        df = pd.read_csv(file_path)
        required_cols = ['time', 'ra', 'dec']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        return df[required_cols].values
    
    elif format == 'numpy':
        return np.load(file_path)
    
    else:
        raise ValueError(f"Unsupported format: {format}")

def preprocess_data(data, time_scale='jd'):
    """Preprocess observation data."""
    times = data[:, 0]
    ra = data[:, 1]
    dec = data[:, 2]
    
    # Convert times if needed
    if time_scale == 'unix':
        # Convert Unix timestamps to Julian Date
        times = Time(times, format='unix').jd
    elif time_scale == 'iso':
        # Convert ISO format strings to Julian Date
        times = Time(times, format='iso').jd
    
    # Sort by time
    idx = np.argsort(times)
    times = times[idx]
    ra = ra[idx]
    dec = dec[idx]
    
    # Normalize times relative to first observation
    t0 = times[0]
    times = times - t0
    
    # Ensure angles are in proper range
    ra = ra % 360
    dec = np.clip(dec, -90, 90)
    
    return times, ra, dec

def topocentric_to_eci(ra, dec, r, observer_lat, observer_lon, observer_alt, time):
    """Convert topocentric coordinates to Earth-Centered Inertial (ECI)."""
    # Convert Julian Date to Time object
    if isinstance(time, (float, int)):
        time = Time(time, format='jd')
    
    # Create observer location
    observer = EarthLocation(
        lat=observer_lat*u.deg,
        lon=observer_lon*u.deg,
        height=observer_alt*u.m
    )
    
    # Create sky coordinates
    coords = SkyCoord(
        ra=ra*u.deg,
        dec=dec*u.deg,
        distance=r*u.km,
        frame='icrs'
    )
    
    # Convert to Earth-fixed frame
    altaz = coords.transform_to(AltAz(obstime=time, location=observer))
    
    # Convert to ECI
    eci = altaz.transform_to('gcrs')
    
    return np.array([
        eci.cartesian.x.value,
        eci.cartesian.y.value,
        eci.cartesian.z.value
    ])

def generate_initial_orbit(ra, dec, r, observer_lat, observer_lon, observer_alt, time):
    """Generate initial orbit estimate from optical observations."""
    # Convert to ECI coordinates
    pos = topocentric_to_eci(ra, dec, r, observer_lat, observer_lon, observer_alt, time)
    
    # Estimate velocity using circular orbit assumption
    r_norm = np.linalg.norm(pos)
    v_circ = np.sqrt(398600.4418 / r_norm)  # Circular orbit velocity
    
    # Compute velocity direction perpendicular to position and angular momentum
    h = np.cross([0, 0, 1], pos)  # Angular momentum assuming prograde orbit
    v_dir = np.cross(h, pos)
    v_dir = v_dir / np.linalg.norm(v_dir)
    
    vel = v_circ * v_dir
    
    return np.concatenate([pos, vel])