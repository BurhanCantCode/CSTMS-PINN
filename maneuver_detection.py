import numpy as np
from scipy.stats import chi2
from dynamics import state_transition_matrix

class ManeuverDetector:
    def __init__(self, confidence_level=0.99):
        self.confidence_level = confidence_level
        self.chi2_threshold = chi2.ppf(confidence_level, df=3)
        self.maneuver_history = []
    
    def detect_maneuver(self, state_history, times, measurement_noise=1e-3):
        """Detect maneuvers using statistical hypothesis testing."""
        maneuvers = []
        n = len(state_history)
        
        if n < 3:
            return maneuvers
        
        # Initialize covariance
        P = np.eye(6) * measurement_noise
        
        for i in range(2, n):
            # Predict state using previous state
            dt = times[i] - times[i-1]
            prev_state = state_history[i-1]
            phi = state_transition_matrix(prev_state[:3], prev_state[3:], dt)
            
            # Propagate state and covariance
            pred_state = np.dot(phi, prev_state)
            P = np.dot(np.dot(phi, P), phi.T)
            
            # Compute innovation
            innovation = state_history[i] - pred_state
            
            # Compute innovation covariance
            S = P + np.eye(6) * measurement_noise
            
            # Compute Mahalanobis distance for velocity components
            vel_innovation = innovation[3:]
            vel_covariance = S[3:, 3:]
            d2 = np.dot(np.dot(vel_innovation.T, np.linalg.inv(vel_covariance)), 
                       vel_innovation)
            
            # Check if maneuver detected
            if d2 > self.chi2_threshold:
                maneuver = {
                    'time': times[i],
                    'state_before': state_history[i-1],
                    'state_after': state_history[i],
                    'delta_v': state_history[i][3:] - state_history[i-1][3:],
                    'confidence': 1 - chi2.cdf(d2, df=3)
                }
                maneuvers.append(maneuver)
                
                # Reset covariance after maneuver
                P = np.eye(6) * measurement_noise
        
        self.maneuver_history.extend(maneuvers)
        return maneuvers
    
    def estimate_maneuver_parameters(self, maneuver):
        """Estimate maneuver parameters (magnitude, direction, etc.)."""
        delta_v = maneuver['delta_v']
        delta_v_mag = np.linalg.norm(delta_v)
        
        # Compute maneuver direction in velocity frame
        r_before = maneuver['state_before'][:3]
        v_before = maneuver['state_before'][3:]
        
        # Orbital frame unit vectors
        h = np.cross(r_before, v_before)
        h_hat = h / np.linalg.norm(h)
        r_hat = r_before / np.linalg.norm(r_before)
        t_hat = np.cross(h_hat, r_hat)
        
        # Project delta-v onto orbital frame
        radial_dv = np.dot(delta_v, r_hat)
        transverse_dv = np.dot(delta_v, t_hat)
        normal_dv = np.dot(delta_v, h_hat)
        
        return {
            'magnitude': delta_v_mag,
            'radial_component': radial_dv,
            'transverse_component': transverse_dv,
            'normal_component': normal_dv,
            'time': maneuver['time'],
            'confidence': maneuver['confidence']
        }
    
    def classify_maneuver(self, maneuver_params, tolerance=0.1):
        """Classify maneuver type based on its parameters."""
        mag = maneuver_params['magnitude']
        rad = abs(maneuver_params['radial_component'])
        trn = abs(maneuver_params['transverse_component'])
        nrm = abs(maneuver_params['normal_component'])
        
        # Determine dominant direction
        components = [
            ('radial', rad),
            ('transverse', trn),
            ('normal', nrm)
        ]
        dominant = max(components, key=lambda x: x[1])[0]
        
        # Classify based on dominant direction and relative magnitudes
        if rad/mag > 1 - tolerance:
            return 'altitude_change'
        elif trn/mag > 1 - tolerance:
            return 'phasing'
        elif nrm/mag > 1 - tolerance:
            return 'plane_change'
        elif dominant == 'radial' and trn/mag > tolerance:
            return 'combined_altitude_phasing'
        elif dominant == 'transverse' and nrm/mag > tolerance:
            return 'combined_phasing_plane'
        else:
            return 'general'
    
    def get_maneuver_history(self):
        """Get the history of detected maneuvers with classifications."""
        classified_maneuvers = []
        
        for maneuver in self.maneuver_history:
            params = self.estimate_maneuver_parameters(maneuver)
            maneuver_type = self.classify_maneuver(params)
            
            classified_maneuvers.append({
                'time': params['time'],
                'magnitude': params['magnitude'],
                'components': {
                    'radial': params['radial_component'],
                    'transverse': params['transverse_component'],
                    'normal': params['normal_component']
                },
                'type': maneuver_type,
                'confidence': params['confidence']
            })
        
        return classified_maneuvers 