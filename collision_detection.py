import numpy as np
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta

class CollisionDetector:
    def __init__(self, warning_distance_km=100, critical_distance_km=10):
        self.warning_distance = warning_distance_km
        self.critical_distance = critical_distance_km
        self.tracked_objects = {}
    
    def add_object(self, object_id, trajectory, times):
        """Add a space object to track"""
        self.tracked_objects[object_id] = {
            'trajectory': np.array(trajectory),
            'times': np.array(times)
        }
    
    def compute_probability_of_collision(self, pos1, vel1, pos2, vel2, 
                                      covariance1=None, covariance2=None):
        """Compute probability of collision between two objects"""
        rel_pos = pos2 - pos1
        rel_vel = vel2 - vel1
        
        # Compute closest approach time
        t_ca = -np.dot(rel_pos, rel_vel) / np.dot(rel_vel, rel_vel)
        
        # Position at closest approach
        if t_ca > 0:
            pos1_ca = pos1 + vel1 * t_ca
            pos2_ca = pos2 + vel2 * t_ca
            min_dist = np.linalg.norm(pos2_ca - pos1_ca)
        else:
            min_dist = np.linalg.norm(rel_pos)
        
        # Simple probability based on distance
        if covariance1 is None or covariance2 is None:
            if min_dist < self.critical_distance:
                return 1.0
            elif min_dist < self.warning_distance:
                return 1.0 - (min_dist - self.critical_distance) / (self.warning_distance - self.critical_distance)
            return 0.0
        
        # More accurate probability using covariance
        combined_covariance = covariance1 + covariance2
        mahalanobis_dist = np.sqrt(
            rel_pos.T @ np.linalg.inv(combined_covariance) @ rel_pos
        )
        return 1.0 - np.exp(-0.5 * mahalanobis_dist)
    
    def detect_collisions(self, time_window_hours=24):
        """Detect potential collisions within time window"""
        collision_warnings = []
        
        # Get all pairs of objects
        object_ids = list(self.tracked_objects.keys())
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                obj1_id = object_ids[i]
                obj2_id = object_ids[j]
                
                obj1 = self.tracked_objects[obj1_id]
                obj2 = self.tracked_objects[obj2_id]
                
                # Compute distances between trajectories
                distances = cdist(obj1['trajectory'], obj2['trajectory'])
                
                # Find close approaches
                close_approaches = np.where(distances < self.warning_distance)
                
                for idx1, idx2 in zip(*close_approaches):
                    time1 = obj1['times'][idx1]
                    time2 = obj2['times'][idx2]
                    
                    # Skip if time difference is too large
                    if abs(time1 - time2) > time_window_hours * 3600:
                        continue
                    
                    # Compute collision probability
                    pos1 = obj1['trajectory'][idx1]
                    pos2 = obj2['trajectory'][idx2]
                    
                    # Estimate velocities
                    vel1 = np.zeros(3) if idx1 == 0 else \
                           (obj1['trajectory'][idx1] - obj1['trajectory'][idx1-1]) / \
                           (obj1['times'][idx1] - obj1['times'][idx1-1])
                    vel2 = np.zeros(3) if idx2 == 0 else \
                           (obj2['trajectory'][idx2] - obj2['trajectory'][idx2-1]) / \
                           (obj2['times'][idx2] - obj2['times'][idx2-1])
                    
                    prob = self.compute_probability_of_collision(pos1, vel1, pos2, vel2)
                    
                    if prob > 0:
                        collision_warnings.append({
                            'object1_id': obj1_id,
                            'object2_id': obj2_id,
                            'time': time1,
                            'probability': prob,
                            'distance': distances[idx1, idx2],
                            'position1': pos1,
                            'position2': pos2
                        })
        
        return collision_warnings
    
    def get_avoidance_maneuver(self, obj_id, collision_warning):
        """Compute avoidance maneuver for a given collision warning"""
        if collision_warning['object1_id'] == obj_id:
            pos = collision_warning['position1']
            other_pos = collision_warning['position2']
        else:
            pos = collision_warning['position2']
            other_pos = collision_warning['position1']
        
        # Compute avoidance direction (perpendicular to current trajectory)
        rel_pos = other_pos - pos
        rel_pos_norm = np.linalg.norm(rel_pos)
        
        # Compute required delta-v based on time to collision and distance
        time_to_collision = collision_warning['time'] - datetime.now().timestamp()
        if time_to_collision <= 0:
            return None
        
        # Simple avoidance maneuver perpendicular to relative position
        if abs(rel_pos[2]) < 1e-6:
            avoid_direction = np.array([rel_pos[1], -rel_pos[0], 0])
        else:
            avoid_direction = np.array([1, 1, -(rel_pos[0] + rel_pos[1])/rel_pos[2]])
        
        avoid_direction = avoid_direction / np.linalg.norm(avoid_direction)
        
        # Scale delta-v based on collision probability and time to collision
        delta_v_magnitude = (self.warning_distance - rel_pos_norm) / time_to_collision * \
                          collision_warning['probability']
        
        return avoid_direction * delta_v_magnitude 