import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

class TrajectoryVisualizer:
    def __init__(self):
        self.fig = go.Figure()
        self._setup_layout()
        
        # Color scheme for maneuver types
        self.maneuver_colors = {
            'altitude_change': 'orange',
            'phasing': 'purple',
            'plane_change': 'green',
            'combined_altitude_phasing': 'yellow',
            'combined_phasing_plane': 'pink',
            'general': 'red'
        }
    
    def _setup_layout(self):
        """Setup the 3D layout with Earth at the center"""
        self.fig.update_layout(
            scene=dict(
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data',
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)'
            ),
            title='Cislunar Space Traffic Visualization',
            showlegend=True,
            template='plotly_dark',
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                      'fromcurrent': True}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                        'mode': 'immediate',
                                        'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'type': 'buttons',
                'showactive': False,
                'x': 0.1,
                'y': 0,
                'xanchor': 'right',
                'yanchor': 'top'
            }]
        )
        
        # Add Earth
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 6378.137 * np.outer(np.cos(u), np.sin(v))
        y = 6378.137 * np.outer(np.sin(u), np.sin(v))
        z = 6378.137 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(0,0,255)']],
            showscale=False,
            name='Earth',
            opacity=0.8
        ))
        
        # Add Moon's orbit (simplified circular orbit)
        theta = np.linspace(0, 2*np.pi, 100)
        r_moon = 384400  # Average lunar distance in km
        x_moon = r_moon * np.cos(theta)
        y_moon = r_moon * np.sin(theta)
        z_moon = np.zeros_like(theta)
        
        self.fig.add_trace(go.Scatter3d(
            x=x_moon, y=y_moon, z=z_moon,
            mode='lines',
            line=dict(color='gray', width=1),
            name="Moon's Orbit",
            opacity=0.5
        ))
    
    def add_trajectory(self, positions, label, color='white', show_uncertainty=False):
        """Add a trajectory to the visualization"""
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        self.fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            name=label,
            line=dict(color=color, width=2)
        ))
        
        # Add current position marker
        self.fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode='markers',
            name=f'{label} (current)',
            marker=dict(size=5, color=color)
        ))
        
        if show_uncertainty:
            # Add uncertainty ellipsoid at current position
            self._add_uncertainty_ellipsoid(
                center=[x[-1], y[-1], z[-1]],
                color=color,
                label=f'{label} uncertainty'
            )
    
    def _add_uncertainty_ellipsoid(self, center, color, label, scale=1000):
        """Add uncertainty ellipsoid around a point"""
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        x = center[0] + scale * np.outer(np.cos(u), np.sin(v))
        y = center[1] + scale * np.outer(np.sin(u), np.sin(v))
        z = center[2] + scale * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.2,
            showscale=False,
            name=label,
            colorscale=[[0, color], [1, color]]
        ))
    
    def add_collision_warning(self, pos1, pos2, time):
        """Add visual warning for potential collision"""
        x = [pos1[0], pos2[0]]
        y = [pos1[1], pos2[1]]
        z = [pos1[2], pos2[2]]
        
        self.fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=f'Collision Warning at {time}',
            line=dict(color='red', width=5),
            marker=dict(size=8, color='red')
        ))
    
    def add_maneuver(self, pos_before, pos_after, maneuver_type, time):
        """Add visualization of a detected maneuver"""
        color = self.maneuver_colors.get(maneuver_type, 'red')
        
        # Add maneuver vector
        x = [pos_before[0], pos_after[0]]
        y = [pos_before[1], pos_after[1]]
        z = [pos_before[2], pos_after[2]]
        
        self.fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=f'{maneuver_type.title()} at {time}',
            line=dict(color=color, width=3, dash='dash'),
            marker=dict(
                size=6,
                color=[color, color],
                symbol=['circle', 'arrow-up']
            )
        ))
        
        # Add maneuver plane for plane changes
        if maneuver_type == 'plane_change':
            self._add_maneuver_plane(pos_before, pos_after, color)
    
    def _add_maneuver_plane(self, pos1, pos2, color):
        """Add a semi-transparent plane to visualize plane changes"""
        # Create a plane that contains both position vectors
        v1 = pos1 / np.linalg.norm(pos1)
        v2 = pos2 / np.linalg.norm(pos2)
        
        # Normal to the plane
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        # Create a grid of points on the plane
        r = max(np.linalg.norm(pos1), np.linalg.norm(pos2))
        u = np.linspace(-r, r, 20)
        v = np.linspace(-r, r, 20)
        U, V = np.meshgrid(u, v)
        
        # Compute points on the plane
        X = U
        Y = V
        Z = (-normal[0]*U - normal[1]*V) / normal[2]
        
        self.fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.2,
            showscale=False,
            name='Maneuver Plane',
            colorscale=[[0, color], [1, color]]
        ))
    
    def show(self):
        """Display the visualization"""
        self.fig.show()
    
    def save(self, filename):
        """Save the visualization to HTML file"""
        self.fig.write_html(filename) 