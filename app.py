import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from astropy.time import Time

from pinn_model import PINN
from visualization import TrajectoryVisualizer
from collision_detection import CollisionDetector
from maneuver_detection import ManeuverDetector
from data_utils import load_data, preprocess_data, generate_initial_orbit
from dynamics import total_acceleration
from fetch_horizons_data import fetch_horizons_data

class CSTMSApp:
    def __init__(self):
        self.pinn_model = PINN()
        self.visualizer = TrajectoryVisualizer()
        self.collision_detector = CollisionDetector()
        self.maneuver_detector = ManeuverDetector()
        self.tracked_objects = {}
        self.state_history = []
        self.time_history = []
    
    def run(self):
        st.set_page_config(page_title="Cislunar Space Traffic Management System",
                          layout="wide")
        
        st.title("Cislunar Space Traffic Management System")
        
        # Sidebar for controls
        with st.sidebar:
            st.header("Controls")
            
            # Data source selection
            data_source = st.radio(
                "Select Data Source",
                ["Upload File", "Fetch from HORIZONS"]
            )
            
            if data_source == "Upload File":
                # Data upload
                uploaded_file = st.file_uploader("Upload observation data (MPC format)",
                                               type=['txt', 'dat', 'csv'])
                
                if uploaded_file:
                    data = load_data(uploaded_file)
                    times, ra, dec = preprocess_data(data)
                    
            else:  # Fetch from HORIZONS
                st.subheader("HORIZONS Data Parameters")
                
                # Target body selection
                target_options = {
                    "301": "Moon",
                    "399": "Earth",
                    "499": "Mars",
                    "-31": "Goldstone",
                    "-248": "Hubble Space Telescope"
                }
                target_body = st.selectbox(
                    "Select Target Body",
                    options=list(target_options.keys()),
                    format_func=lambda x: f"{target_options[x]} ({x})"
                )
                
                # Date range selection
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        datetime.now().date()
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        (datetime.now() + timedelta(days=7)).date()
                    )
                
                # Time step selection
                step_size = st.selectbox(
                    "Time Step",
                    ["15m", "30m", "1h", "2h", "6h", "12h", "1d"],
                    index=2,
                    help="Time interval between observations"
                )
                
                if st.button("Fetch Data"):
                    with st.spinner("Fetching data from HORIZONS..."):
                        try:
                            data = fetch_horizons_data(
                                target_body=target_body,
                                start_time=start_date.strftime("%Y-%m-%d"),
                                stop_time=end_date.strftime("%Y-%m-%d"),
                                step_size=step_size
                            )
                            times = data[:, 0]  # Julian dates
                            ra = data[:, 1]     # Right ascension
                            dec = data[:, 2]    # Declination
                            st.success("Data fetched successfully!")
                            
                            # Store in session state
                            st.session_state['horizons_data'] = data
                            st.session_state['times'] = times
                            st.session_state['ra'] = ra
                            st.session_state['dec'] = dec
                            
                        except Exception as e:
                            st.error(f"Error fetching data: {str(e)}")
                            return
            
            # Only show these controls if we have data
            if 'times' in st.session_state and 'ra' in st.session_state and 'dec' in st.session_state:
                # Observer location
                st.subheader("Observer Location")
                lat = st.number_input("Latitude (deg)", value=0.0)
                lon = st.number_input("Longitude (deg)", value=0.0)
                alt = st.number_input("Altitude (m)", value=0.0)
                
                # Initial orbit estimation
                if st.button("Generate Initial Orbit"):
                    with st.spinner("Estimating initial orbit..."):
                        try:
                            # Convert time to astropy Time object
                            time = Time(st.session_state['times'][0], format='jd')
                            
                            initial_state = generate_initial_orbit(
                                st.session_state['ra'][0],
                                st.session_state['dec'][0],
                                42000,  # Assume GEO
                                lat, lon, alt,
                                time
                            )
                            st.session_state['initial_state'] = initial_state
                            st.success("Initial orbit estimated!")
                        except Exception as e:
                            st.error(f"Error generating initial orbit: {str(e)}")
                
                # Train PINN model
                if st.button("Process Data"):
                    if 'initial_state' not in st.session_state:
                        st.error("Please generate initial orbit first!")
                    else:
                        with st.spinner("Training PINN model..."):
                            self.pinn_model.fit(
                                st.session_state['times'],
                                st.session_state['ra'],
                                st.session_state['dec'],
                                np.ones_like(st.session_state['times'])*42000,
                                initial_state=st.session_state['initial_state']
                            )
                            st.success("Model trained successfully!")
            
            # Time window for predictions
            hours_ahead = st.slider("Prediction window (hours)", 1, 168, 24)
            
            # Collision detection settings
            st.header("Collision Detection")
            warning_distance = st.number_input("Warning distance (km)", 
                                             value=100, min_value=10)
            critical_distance = st.number_input("Critical distance (km)", 
                                             value=10, min_value=1)
            
            self.collision_detector.warning_distance = warning_distance
            self.collision_detector.critical_distance = critical_distance
            
            # Maneuver detection settings
            st.header("Maneuver Detection")
            confidence_level = st.slider("Confidence level", 0.9, 0.999, 0.99)
            self.maneuver_detector.confidence_level = confidence_level
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Trajectory Visualization")
            
            # Generate predictions if model is trained
            if hasattr(self.pinn_model, 'network'):
                current_time = datetime.now().timestamp()
                pred_times = np.linspace(current_time,
                                       current_time + hours_ahead*3600,
                                       100)
                
                positions, velocities = self.pinn_model.predict(pred_times)
                
                # Update state history
                self.state_history.append(
                    np.concatenate([positions[-1], velocities[-1]])
                )
                self.time_history.append(current_time)
                
                # Detect maneuvers
                if len(self.state_history) > 2:
                    maneuvers = self.maneuver_detector.detect_maneuver(
                        np.array(self.state_history),
                        np.array(self.time_history)
                    )
                    
                    # Visualize maneuvers
                    for maneuver in maneuvers:
                        params = self.maneuver_detector.estimate_maneuver_parameters(maneuver)
                        maneuver_type = self.maneuver_detector.classify_maneuver(params)
                        
                        self.visualizer.add_maneuver(
                            maneuver['state_before'][:3],
                            maneuver['state_after'][:3],
                            maneuver_type,
                            datetime.fromtimestamp(maneuver['time'])
                        )
                
                # Update visualization
                self.visualizer.add_trajectory(positions, "Spacecraft 1", 
                                            color='cyan',
                                            show_uncertainty=True)
                
                # Add to collision detector
                self.collision_detector.add_object("Spacecraft 1", 
                                                positions, pred_times)
                
                # Detect collisions
                collision_warnings = self.collision_detector.detect_collisions(
                    time_window_hours=hours_ahead
                )
                
                # Visualize collision warnings
                for warning in collision_warnings:
                    self.visualizer.add_collision_warning(
                        warning['position1'],
                        warning['position2'],
                        datetime.fromtimestamp(warning['time']).strftime(
                            '%Y-%m-%d %H:%M:%S'
                        )
                    )
                
                # Display the plot
                st.plotly_chart(self.visualizer.fig, use_container_width=True)
        
        with col2:
            # Maneuver Detection Tab
            st.header("Detected Maneuvers")
            maneuver_history = self.maneuver_detector.get_maneuver_history()
            
            if maneuver_history:
                for i, maneuver in enumerate(maneuver_history):
                    with st.expander(f"Maneuver {i+1} - {maneuver['type'].title()}"):
                        st.write(f"Time: {datetime.fromtimestamp(maneuver['time'])}")
                        st.write(f"Magnitude: {maneuver['magnitude']:.2f} km/s")
                        st.write("Components:")
                        st.write(f"- Radial: {maneuver['components']['radial']:.2f} km/s")
                        st.write(f"- Transverse: {maneuver['components']['transverse']:.2f} km/s")
                        st.write(f"- Normal: {maneuver['components']['normal']:.2f} km/s")
                        st.write(f"Confidence: {maneuver['confidence']:.2%}")
            
            # Collision Warnings Tab
            st.header("Collision Warnings")
            if 'collision_warnings' in locals():
                for i, warning in enumerate(collision_warnings):
                    with st.expander(f"Warning {i+1}"):
                        st.write(f"Time: {datetime.fromtimestamp(warning['time'])}")
                        st.write(f"Objects: {warning['object1_id']} - "
                               f"{warning['object2_id']}")
                        st.write(f"Probability: {warning['probability']:.2%}")
                        st.write(f"Distance: {warning['distance']:.2f} km")
                        
                        if st.button(f"Compute Avoidance Maneuver {i+1}"):
                            maneuver = self.collision_detector.get_avoidance_maneuver(
                                "Spacecraft 1", warning
                            )
                            if maneuver is not None:
                                st.write("Recommended maneuver (delta-v in km/s):")
                                st.write(f"X: {maneuver[0]:.4f}")
                                st.write(f"Y: {maneuver[1]:.4f}")
                                st.write(f"Z: {maneuver[2]:.4f}")
                            else:
                                st.warning("No avoidance maneuver available")
            
            st.header("System Status")
            st.write("Model Status: ",
                    "Trained" if hasattr(self.pinn_model, 'network')
                    else "Not Trained")
            st.write("Objects Tracked: ", len(self.collision_detector.tracked_objects))
            st.write("Maneuvers Detected: ", len(maneuver_history))
            st.write("Last Update: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == "__main__":
    app = CSTMSApp()
    app.run() 