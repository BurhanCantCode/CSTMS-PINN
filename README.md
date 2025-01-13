# Cislunar Space Traffic Management System (CSTMS)

A comprehensive system for managing space traffic in cislunar space using Physics-Informed Neural Networks (PINNs) for orbit determination and collision avoidance.

## Features

- **Physics-Informed Orbit Determination**: Uses PINNs to accurately predict spacecraft trajectories while respecting physical constraints
- **Real-time Collision Detection**: Monitors and predicts potential collisions between spacecraft
- **Avoidance Maneuver Planning**: Generates optimal avoidance maneuvers when collisions are predicted
- **Interactive 3D Visualization**: Beautiful and intuitive visualization of spacecraft trajectories and collision warnings
- **User-Friendly Interface**: Streamlit-based web interface for easy interaction with the system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cstms.git
cd cstms
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Use the system:
   - Upload observation data in MPC format using the sidebar
   - Click "Process Data" to train the PINN model
   - Adjust the prediction window and collision detection parameters as needed
   - View trajectories and collision warnings in the main display
   - Generate avoidance maneuvers when warnings are detected

## Data Format

The system accepts observation data in Minor Planet Center (MPC) 80-column format. Example:

```
00001    C2019 04 20.1234  11 22 33.44 +12 34 56.7          12.3 R      568
```

You can also use custom format files with columns: time, RA (degrees), DEC (degrees).

## System Components

- `app.py`: Main application file with Streamlit interface
- `pinn_model.py`: Implementation of the Physics-Informed Neural Network
- `dynamics.py`: Physical models for orbital dynamics
- `collision_detection.py`: Collision detection and avoidance algorithms
- `visualization.py`: 3D visualization tools using Plotly
- `data_utils.py`: Data loading and preprocessing utilities

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Research paper: "Physics-Informed Orbit Determination for Cislunar Space Applications"
- PyTorch team for the deep learning framework
- Plotly team for the visualization library
- Streamlit team for the web interface framework 