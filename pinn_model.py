import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dynamics import total_acceleration
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PINN(nn.Module):
    def __init__(self, hidden_layers=[128, 128, 128], activation=nn.Tanh()):
        super().__init__()
        
        # Network architecture
        layers = []
        input_dim = 1  # Time input
        
        for neurons in hidden_layers:
            layers.append(nn.Linear(input_dim, neurons))
            layers.append(activation)
            input_dim = neurons
        
        # Output layer for state vector (position and velocity)
        layers.append(nn.Linear(input_dim, 6))
        
        self.network = nn.Sequential(*layers)
        
        # Constants for non-dimensionalization
        self.mu = 398600.4418  # Earth's gravitational parameter (km³/s²)
        self.Re = 6378.137  # Earth's radius (km)
        self.T = np.sqrt(self.Re**3/self.mu)  # Time scale
        
        # Move model to available device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Data normalization parameters
        self.t_mean = 0.0
        self.t_std = 1.0
        self.r_scale = self.Re  # Use Earth radius as position scale
        self.v_scale = self.Re / self.T  # Characteristic velocity
    
    def normalize_time(self, t):
        """Normalize time input"""
        return (t - self.t_mean) / self.t_std
    
    def forward(self, t):
        """Forward pass to predict state vector"""
        t = t.to(self.device)
        t_norm = self.normalize_time(t)
        return self.network(t_norm.unsqueeze(-1))
    
    def get_derivatives(self, t, state):
        """Compute time derivatives of state vector"""
        # Scale state back to physical units
        pos = state[:, :3] * self.r_scale
        vel = state[:, 3:] * self.v_scale
        
        # Compute acceleration using physics (in physical units)
        acc = torch.tensor(total_acceleration(pos.detach().cpu().numpy()), 
                          device=self.device, dtype=torch.float32)
        
        # Scale back to normalized units
        acc = acc / self.r_scale * (self.t_std ** 2)
        vel = vel / self.r_scale * self.t_std
        
        return vel, acc
    
    def compute_gradients(self, t, state):
        """Compute gradients of state with respect to time"""
        batch_size = t.shape[0]
        
        # Create graph for each component
        grads = []
        for i in range(6):  # For each component of the state vector
            component = state[:, i]
            grad = torch.autograd.grad(
                component.sum(), t,
                create_graph=True,
                retain_graph=True
            )[0]
            grads.append(grad)
        
        # Stack gradients
        return torch.stack(grads, dim=1)  # [batch_size, 6]
    
    def physics_loss(self, t):
        """Physics-informed loss based on equations of motion"""
        t.requires_grad_(True)
        
        # Forward pass to get state
        state = self.forward(t)  # [batch_size, 6]
        
        # Compute gradients
        state_dot = self.compute_gradients(t, state)  # [batch_size, 6]
        
        # Get derivatives from physics
        vel_pred, acc_pred = self.get_derivatives(t, state)  # [batch_size, 3]
        
        # Split state_dot into position and velocity derivatives
        pos_dot = state_dot[:, :3]  # [batch_size, 3]
        vel_dot = state_dot[:, 3:]  # [batch_size, 3]
        
        # Physics residuals (in normalized units)
        pos_residual = torch.mean((pos_dot - vel_pred)**2)
        vel_residual = torch.mean((vel_dot - acc_pred)**2)
        
        return pos_residual + vel_residual
    
    def observation_loss(self, t, ra, dec, r):
        """Loss based on observation data"""
        state = self.forward(t)
        
        # Scale position to physical units
        pos = state[:, :3] * self.r_scale
        
        # Convert Cartesian to spherical coordinates
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r_pred = torch.sqrt(x**2 + y**2 + z**2)
        
        # Compute angles in degrees
        ra_pred = torch.atan2(y, x) * 180/np.pi
        dec_pred = torch.asin(z/r_pred) * 180/np.pi
        
        # Normalize predictions
        r_pred = r_pred / self.r_scale
        
        # Compute observation loss
        ra_loss = torch.mean((ra_pred - ra)**2) / (360.0**2)  # Normalize by max range
        dec_loss = torch.mean((dec_pred - dec)**2) / (180.0**2)
        r_loss = torch.mean((r_pred - r/self.r_scale)**2)
        
        return ra_loss + dec_loss + r_loss
    
    def initial_state_loss(self, t0, initial_state):
        """Loss based on initial state"""
        if initial_state is None:
            return 0.0
        
        state0 = self.forward(t0[0:1])
        
        # Normalize initial state
        initial_state_norm = torch.tensor([
            initial_state[0:3] / self.r_scale,  # Position
            initial_state[3:6] / self.v_scale   # Velocity
        ], dtype=torch.float32).to(self.device).flatten()
        
        return torch.mean((state0 - initial_state_norm)**2)
    
    def fit(self, times, ra, dec, r, epochs=1000, batch_size=32, learning_rate=1e-3, initial_state=None):
        """Train the model using both physics and observation losses"""
        # Compute normalization parameters
        self.t_mean = float(np.mean(times))
        self.t_std = float(np.std(times)) if np.std(times) > 0 else 1.0
        
        # Convert inputs to tensors
        t = torch.tensor(times, dtype=torch.float32).to(self.device)
        ra = torch.tensor(ra, dtype=torch.float32).to(self.device)
        dec = torch.tensor(dec, dtype=torch.float32).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        
        # Generate collocation points for physics loss
        t_physics = torch.linspace(t.min(), t.max(), 100).to(self.device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
        
        # Training loop
        pbar = tqdm(range(epochs), desc="Training PINN")
        best_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        for epoch in pbar:
            # Compute losses
            phys_loss = self.physics_loss(t_physics)
            obs_loss = self.observation_loss(t, ra, dec, r)
            init_loss = self.initial_state_loss(t, initial_state) if initial_state is not None else 0.0
            
            # Total loss with weights
            loss = phys_loss + obs_loss + init_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.state_dict(), 'models/best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("\nEarly stopping triggered!")
                    break
            
            pbar.set_postfix({
                'loss': f"{loss.item():.2e}",
                'phys_loss': f"{phys_loss.item():.2e}",
                'obs_loss': f"{obs_loss.item():.2e}",
                'init_loss': f"{init_loss if isinstance(init_loss, float) else init_loss.item():.2e}"
            })
    
    def predict(self, t):
        """Generate predictions for given times"""
        self.eval()
        with torch.no_grad():
            t = torch.tensor(t, dtype=torch.float32).to(self.device)
            state = self.forward(t)
            
            # Convert back to physical units
            pos = state[:, :3] * self.r_scale
            vel = state[:, 3:] * self.v_scale
            
            return pos.cpu().numpy(), vel.cpu().numpy()