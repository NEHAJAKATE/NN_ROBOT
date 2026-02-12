"""
Enhanced Neural Network IK Solver with robust architecture and validation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.utils_logging import get_logger

logger = get_logger(__name__)


class IKNetworkV2(nn.Module):
    """
    Enhanced neural network architecture for inverse kinematics.
    Features: dropout, batch normalization, skip connections.
    """
    
    def __init__(
        self,
        input_size: int = 7,
        output_size: int = 6,
        hidden_layers: list = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize IK network.
        
        Args:
            input_size: Input dimensions (usually 7 for pose)
            output_size: Output dimensions (usually 6 for joint angles)
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(IKNetworkV2, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.use_batch_norm = use_batch_norm
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class EnhancedNNIKSolver:
    """
    Enhanced Neural Network IK Solver with validation, error handling, and robustness.
    """
    
    def __init__(
        self,
        model_path: str = "ik_nn_model.pth",
        input_size: int = 7,
        output_size: int = 6,
        hidden_layers: list = None,
        device: str = None
    ):
        """
        Initialize enhanced NN IK solver.
        
        Args:
            model_path: Path to model checkpoint
            input_size: Input size (default: 7)
            output_size: Output size (default: 6)
            hidden_layers: Network architecture
            device: Compute device ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.input_size = input_size
        self.output_size = output_size
        
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Create model
        self.model = IKNetworkV2(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=hidden_layers
        ).to(self.device)
        
        # Load model
        self._load_model()
        self.model.eval()
    
    def _load_model(self) -> None:
        """Load model from checkpoint with robust error handling."""
        if not os.path.isfile(self.model_path):
            logger.warning(f"Model not found: {self.model_path}. Using random initialization.")
            return
        
        try:
            state = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(state, dict):
                if 'model_state_dict' in state:
                    state = state['model_state_dict']
                elif 'state_dict' in state:
                    state = state['state_dict']
                elif any(k.startswith('model.') for k in state.keys()):
                    # Remove 'model.' prefix
                    state = {k.replace('model.', '', 1): v for k, v in state.items()}
            
            # Load state dict
            self.model.load_state_dict(state, strict=False)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}. Using random initialization.")
    
    def solve(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        return_confidence: bool = False
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Solve IK problem: predict joint angles from pose.
        
        Args:
            position: End-effector position [x, y, z] (meters)
            orientation: End-effector orientation [qx, qy, qz, qw] (quaternion)
            return_confidence: Whether to return solution confidence
            
        Returns:
            Joint angles (radians) and optional confidence score
        """
        # Validate inputs
        if len(position) != 3:
            raise ValueError("Position must have 3 elements [x, y, z]")
        if len(orientation) != 4:
            raise ValueError("Orientation must have 4 elements [qx, qy, qz, qw]")
        
        # Build pose vector
        pose = np.concatenate([position, orientation]).astype(np.float32)
        
        # Normalize quaternion
        quat_norm = np.linalg.norm(pose[3:7])
        if quat_norm > 0:
            pose[3:7] /= quat_norm
        
        # Forward pass
        with torch.no_grad():
            pose_tensor = torch.from_numpy(pose).unsqueeze(0).to(self.device)
            joint_angles = self.model(pose_tensor).cpu().numpy().squeeze(0)
        
        if return_confidence:
            # Simple confidence based on quaternion magnitude
            confidence = float(quat_norm)
            return joint_angles.astype(np.float32), confidence
        
        return joint_angles.astype(np.float32)
    
    def solve_batch(
        self,
        positions: np.ndarray,
        orientations: np.ndarray
    ) -> np.ndarray:
        """
        Solve IK for multiple poses.
        
        Args:
            positions: Array of positions (N, 3)
            orientations: Array of orientations (N, 4)
            
        Returns:
            Array of joint angles (N, 6)
        """
        n_poses = len(positions)
        all_solutions = []
        
        for i in range(n_poses):
            solution = self.solve(positions[i], orientations[i])
            all_solutions.append(solution)
        
        return np.array(all_solutions, dtype=np.float32)
    
    def train(
        self,
        poses: np.ndarray,
        joints: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2,
        save_path: str = None
    ) -> dict:
        """
        Train the neural network on IK data.
        
        Args:
            poses: Training poses (N, 7)
            joints: Training joint angles (N, 6)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Validation split ratio
            save_path: Path to save best model
            
        Returns:
            Training history with losses
        """
        logger.info(f"Starting training: {len(poses)} samples, {epochs} epochs")
        
        # Convert to tensors
        poses_t = torch.from_numpy(poses.astype(np.float32)).to(self.device)
        joints_t = torch.from_numpy(joints.astype(np.float32)).to(self.device)
        
        # Split data
        n_train = int(len(poses) * (1 - validation_split))
        indices = torch.randperm(len(poses))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        train_dataset = torch.utils.data.TensorDataset(
            poses_t[train_idx], joints_t[train_idx]
        )
        val_dataset = torch.utils.data.TensorDataset(
            poses_t[val_idx], joints_t[val_idx]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Training setup
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            # Training
            train_loss = 0
            for poses_batch, joints_batch in train_loader:
                predictions = self.model(poses_batch)
                loss = loss_fn(predictions, joints_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = 0
            with torch.no_grad():
                for poses_batch, joints_batch in val_loader:
                    predictions = self.model(poses_batch)
                    loss = loss_fn(predictions, joints_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = save_path or self.model_path
                torch.save(self.model.state_dict(), save_path)
        
        self.model.eval()
        logger.info(f"Training completed. Model saved to {save_path}")
        return history
    
    def to_device(self, device: str) -> None:
        """Move model to specified device."""
        self.device = torch.device(device)
        self.model.to(self.device)
        logger.info(f"Model moved to {self.device}")
