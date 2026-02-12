# Neural Network Inverse Kinematics Robot Control with Gazebo Simulation

A professional-grade robotic arm inverse kinematics solver using neural networks, with integrated Gazebo simulation support for real-time visualization and control.

## Features

✅ **Neural Network IK Solver**
- Advanced neural network architecture with batch normalization and dropout
- Efficient forward pass for real-time predictions
- Robust model loading and error handling
- Support for GPU acceleration

✅ **Gazebo Integration**
- Real-time robot visualization in Gazebo
- Direct control of joint positions
- End-effector pose feedback
- Configurable simulation parameters

✅ **Flexible Input/Output**
- Position input: [x, y, z] in meters
- Orientation input: quaternion [qx, qy, qz, qw] or Euler angles
- Output: joint angles in degrees or radians
- Configurable through YAML configuration

✅ **Production-Ready Code**
- Comprehensive logging system
- Configuration management
- Error handling and validation
- Modular architecture
- Type hints and documentation

## Project Structure

```
NN_ROBOT/
├── assets/
│   ├── urdf/                  # Robot URDF files
│   │   └── my_robot.urdf
│   ├── mesh/                  # 3D mesh files
│   ├── models/                # Additional model files
│   └── stl_models/            # STL files for custom robot (ADD YOUR STL HERE)
├── gazebo_simulation/
│   ├── launch/                # ROS launch files
│   │   └── robot_arm.launch
│   ├── worlds/                # Gazebo world files
│   │   └── robot_arm.world
│   └── models/                # Gazebo model definitions
├── config/
│   └── config.yaml            # Configuration file
├── logs/                      # Log files
├── src/
│   ├── __init__.py
│   ├── main_controller.py     # Main entry point
│   ├── nn_ik_solver.py        # Enhanced NN IK solver
│   ├── gazebo_simulator.py    # Gazebo integration
│   ├── utils_logging.py       # Logging utilities
│   ├── utils_config.py        # Configuration management
│   ├── train_ik.py            # Training script
│   ├── ik_dataset.py          # Dataset generation
│   ├── sim_pybullet.py        # PyBullet simulator
│   └── demo_sim_with_nn.py    # Demo script
├── requirements.txt           # Python dependencies
├── ik_nn_model.pth           # Trained neural network model
└── README.md                  # This file
```

## Installation

### Prerequisites

- **Python 3.7+**
- **ROS** (for Gazebo integration) - See ROS installation below
- **CUDA** (optional, for GPU acceleration)

### Python Dependencies

```bash
pip install -r requirements.txt
```

### ROS & Gazebo Setup (For Simulation)

#### Ubuntu/Linux:
```bash
# Install ROS (follow official ROS documentation)
# Then install Gazebo and related packages:
sudo apt update
sudo apt install ros-DISTRO-gazebo-ros ros-DISTRO-gazebo-ros-control
sudo apt install ros-DISTRO-controller-manager ros-DISTRO-robot-state-publisher
sudo apt install ros-DISTRO-joint-state-controller ros-DISTRO-position-controllers
```

Replace `DISTRO` with your ROS distribution (e.g., `noetic`, `melodic`).

#### Windows/macOS:
ROS runs primarily on Linux. For Windows/macOS development:
- Use WSL 2 on Windows
- Or use Docker with ROS
- Or use PyBullet simulation (see below)

## Quick Start

### 1. Basic Usage (Without Gazebo)

```bash
python -m src.main_controller --position 0.5 0.2 0.4 --rotation 0 0 0
```

This will:
- Load the trained NN model
- Solve IK for position [0.5, 0.2, 0.4] and rotation [0°, 0°, 0°]
- Output joint angles in degrees

### 2. Interactive Mode

```bash
python -m src.main_controller --interactive
```

Commands:
- `position [x] [y] [z]` - Set target position
- `rotation [roll] [pitch] [yaw]` - Set target rotation (degrees)
- `execute` - Solve IK and execute
- `auto [x] [y] [z] [roll] [pitch] [yaw]` - Full command in one line
- `status` - Show current robot state
- `quit` - Exit

### 3. With Gazebo Simulation

```bash
python -m src.main_controller --launch-gazebo --position 0.5 0.2 0.4 --rotation 0 0 0
```

This will:
1. Launch Gazebo UI
2. Load your robot
3. Solve IK for the given pose
4. Execute movement in gazebo
5. Show position error feedback

## Configuration

Edit `config/config.yaml` to customize:

```yaml
robot:
  num_joints: 6
  urdf_file: "assets/urdf/my_robot.urdf"
  workspace:
    x: [0.0, 1.0]
    y: [-0.5, 0.5]
    z: [0.0, 1.0]

nn_model:
  model_path: "ik_nn_model.pth"
  architecture:
    hidden_layers: [128, 64, 32]
  training:
    epochs: 50
    batch_size: 64

ik_solver:
  max_position_error: 0.01
  validate_solution: true

io:
  output_format: "degrees"  # or "radians"
```

## Training a New Model

### Generate Training Data

```python
from src.ik_dataset import generate_dataset
from src.nn_ik_solver import EnhancedNNIKSolver

# Generate dataset
poses, joints = generate_dataset(n_samples=5000)

# Train model
solver = EnhancedNNIKSolver()
history = solver.train(
    poses, joints,
    epochs=50,
    batch_size=64,
    save_path="ik_nn_model.pth"
)
```

Or use the training script:

```bash
python -m src.train_ik
```

## Adding Your Custom Robot STL

1. **Prepare STL Files**: Convert your robot CAD model to STL format
2. **Place Files**: Add STL files to `assets/stl_models/`
3. **Create URDF**: Generate URDF file in `assets/urdf/`
4. **Update Configuration**: Modify `config.yaml` with your robot dimensions

### Example URDF for Custom Robot:

```xml
<?xml version="1.0"?>
<robot name="my_custom_arm">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="package://robot_nn_ik/assets/stl_models/base.stl"/>
      </geometry>
    </visual>
  </link>
  <!-- Add more links and joints -->
</robot>
```

## Usage Examples

### Example 1: Solve IK in Python

```python
from src.nn_ik_solver import EnhancedNNIKSolver
import numpy as np

solver = EnhancedNNIKSolver(model_path="ik_nn_model.pth")

# Define target pose
position = np.array([0.5, 0.2, 0.4])  # x, y, z in meters
orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion [qx, qy, qz, qw]

# Solve
joint_angles = solver.solve(position, orientation)
print(f"Joint angles (radians): {joint_angles}")
print(f"Joint angles (degrees): {np.degrees(joint_angles)}")
```

### Example 2: Batch Processing

```python
# Solve multiple poses
positions = np.random.rand(10, 3) * 0.5 + 0.2
orientations = np.tile([0, 0, 0, 1], (10, 1))

solutions = solver.solve_batch(positions, orientations)
print(f"Solutions shape: {solutions.shape}")  # (10, 6)
```

### Example 3: With Gazebo Control

```python
from src.main_controller import RoboticArmIKController

controller = RoboticArmIKController()
controller.launch_gazebo()  # Starts Gazebo UI

# Execute movement
result = controller.execute_movement(
    position=np.array([0.5, 0.2, 0.4]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0])
)

print(f"Success: {result['success']}")
print(f"Joint angles: {result['joint_angles_degrees']}")
print(f"Position error: {result.get('position_error', 'N/A')} m")
```

## API Reference

### EnhancedNNIKSolver

```python
class EnhancedNNIKSolver:
    def solve(position, orientation, return_confidence=False)
        """Predict joint angles from pose"""
    
    def solve_batch(positions, orientations)
        """Solve multiple poses"""
    
    def train(poses, joints, epochs=50, batch_size=64, ...)
        """Train on new dataset"""
```

### RoboticArmIKController

```python
class RoboticArmIKController:
    def solve_ik(position, orientation)
        """Solve IK only"""
    
    def execute_movement(position, orientation, execute_in_gazebo=True)
        """Solve IK and execute in Gazebo"""
    
    def interactive_mode()
        """Run interactive command loop"""
    
    def launch_gazebo()
        """Launch Gazebo simulation"""
```

## Logging

Logs are automatically saved to `logs/nn_ik_YYYYMMDD_HHMMSS.log`

Configure logging in `config/config.yaml`:
```yaml
logging:
  level: "INFO"
  log_dir: "logs"
  console_output: true
```

Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## Troubleshooting

### Issue: Gazebo not launching

**Solution**: Ensure ROS is properly installed:
```bash
source /opt/ros/DISTRO/setup.bash
echo $ROS_PACKAGE_PATH
```

### Issue: Model not found

**Solution**: Check URDF path in config:
```bash
ls -la assets/urdf/my_robot.urdf
```

### Issue: Large position error

**Possible causes**:
- URDF doesn't match actual robot dimensions
- Robot outside workspace
- NN model needs retraining

**Solution**: Retrain with corrected robot geometry

### Issue: GPU not detected

**Solution**: Ensure PyTorch is installed with CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Performance Optimization

1. **Check GPU Usage**:
   ```bash
   nvidia-smi
   ```

2. **Batch Processing**: Use `solve_batch()` for multiple poses

3. **Reduce Model Complexity**: Modify hidden layers in config.yaml

4. **Optimize URDF**: Simplify mesh files for faster simulation

## Publishing Your Project

When ready to share:

1. Clean up temporary files
2. Update documentation
3. Tag a release
4. Generate documentation:
   ```bash
   pip install sphinx
   sphinx-build -b html docs/ docs/_build/
   ```

## Paper/Citation

If you use this project in research, please cite:

```
@software{nn_ik_robot_2024,
  title={Neural Network Inverse Kinematics with Gazebo Simulation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourrepo}
}
```

## License

See [LICENSE](LICENSE) file

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues and questions:
- Open a GitHub issue
- Check documentation
- Review examples

## Future Enhancements

- [ ] Real-time trajectory planning
- [ ] Multi-arm coordination
- [ ] Obstacle avoidance
- [ ] Collision detection
- [ ] Web-based UI
- [ ] ROS 2 support
- [ ] Docker containers
- [ ] Reinforcement Learning IK solver

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready ✅