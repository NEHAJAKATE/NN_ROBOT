# Quick Reference Guide - NN IK Robot

## 🚀 Quick Commands

### Test Basic IK (No Gazebo Needed)
```bash
python -m src.main_controller --position 0.5 0.2 0.4 --rotation 0 0 0
```

### Interactive Mode
```bash
python -m src.main_controller --interactive
```

### Launch Gazebo + Execute
```bash
python -m src.main_controller --launch-gazebo --position 0.5 0.2 0.4 --rotation 0 0 0
```

### Run Examples
```bash
python examples.py
```

---

## 📍 File Locations

| What | Where |
|------|-------|
| **STL Files** | `assets/stl_models/` ← Add your STL here |
| **Robot Model** | `assets/urdf/my_robot.urdf` |
| **Configuration** | `config/config.yaml` |
| **Source Code** | `src/` |
| **Logs** | `logs/nn_ik_*.log` |
| **Examples** | `examples.py` |
| **Documentation** | `README.md`, `SETUP.md` |

---

## 🔧 Configuration (config/config.yaml)

### Output Format
```yaml
io:
  output_format: "degrees"  # or "radians"
```

### Workspace Limits
```yaml
robot:
  workspace:
    x: [0.0, 1.0]  # meters
    y: [-0.5, 0.5]
    z: [0.0, 1.0]
```

### Network Architecture
```yaml
nn_model:
  architecture:
    hidden_layers: [128, 64, 32]
  training:
    epochs: 50
    batch_size: 64
    learning_rate: 0.001
```

---

## 💻 Python API Examples

### Load Model & Solve
```python
from src.nn_ik_solver import EnhancedNNIKSolver
import numpy as np

solver = EnhancedNNIKSolver()
joint_angles = solver.solve(
    position=np.array([0.5, 0.2, 0.4]),
    orientation=np.array([0, 0, 0, 1])
)
print(f"Solution: {np.degrees(joint_angles)}")
```

### Batch Processing
```python
positions = np.random.rand(100, 3)
orientations = np.tile([0, 0, 0, 1], (100, 1))
solutions = solver.solve_batch(positions, orientations)  # 10x faster!
```

### Full Control with Gazebo
```python
from src.main_controller import RoboticArmIKController

controller = RoboticArmIKController()
result = controller.execute_movement(
    position=np.array([0.5, 0.2, 0.4]),
    orientation=np.array([0, 0, 0, 1]),
    execute_in_gazebo=True
)
print(f"Joint angles: {result['joint_angles_degrees']}")
print(f"Error: {result.get('position_error')} m")
```

### Train New Model
```python
from src.ik_dataset import generate_dataset
from src.nn_ik_solver import EnhancedNNIKSolver

poses, joints = generate_dataset(n_samples=5000)
solver = EnhancedNNIKSolver()
history = solver.train(poses, joints, epochs=50)
```

---

## 📊 Input/Output Formats

### Position (always meters)
```
[x, y, z]
Example: [0.5, 0.2, 0.4]
```

### Orientation (two options)

**Quaternion** (default)
```
[qx, qy, qz, qw]
Example: [0.0, 0.0, 0.0, 1.0]
```

**Euler Angles** (if converting)
```
[roll, pitch, yaw] in degrees
Example: [0.0, 45.0, 90.0]
```

### Output (configurable)

**Degrees** (default, more readable)
```
[45.2, -30.5, 60.1, 15.3, -20.4, 5.2]
```

**Radians** (if configured)
```
[0.789, -0.532, 1.047, 0.267, -0.356, 0.087]
```

---

## 🐛 Troubleshooting

### Model not found
```bash
# Check path
ls -la ik_nn_model.pth

# Update in config.yaml:
nn_model:
  model_path: "ik_nn_model.pth"
```

### Gazebo not launching
```bash
# Check ROS
echo $ROS_DISTRO
source /opt/ros/DISTRO/setup.bash

# Test Gazebo
gazebo --version
```

### Large position error
- URDF dimensions don't match actual robot
- Robot is outside workspace
- Need to retrain model
- Check joint limits in URDF

### GPU not detected
```bash
# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📈 Performance Tips

1. **Use batch processing** for multiple poses
2. **GPU acceleration** is automatic if available
3. **Reduce model size** for faster inference
4. **Simplify geometry** in URDF for faster simulation
5. **Cache results** if solving same pose repeatedly

---

## 🎯 Usage Examples

### Example 1: Pick and Place
```python
controller = RoboticArmIKController()
controller.launch_gazebo()

# Ready position
controller.execute_movement(np.array([0.5, 0.0, 0.5]), np.array([0, 0, 0, 1]))

# Pick (lower)
controller.execute_movement(np.array([0.4, 0.1, 0.3]), np.array([0, 0, 0, 1]))

# Place (different location)
controller.execute_movement(np.array([0.3, -0.2, 0.3]), np.array([0, 0, 0, 1]))
```

### Example 2: Orientation Testing
```python
# Test different orientations
for roll in [0, 45, 90]:
    orn = controller._euler_to_quaternion(roll, 0, 0)
    result = controller.solve_ik(np.array([0.5, 0.2, 0.4]), orn)
    if result['success']:
        print(f"Roll {roll}°: {result['joint_angles']}")
```

### Example 3: Workspace Exploration
```python
# Sample workspace
solutions = {}
for x in np.linspace(0.2, 0.8, 5):
    for y in np.linspace(-0.3, 0.3, 5):
        for z in np.linspace(0.2, 0.8, 5):
            angle = solver.solve(np.array([x, y, z]), np.array([0, 0, 0, 1]))
            solutions[(x,y,z)] = angle
```

---

## 📦 Adding You Custom Robot - Step by Step

1. **Get STL files** from CAD/supplier
2. **Place in** `assets/stl_models/`
3. **Measure dimensions** (link lengths, masses)
4. **Create URDF** using `assets/urdf/robot_template.urdf` as template
5. **Update config.yaml** with workspace bounds
6. **Test**: `python -m src.main_controller --launch-gazebo`
7. **Train** (if needed): `python examples.py` → Choose example 4

---

## 🔗 Important Links

- **Full Documentation**: `README.md`
- **Installation Guide**: `SETUP.md`
- **Examples File**: `examples.py`
- **Configuration**: `config/config.yaml`
- **URDF Template**: `assets/urdf/robot_template.urdf`
- **Project Summary**: `PROJECT_SUMMARY.md`

---

## 🆘 When Things Go Wrong

### Step 1: Check Logs
```bash
tail -f logs/nn_ik_*.log
```

### Step 2: Verify Installation
```bash
python -c "from src.nn_ik_solver import EnhancedNNIKSolver; print('✓ OK')"
```

### Step 3: Test Basic Functionality
```bash
python -m src.main_controller --position 0.5 0.2 0.4 --rotation 0 0 0
```

### Step 4: Run Examples
```bash
python examples.py
```

### Step 5: Check Documentation
- See `README.md` for detailed info
- See `SETUP.md` for installation issues
- See `PROJECT_SUMMARY.md` for architecture

---

## ✅ Success Checklist

- [ ] Python dependencies installed
- [ ] Basic IK solving works
- [ ] Custom URDF created (or using template)
- [ ] STL files added to `assets/stl_models/`
- [ ] Config updated for your robot
- [ ] Gazebo launches successfully (Linux)
- [ ] Can execute movements
- [ ] Getting reasonable position errors

---

## 🌟 Pro Tips

1. **Always check logs** - they contain important debug info
2. **Use configuration** - don't hardcode values
3. **Batch your requests** - much faster for multiple poses
4. **Check workspace** - robot can't reach outside it
5. **Validate URDF** - use `urdf_check` tool if available
6. **Test incrementally** - verify each part works
7. **GPU matters** - huge speed improvement if available
8. **Log levels** - set to DEBUG for detailed debugging

---

## 📞 Support

- Check `README.md` troubleshooting section
- See `SETUP.md` installation guide
- Review `examples.py` for usage patterns
- Enable DEBUG logging in `config/config.yaml`
- Check logs in `logs/` directory

---

**Version**: 1.0.0 - Production Ready ✅
