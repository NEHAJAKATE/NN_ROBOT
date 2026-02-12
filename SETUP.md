# Setup and Installation Guide

## Quick Setup (5 minutes)

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Basic IK Solver
```bash
python -m src.main_controller --position 0.5 0.2 0.4 --rotation 0 0 0
```

Expected output:
```
Joint Angles (degrees): [angle1 angle2 angle3 angle4 angle5 angle6]
```

---

## Complete Setup with Gazebo (15-20 minutes on Linux)

### Step 1: Install ROS (Ubuntu 18.04/20.04)

```bash
# Setup ROS repository
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Add ROS key
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Update and install ROS
sudo apt update
sudo apt install ros-DISTRO-desktop-full

# Replace DISTRO with your version (noetic, melodic, etc.)
# For Ubuntu 20.04: ros-noetic-desktop-full
```

### Step 2: Initialize ROS

```bash
source /opt/ros/DISTRO/setup.bash
sudo rosdep init
rosdep update
```

### Step 3: Create ROS Workspace

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### Step 4: Install Gazebo & Controllers

```bash
sudo apt update
sudo apt install ros-DISTRO-gazebo-ros-control
sudo apt install ros-DISTRO-controller-manager
sudo apt install ros-DISTRO-robot-state-publisher
sudo apt install ros-DISTRO-joint-state-controller
sudo apt install ros-DISTRO-effort-controllers
sudo apt install ros-DISTRO-position-controllers
```

### Step 5: Create ROS Package

```bash
cd ~/catkin_ws/src
catkin_create_pkg robot_nn_ik rospy gazebo_ros controller_manager std_msgs
cd ~/catkin_ws
catkin_make
```

### Step 6: Link Project

```bash
cd ~/catkin_ws/src/robot_nn_ik
# Copy your project files here, or create symlinks
ln -s /path/to/NN_ROBOT/* .
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### Step 7: Test Gazebo

```bash
python -m src.main_controller --launch-gazebo --position 0.5 0.2 0.4 --rotation 0 0 0
```

---

## Setup for Windows/macOS (Using WSL or Docker)

### Option A: WSL 2 (Windows)

```bash
# Install WSL 2 Ubuntu
# Then follow Linux setup above
```

### Option B: Docker

```bash
# Install Docker Desktop

# Create Dockerfile
cat > Dockerfile << EOF
FROM osrf/ros:DISTRO-desktop-full
RUN apt-get update && apt-get install -y python3-pip
WORKDIR /workspace
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "-m", "src.main_controller", "--interactive"]
EOF

# Build and run
docker build -t nn-ik-robot .
docker run -it --rm nn-ik-robot
```

---

## Adding Your Custom Robot

### Step 1: Prepare STL Files

Your supplier will provide STL files. Place them in:
```
assets/stl_models/
├── base.stl
├── link1.stl
├── link2.stl
├── link3.stl
├── link4.stl
├── link5.stl
└── link6.stl
```

### Step 2: Measure Robot Parameters

Measure or get from CAD:
- **Link lengths**: Distance between joint centers
- **Link masses**: For simulation
- **Joint limits**: Min/max angles
- **Joint types**: Revolute or prismatic

Example measurements for 6-DOF arm:
```
Link 0 (base): height = 0.1 m
Link 1: length = 0.3 m, mass = 2 kg
Link 2: length = 0.35 m, mass = 1.5 kg
Link 3: length = 0.3 m, mass = 1 kg
Link 4-6: length = 0.1 m each, mass = 0.5 kg each

Joint limits (all revolute): [-180°, +180°]
```

### Step 3: Create URDF File

Use the template to create `assets/urdf/my_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="package://robot_nn_ik/assets/stl_models/base.stl" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="package://robot_nn_ik/assets/stl_models/base.stl" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>

  <!-- Link 1 -->
  <link name="link1">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="package://robot_nn_ik/assets/stl_models/link1.stl" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
  </link>

  <!-- Joint 1 (connects base_link to link1) -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="10" velocity="1"/>
    <dynamics friction="0.1" damping="0.1"/>
  </joint>

  <!-- Repeat for link2, link3, ... link6 -->
  
  <!-- End-effector link (tool) -->
  <link name="link6">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="package://robot_nn_ik/assets/stl_models/link6.stl" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
  </link>

  <joint name="joint6" type="revolute">
    <parent link="link5"/>
    <child link="link6"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="5" velocity="1"/>
    <dynamics friction="0.05" damping="0.05"/>
  </joint>

</robot>
```

### Step 4: Update Configuration

Edit `config/config.yaml`:

```yaml
robot:
  name: "my_custom_robot"
  num_joints: 6
  base_link: "base_link"
  end_effector_link: "link6"
  urdf_file: "assets/urdf/my_robot.urdf"
  
  # Update based on your measurements
  joint_limits:
    lower: [-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159]
    upper: [3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159]
  
  workspace:
    x: [0.0, 1.0]  # Adjust based on your robot reach
    y: [-0.5, 0.5]
    z: [0.0, 1.0]
```

### Step 5: Test

```bash
# Test Gazebo with your robot
python -m src.main_controller --launch-gazebo --position 0.5 0.2 0.4 --rotation 0 0 0
```

---

## Training with Your Robot

If you need to train a new NN model with your robot's kinematics:

```python
from src.ik_dataset import generate_dataset
from src.nn_ik_solver import EnhancedNNIKSolver

# This generates data using PyBullet
# It will use your URDF file automatically
print("Generating 5000 training samples...")
poses, joints = generate_dataset(n_samples=5000)

print("Training neural network...")
solver = EnhancedNNIKSolver()
history = solver.train(
    poses, joints,
    epochs=50,
    batch_size=64,
    save_path="ik_nn_model_custom.pth"
)

print("Training complete! Model saved.")
```

Then update config to use the new model:
```yaml
nn_model:
  model_path: "ik_nn_model_custom.pth"
```

---

## Environmental Setup for Development

### Create Python Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### IDE Setup

**VSCode**:
1. Install Python extension
2. Select interpreter: `venv/bin/python`
3. Install PyLance for better autocomplete
4. Press Ctrl+K Ctrl+0 to open settings
5. Set Python linting and formatting

**PyCharm**:
1. File → Settings → Project → Python Interpreter
2. Select `venv/bin/python`
3. Enable code inspection

---

## Verification Checklist

After setup, verify everything works:

- [ ] Python dependencies installed: `pip list`
- [ ] NN model loads: `python -c "from src.nn_ik_solver import EnhancedNNIKSolver; EnhancedNNIKSolver()"`
- [ ] Basic IK works: `python -m src.main_controller --position 0.5 0.2 0.4 --rotation 0 0 0`
- [ ] ROS installed (if using Gazebo): `echo $ROS_DISTRO`
- [ ] Gazebo works: `gazebo --version`
- [ ] Can launch Gazebo: `python -m src.main_controller --launch-gazebo`

---

## Troubleshooting

### Python ModuleNotFoundError

```bash
# Make sure you're in the project root
pwd  # Should show NN_ROBOT directory

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### PyTorch Installation Issues

```bash
# CPU only (faster to install)
pip install torch torchvision torchaudio

# GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ROS/Gazebo Not Found

```bash
# Source ROS setup
source /opt/ros/DISTRO/setup.bash

# Add to ~/.bashrc for permanent
echo "source /opt/ros/DISTRO/setup.bash" >> ~/.bashrc
```

---

## Next Steps

1. ✅ Complete setup
2. ✅ Test basic functionality
3. ✅ Add custom robot STL files
4. ✅ Create robot URDF
5. ✅ Train/fine-tune NN model
6. ✅ Use in production
7. 📊 Monitor logs and performance
8. 🔄 Iterate and improve

---

For detailed documentation, see [README.md](README.md)
