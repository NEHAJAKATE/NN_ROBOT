import os
import time
import numpy as np
import pybullet as p
import pybullet_data


class PyBulletRobot:
    def __init__(self, urdf_path: str, gui: bool = False):
        # Resolve absolute URDF path
        self.urdf_path = os.path.abspath(urdf_path)
        self.gui = gui
        self.client = None
        self.robot_id = None
        self.num_joints = 0
        self.ee_link_index = None

    def connect(self):
        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        return self.client

    def load_robot(self):
        if not os.path.isfile(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        # Use fixed base for stability
        self.robot_id = p.loadURDF(self.urdf_path, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)
        # Detect end-effector link index
        ee_candidates = ["link5", "ee", "end_effector", "tool0"]
        found = None
        for j in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, j)
            link_name = info[12]
            if isinstance(link_name, bytes):
                link_name = link_name.decode("utf-8", errors="ignore")
            if any(tag in (link_name or "") for tag in ee_candidates):
                found = j
                break
        self.ee_link_index = found if found is not None else (self.num_joints - 1 if self.num_joints > 0 else 0)
        # Initialize motors to zero
        for j in range(self.num_joints):
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=500)
        return self.robot_id

    def get_joint_count(self):
        return self.num_joints

    def set_joints(self, q):
        # Clamp to joint limits if available, else [-pi, pi]
        for j in range(min(len(q), self.num_joints)):
            info = p.getJointInfo(self.robot_id, j)
            lower, upper = info[8], info[9]
            # PyBullet uses +/-1e30 when no limit
            lo = lower if lower > -1e10 else -np.pi
            hi = upper if upper < 1e10 else np.pi
            target = float(np.clip(q[j], lo, hi))
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=target, force=500)

    def get_joint_positions(self):
        qs = []
        for j in range(self.num_joints):
            state = p.getJointState(self.robot_id, j)
            qs.append(state[0])
        return np.array(qs, dtype=np.float32)

    def get_ee_pose(self):
        if self.ee_link_index is None:
            raise RuntimeError("Robot not loaded or EE link not set")
        ls = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        pos = ls[4]
        orn = ls[5]
        return np.array(list(pos) + list(orn), dtype=np.float32)

    def step(self, sleep: float = 0.01):
        p.stepSimulation()
        if sleep:
            time.sleep(sleep)

    def disconnect(self):
        try:
            p.disconnect()
        except Exception:
            pass