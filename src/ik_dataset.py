# ...existing code...
import numpy as np
from src.sim_pybullet import RobotSim


def generate_dataset(n: int = 3000):
    """
    Generate dataset of (pose -> joints) samples using the local kinematic RobotSim.
    Returns:
        X: (n,7) array of poses [x,y,z,qx,qy,qz,qw]
        Y: (n,6) array of joint angles (radians)
    """
    sim = RobotSim(gui=False)

    X, Y = [], []
    for i in range(n):
        q = sim.sample_random_configuration()  # (6,)
        pose = sim.compute_fk(q)  # (7,)

        X.append(pose.astype(np.float32))
        Y.append(q.astype(np.float32))

        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"[Dataset] Collected {i+1}/{n} samples")

    sim.disconnect()

    return np.vstack(X), np.vstack(Y)
