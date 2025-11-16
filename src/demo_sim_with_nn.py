import numpy as np
from src.nn_ik_model import NNIKSolver


def euler_to_quat_degrees(roll_deg, pitch_deg, yaw_deg):
    """Convert euler angles in degrees to quaternion."""
    # Convert degrees to radians for calculation only
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    
    # Euler to quaternion (ZYX convention)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return qx, qy, qz, qw


def get_user_input():
    """Get target pose from user."""
    print("\nEnter target position as three floats 'x y z' (meters). Press Enter for default 0.3 0.2 0.4")
    pos_input = input("x y z > ").strip()
    
    if pos_input:
        try:
            parts = pos_input.split()
            if len(parts) != 3:
                print("Invalid position; using default 0.3 0.2 0.4")
                x, y, z = 0.3, 0.2, 0.4
            else:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            print("Invalid position; using default 0.3 0.2 0.4")
            x, y, z = 0.3, 0.2, 0.4
    else:
        x, y, z = 0.3, 0.2, 0.4
    
    print("Choose orientation type: 'quat' for quaternion [qx qy qz qw], or 'euler' for roll pitch yaw (degrees). Press Enter for default quaternion [0 0 0 1]")
    orn_type = input("orientation type (quat/euler) > ").strip().lower() or "quat"
    
    if orn_type == "euler":
        euler_input = input("roll pitch yaw (degrees) > ").strip()
        try:
            parts = euler_input.split()
            if len(parts) != 3:
                print("Invalid euler; using default 0 0 0")
                qx, qy, qz, qw = 0, 0, 0, 1
            else:
                roll_deg = float(parts[0])
                pitch_deg = float(parts[1])
                yaw_deg = float(parts[2])
                qx, qy, qz, qw = euler_to_quat_degrees(roll_deg, pitch_deg, yaw_deg)
        except ValueError:
            print("Invalid euler; using default 0 0 0")
            qx, qy, qz, qw = 0, 0, 0, 1
    else:
        quat_input = input("quaternion [qx qy qz qw] > ").strip()
        try:
            if quat_input:
                parts = quat_input.split()
                if len(parts) != 4:
                    print("Invalid quaternion; using default [0 0 0 1]")
                    qx, qy, qz, qw = 0, 0, 0, 1
                else:
                    qx, qy, qz, qw = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            else:
                qx, qy, qz, qw = 0, 0, 0, 1
        except ValueError:
            print("Invalid quaternion; using default [0 0 0 1]")
            qx, qy, qz, qw = 0, 0, 0, 1
    
    return x, y, z, qx, qy, qz, qw


def main():
    """Main demo loop."""
    # Initialize solver
    solver = NNIKSolver()
    
    while True:
        x, y, z, qx, qy, qz, qw = get_user_input()
        
        print(f"\n[Target] Position: ({x:.3f}, {y:.3f}, {z:.3f}) meters")
        print(f"[Target] Orientation (quat): ({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")
        
        # Predict joint angles (in radians from model)
        q_rad = solver.solve(x, y, z, qx, qy, qz, qw)
        
        # Convert to degrees for display only
        q_deg = np.degrees(q_rad)
        
        print(f"\n[Solution] Joint angles (degrees):")
        for i, angle in enumerate(q_deg):
            print(f"  Joint {i}: {angle:.2f}°")
        
        # Ask for next iteration
        cont = input("\nSolve another pose? (y/n) [y]: ").strip().lower() or "y"
        if cont != "y":
            print("[Demo] Finished.")
            break


if __name__ == "__main__":
    main()