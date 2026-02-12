"""
Main application launcher for Neural Network Inverse Kinematics with Gazebo simulation.
This is the entry point for the entire system.
"""

import os
import sys
import subprocess
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from src.utils_logging import get_logger
from src.utils_config import get_config
from src.nn_ik_solver import EnhancedNNIKSolver
from src.gazebo_simulator import GazeboSimulator

logger = get_logger(__name__)


class RoboticArmIKController:
    """Main controller for robotic arm with NN IK solver and Gazebo simulation."""
    
    def __init__(self, config_file: str = "config/config.yaml"):
        """
        Initialize the controller.
        
        Args:
            config_file: Path to configuration file
        """
        logger.info("Initializing NN IK Robot Controller...")
        
        # Load configuration
        self.config = get_config()
        
        # Get configuration values
        robot_cfg = self.config.get_robot_config()
        nn_cfg = self.config.get_nn_config()
        ik_cfg = self.config.get_ik_solver_config()
        
        self.robot_name = robot_cfg.get('name', 'robotic_arm')
        self.num_joints = robot_cfg.get('num_joints', 6)
        self.urdf_file = robot_cfg.get('urdf_file', 'assets/urdf/my_robot.urdf')
        
        # Initialize NN IK Solver
        logger.info("Loading Neural Network IK model...")
        self.ik_solver = EnhancedNNIKSolver(
            model_path=nn_cfg.get('model_path', 'ik_nn_model.pth'),
            input_size=nn_cfg.get('input_size', 7),
            output_size=nn_cfg.get('output_size', 6),
            hidden_layers=nn_cfg.get('architecture', {}).get('hidden_layers', [128, 64, 32])
        )
        
        # Initialize Gazebo simulator
        logger.info("Initializing Gazebo simulator...")
        self.gazebo = GazeboSimulator(
            robot_name=self.robot_name,
            num_joints=self.num_joints
        )
        
        # Configuration
        self.max_position_error = ik_cfg.get('max_position_error', 0.01)
        self.output_format = self.config.get('io.output_format', 'degrees')
        
        logger.info("Controller initialized successfully!")
    
    def solve_ik(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        return_grasp: bool = False
    ) -> dict:
        """
        Solve inverse kinematics for target pose.
        
        Args:
            position: Target position [x, y, z] in meters
            orientation: Target orientation [qx, qy, qz, qw] as quaternion
            return_grasp: Whether to add grasp information
            
        Returns:
            Dictionary with solution information
        """
        logger.info(f"Solving IK for pose: pos={position}, orn={orientation}")
        
        try:
            # Solve using NN
            joint_angles_rad = self.ik_solver.solve(position, orientation)
            
            # Convert to output format
            if self.output_format.lower() == 'degrees':
                joint_angles = np.degrees(joint_angles_rad)
            else:
                joint_angles = joint_angles_rad
            
            result = {
                'success': True,
                'joint_angles': joint_angles,
                'joint_angles_rad': joint_angles_rad,
                'position': position,
                'orientation': orientation,
                'format': self.output_format
            }
            
            logger.info(f"IK Solution: {joint_angles}")
            return result
            
        except Exception as e:
            logger.error(f"Error solving IK: {e}")
            return {
                'success': False,
                'error': str(e),
                'position': position,
                'orientation': orientation
            }
    
    def execute_movement(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        execute_in_gazebo: bool = True,
        wait_for_completion: bool = True,
        timeout: float = 10.0
    ) -> dict:
        """
        Execute complete movement: solve IK and move in Gazebo.
        
        Args:
            position: Target position [x, y, z]
            orientation: Target orientation [qx, qy, qz, qw]
            execute_in_gazebo: Whether to execute in simulation
            wait_for_completion: Whether to wait for movement completion
            timeout: Timeout for movement (seconds)
            
        Returns:
            Dictionary with execution result
        """
        logger.info("="*60)
        logger.info("Executing Movement Request")
        logger.info("="*60)
        
        # Step 1: Solve IK
        ik_result = self.solve_ik(position, orientation)
        
        if not ik_result['success']:
            return {
                'success': False,
                'error': ik_result.get('error', 'IK solving failed'),
                'ik_result': ik_result
            }
        
        joint_angles_rad = ik_result['joint_angles_rad']
        joint_angles_deg = ik_result['joint_angles']
        
        # Step 2: Execute in Gazebo
        result = {
            'success': True,
            'joint_angles_degrees': joint_angles_deg,
            'joint_angles_radians': joint_angles_rad,
            'position': position,
            'orientation': orientation,
            'gazebo_executed': False
        }
        
        if execute_in_gazebo:
            try:
                logger.info(f"Sending command to Gazebo: {joint_angles_deg}")
                self.gazebo.set_joint_positions(joint_angles_rad, delay=0.05)
                result['gazebo_executed'] = True
                
                # Get updated end-effector pose
                if wait_for_completion:
                    time.sleep(1.0)  # Wait for movement
                    ee_pose = self.gazebo.get_ee_pose()
                    if ee_pose:
                        result['actual_position'] = ee_pose[0]
                        result['actual_orientation'] = ee_pose[1]
                        pos_error = np.linalg.norm(ee_pose[0] - position)
                        result['position_error'] = pos_error
                        logger.info(f"Position error: {pos_error:.6f} m")
                
            except Exception as e:
                logger.error(f"Error executing in Gazebo: {e}")
                result['gazebo_error'] = str(e)
        
        return result
    
    def interactive_mode(self) -> None:
        """Run interactive mode for user input."""
        logger.info("Entering interactive mode. Type 'help' for commands.")
        
        while True:
            try:
                print("\n" + "="*70)
                print("NN IK Robot Controller - Interactive Mode")
                print("="*70)
                print("Commands:")
                print("  position [x] [y] [z]  - Set target position (meters)")
                print("  rotation [roll] [pitch] [yaw]  - Set target rotation (degrees)")
                print("  execute - Execute movement in Gazebo")
                print("  auto [x] [y] [z] [roll] [pitch] [yaw] - Automatic full command")
                print("  status - Show current robot status")
                print("  help - Show this help")
                print("  quit - Exit")
                print("="*70)
                
                command = input("\nEnter command > ").strip().lower()
                
                if command == "quit":
                    logger.info("Exiting interactive mode")
                    break
                
                elif command == "help":
                    continue
                
                elif command.startswith("position"):
                    parts = command.split()
                    if len(parts) == 4:
                        try:
                            pos = np.array([float(p) for p in parts[1:]], dtype=np.float32)
                            print(f"Position set to: {pos}")
                            self.current_position = pos
                        except ValueError:
                            print("Invalid position values")
                
                elif command.startswith("rotation"):
                    parts = command.split()
                    if len(parts) == 4:
                        try:
                            roll, pitch, yaw = [float(p) for p in parts[1:]]
                            orn = self._euler_to_quaternion(roll, pitch, yaw)
                            print(f"Orientation set to: {orn}")
                            self.current_orientation = orn
                        except ValueError:
                            print("Invalid rotation values")
                
                elif command == "execute":
                    if hasattr(self, 'current_position') and hasattr(self, 'current_orientation'):
                        result = self.execute_movement(
                            self.current_position,
                            self.current_orientation
                        )
                        print(f"\nExecution Result:")
                        print(f"  Success: {result['success']}")
                        print(f"  Joint Angles (deg): {result.get('joint_angles_degrees', 'N/A')}")
                        if 'position_error' in result:
                            print(f"  Position Error: {result['position_error']:.6f} m")
                    else:
                        print("Please set position and rotation first")
                
                elif command.startswith("auto"):
                    parts = command.split()
                    if len(parts) == 7:
                        try:
                            pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])], 
                                          dtype=np.float32)
                            roll, pitch, yaw = float(parts[4]), float(parts[5]), float(parts[6])
                            orn = self._euler_to_quaternion(roll, pitch, yaw)
                            
                            result = self.execute_movement(pos, orn)
                            print(f"\nExecution Result:")
                            print(f"  Success: {result['success']}")
                            print(f"  Joint Angles (deg): {result.get('joint_angles_degrees', 'N/A')}")
                            if 'position_error' in result:
                                print(f"  Position Error: {result['position_error']:.6f} m")
                        except (ValueError, IndexError):
                            print("Invalid auto command format")
                
                elif command == "status":
                    ee_pose = self.gazebo.get_ee_pose()
                    if ee_pose:
                        print(f"Current EE Position: {ee_pose[0]}")
                        print(f"Current EE Orientation: {ee_pose[1]}")
                    else:
                        print("Unable to get end-effector status")
            
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
    
    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles (degrees) to quaternion."""
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        cy = np.cos(yaw_rad * 0.5)
        sy = np.sin(yaw_rad * 0.5)
        cp = np.cos(pitch_rad * 0.5)
        sp = np.sin(pitch_rad * 0.5)
        cr = np.cos(roll_rad * 0.5)
        sr = np.sin(roll_rad * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw], dtype=np.float32)
    
    def launch_gazebo(self) -> None:
        """Launch Gazebo simulation."""
        logger.info("Launching Gazebo simulation...")
        
        try:
            # Check if ROS is available
            result = subprocess.run(
                ["roslaunch", "robot_nn_ik", "robot_arm.launch"],
                cwd=str(Path(__file__).parent.parent),
                capture_output=False
            )
            
            if result.returncode != 0:
                logger.error("Failed to launch Gazebo")
        except FileNotFoundError:
            logger.error("ROS/Gazebo not found. Install ROS to use Gazebo simulation.")
        except Exception as e:
            logger.error(f"Error launching Gazebo: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Neural Network Inverse Kinematics Robot Controller with Gazebo"
    )
    parser.add_argument(
        '--launch-gazebo',
        action='store_true',
        help='Launch Gazebo simulation'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--position',
        nargs=3,
        type=float,
        help='Target position [x y z] in meters'
    )
    parser.add_argument(
        '--rotation',
        nargs=3,
        type=float,
        help='Target rotation [roll pitch yaw] in degrees'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize controller
        controller = RoboticArmIKController(args.config)
        
        # Launch Gazebo if requested
        if args.launch_gazebo:
            controller.launch_gazebo()
            time.sleep(5)  # Wait for Gazebo to load
        
        # Execute command if provided
        if args.position and args.rotation:
            roll, pitch, yaw = args.rotation
            position = np.array(args.position, dtype=np.float32)
            orientation = controller._euler_to_quaternion(roll, pitch, yaw)
            
            logger.info(f"Target Position: {position}")
            logger.info(f"Target Rotation (deg): {args.rotation}")
            
            result = controller.execute_movement(position, orientation)
            
            print("\n" + "="*70)
            print("EXECUTION RESULT")
            print("="*70)
            print(f"Success: {result['success']}")
            print(f"Joint Angles (degrees): {result.get('joint_angles_degrees', 'N/A')}")
            if 'position_error' in result:
                print(f"Position Error: {result['position_error']:.6f} m")
            print("="*70)
        
        # Interactive mode
        elif args.interactive or (not args.launch_gazebo and not args.position):
            controller.interactive_mode()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
