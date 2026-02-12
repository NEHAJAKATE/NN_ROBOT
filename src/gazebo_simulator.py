"""
Gazebo simulation interface for robotic arm control and visualization.
Handles connection, model loading, and motion execution.
"""

import os
import time
import numpy as np
from typing import Tuple, Optional
from src.utils_logging import get_logger

logger = get_logger(__name__)

# Try to import Gazebo client
try:
    import rospy
    from gazebo_msgs.srv import GetModelState, SetModelState
    from gazebo_msgs.msg import ModelState
    from std_msgs.msg import Float64
    GAZEBO_AVAILABLE = True
except ImportError:
    GAZEBO_AVAILABLE = False
    logger.warning("ROS/Gazebo not available. Simulation mode disabled.")


class GazeboSimulator:
    """Interface for Gazebo simulation control."""
    
    def __init__(
        self,
        robot_name: str = "robotic_arm",
        base_link: str = "base_link",
        ee_link: str = "link6",
        num_joints: int = 6
    ):
        """
        Initialize Gazebo simulator.
        
        Args:
            robot_name: Name of robot model in Gazebo
            base_link: Base link name
            ee_link: End-effector link name
            num_joints: Number of joints
        """
        self.robot_name = robot_name
        self.base_link = base_link
        self.ee_link = ee_link
        self.num_joints = num_joints
        self.is_connected = False
        self.joint_publishers = []
        self.get_model_state_srv = None
        
        if GAZEBO_AVAILABLE:
            self._connect()
    
    def _connect(self) -> None:
        """Connect to Gazebo via ROS."""
        try:
            # Initialize ROS node
            if not rospy.core.is_initialized():
                rospy.init_node('nn_ik_solver', anonymous=True)
            
            # Wait for services
            rospy.wait_for_service('/gazebo/get_model_state', timeout=5)
            self.get_model_state_srv = rospy.ServiceProxy(
                '/gazebo/get_model_state',
                GetModelState
            )
            
            # Create joint publishers
            for i in range(self.num_joints):
                topic = f"/{self.robot_name}/joint{i+1}_position_controller/command"
                pub = rospy.Publisher(topic, Float64, queue_size=10)
                self.joint_publishers.append(pub)
            
            self.is_connected = True
            logger.info(f"Connected to Gazebo simulator for {self.robot_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Gazebo: {e}")
            self.is_connected = False
    
    def set_joint_positions(self, joint_angles: np.ndarray, delay: float = 0.1) -> None:
        """
        Set robot joint positions.
        
        Args:
            joint_angles: Target joint angles (radians)
            delay: Delay between commands (seconds)
        """
        if not self.is_connected or not GAZEBO_AVAILABLE:
            logger.warning("Gazebo not connected. Skipping joint position command.")
            return
        
        if len(joint_angles) != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joint angles, got {len(joint_angles)}"
            )
        
        try:
            for i, angle in enumerate(joint_angles):
                msg = Float64(data=float(angle))
                self.joint_publishers[i].publish(msg)
                time.sleep(delay / self.num_joints)
            
            logger.debug(f"Set joint positions: {np.degrees(joint_angles)}")
            
        except Exception as e:
            logger.error(f"Error setting joint positions: {e}")
    
    def get_ee_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get end-effector pose from Gazebo.
        
        Returns:
            Tuple of (position, orientation) or None if error
        """
        if not self.is_connected or not GAZEBO_AVAILABLE:
            logger.debug("Gazebo not connected. Cannot get EE pose.")
            return None
        
        try:
            state = self.get_model_state_srv(self.robot_name, self.ee_link)
            
            pos = state.pose.position
            orn = state.pose.orientation
            
            position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
            orientation = np.array([orn.x, orn.y, orn.z, orn.w], dtype=np.float32)
            
            return position, orientation
            
        except Exception as e:
            logger.error(f"Error getting EE pose: {e}")
            return None
    
    def disconnect(self) -> None:
        """Disconnect from Gazebo."""
        self.is_connected = False
        logger.info("Disconnected from Gazebo")


class GazeboSpawner:
    """Helper class for spawning models in Gazebo."""
    
    @staticmethod
    def spawn_model_from_urdf(
        model_name: str,
        urdf_path: str,
        position: Tuple[float, float, float] = (0, 0, 0),
        orientation: Tuple[float, float, float] = (0, 0, 0)
    ) -> bool:
        """
        Spawn a model from URDF file in Gazebo.
        
        Args:
            model_name: Name to give the model
            urdf_path: Path to URDF file
            position: Initial position (x, y, z)
            orientation: Initial orientation (r, p, y)
            
        Returns:
            True if successful
        """
        if not GAZEBO_AVAILABLE:
            logger.warning("ROS/Gazebo not available. Cannot spawn model.")
            return False
        
        try:
            # Read URDF file
            if not os.path.exists(urdf_path):
                logger.error(f"URDF file not found: {urdf_path}")
                return False
            
            with open(urdf_path, 'r') as f:
                model_xml = f.read()
            
            # Use ROS service to spawn
            from gazebo_msgs.srv import SpawnModel
            rospy.wait_for_service('/gazebo/spawn_urdf_model')
            spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            
            from geometry_msgs.msg import Pose, Point, Quaternion
            from tf_conversions import transformations as tf
            
            # Convert Euler to quaternion
            quat = tf.quaternion_from_euler(*orientation)
            
            pose = Pose(
                position=Point(x=position[0], y=position[1], z=position[2]),
                orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            )
            
            spawn_model(model_name, model_xml, "", pose, "world")
            logger.info(f"Spawned model {model_name} in Gazebo")
            return True
            
        except Exception as e:
            logger.error(f"Error spawning model: {e}")
            return False
    
    @staticmethod
    def delete_model(model_name: str) -> bool:
        """
        Delete a model from Gazebo.
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            True if successful
        """
        if not GAZEBO_AVAILABLE:
            logger.warning("ROS/Gazebo not available. Cannot delete model.")
            return False
        
        try:
            from gazebo_msgs.srv import DeleteModel
            rospy.wait_for_service('/gazebo/delete_model')
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            delete_model(model_name)
            logger.info(f"Deleted model {model_name} from Gazebo")
            return True
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
