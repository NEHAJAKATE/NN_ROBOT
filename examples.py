"""
Comprehensive Examples for NN IK Robot Controller

This file demonstrates various use cases of the robotics system:
1. Basic IK solving
2. Batch processing
3. Gazebo integration
4. Model training
5. Performance testing
"""

import numpy as np
import time
from src.nn_ik_solver import EnhancedNNIKSolver
from src.main_controller import RoboticArmIKController
from src.ik_dataset import generate_dataset


# ============================================================================
# Example 1: Basic IK Solving
# ============================================================================

def example_basic_ik():
    """
    Example 1: Solve inverse kinematics for a single pose
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Inverse Kinematics Solving")
    print("="*70)
    
    # Initialize solver
    solver = EnhancedNNIKSolver(model_path="ik_nn_model.pth")
    
    # Target pose
    position = np.array([0.5, 0.2, 0.4], dtype=np.float32)  # x, y, z in meters
    orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # quaternion
    
    print(f"\nTarget Position: {position}")
    print(f"Target Orientation (quaternion): {orientation}")
    
    # Solve with confidence score
    joint_angles_rad, confidence = solver.solve(
        position, orientation,
        return_confidence=True
    )
    
    # Convert to degrees for display
    joint_angles_deg = np.degrees(joint_angles_rad)
    
    print(f"\nSolution:")
    print(f"  Joint Angles (radians): {joint_angles_rad}")
    print(f"  Joint Angles (degrees): {joint_angles_deg}")
    print(f"  Confidence: {confidence:.4f}")


# ============================================================================
# Example 2: Batch IK Solving
# ============================================================================

def example_batch_ik():
    """
    Example 2: Solve inverse kinematics for multiple poses
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Processing - Multiple Poses")
    print("="*70)
    
    solver = EnhancedNNIKSolver(model_path="ik_nn_model.pth")
    
    # Generate 5 random target poses
    num_poses = 5
    positions = np.random.rand(num_poses, 3) * 0.5 + np.array([0.2, -0.25, 0.2])
    orientations = np.tile([0.0, 0.0, 0.0, 1.0], (num_poses, 1))
    
    print(f"\nSolving for {num_poses} different poses...")
    
    # Solve batch
    start_time = time.time()
    solutions = solver.solve_batch(positions, orientations)
    elapsed_time = time.time() - start_time
    
    print(f"\nBatch Processing Results:")
    print(f"  Processed: {num_poses} poses")
    print(f"  Time: {elapsed_time:.4f} seconds")
    print(f"  Speed: {num_poses/elapsed_time:.1f} poses/second")
    
    print(f"\nFirst 3 Solutions (degrees):")
    for i in range(min(3, num_poses)):
        print(f"  Pose {i+1}: {np.degrees(solutions[i])}")


# ============================================================================
# Example 3: Gazebo Integration
# ============================================================================

def example_gazebo_control():
    """
    Example 3: Solve IK and execute in Gazebo simulation
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Gazebo Simulation Integration")
    print("="*70)
    
    # Initialize controller
    controller = RoboticArmIKController()
    
    # Define target poses
    targets = [
        {
            'position': np.array([0.5, 0.0, 0.5], dtype=np.float32),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            'name': 'Ready Position'
        },
        {
            'position': np.array([0.4, 0.2, 0.4], dtype=np.float32),
            'orientation': np.array([0.0, 0.707, 0.0, 0.707], dtype=np.float32),
            'name': 'Pick Position'
        },
        {
            'position': np.array([0.3, -0.3, 0.6], dtype=np.float32),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            'name': 'Place Position'
        },
    ]
    
    print("\nNote: Gazebo must be running. Start with:")
    print("  python -m src.main_controller --launch-gazebo")
    print("\nExecuting sequential movements...")
    
    for i, target in enumerate(targets):
        print(f"\n--- Movement {i+1}: {target['name']} ---")
        print(f"Position: {target['position']}")
        
        result = controller.execute_movement(
            position=target['position'],
            orientation=target['orientation'],
            execute_in_gazebo=True,
            wait_for_completion=True
        )
        
        if result['success']:
            print(f"✓ Success")
            angles_deg = result['joint_angles_degrees']
            print(f"  Joint Angles: {angles_deg}")
            
            if 'position_error' in result:
                print(f"  Position Error: {result['position_error']:.6f} m")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")
        
        time.sleep(1)  # Pause between movements


# ============================================================================
# Example 4: Model Training
# ============================================================================

def example_training():
    """
    Example 4: Generate training data and train a new model
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Training a New Neural Network Model")
    print("="*70)
    
    # Step 1: Generate training data
    print("\nStep 1: Generating training dataset...")
    print("This will collect poses using PyBullet physics simulation")
    print("(This may take a few minutes)")
    
    num_samples = 1000  # Use fewer samples for demo (normally 5000+)
    poses, joints = generate_dataset(n_samples=num_samples)
    
    print(f"✓ Generated {num_samples} samples")
    print(f"  Poses shape: {poses.shape}")
    print(f"  Joints shape: {joints.shape}")
    
    # Step 2: Train model
    print("\nStep 2: Training neural network...")
    
    solver = EnhancedNNIKSolver()
    history = solver.train(
        poses, joints,
        epochs=20,  # Reduced for demo
        batch_size=64,
        learning_rate=0.001,
        validation_split=0.2,
        save_path="ik_nn_model_demo.pth"
    )
    
    print("\n✓ Training Complete")
    print(f"  Final Training Loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final Validation Loss: {history['val_loss'][-1]:.6f}")
    print(f"  Model saved to: ik_nn_model_demo.pth")


# ============================================================================
# Example 5: Performance Analysis
# ============================================================================

def example_performance_analysis():
    """
    Example 5: Analyze system performance and accuracy
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Performance Analysis")
    print("="*70)
    
    solver = EnhancedNNIKSolver()
    
    # Generate random test poses
    num_test = 100
    test_positions = np.random.rand(num_test, 3) * 0.5 + np.array([0.2, -0.25, 0.2])
    test_orientations = np.tile([0.0, 0.0, 0.0, 1.0], (num_test, 1))
    
    print(f"\nPerformance Test: Solving {num_test} random poses")
    
    # Measure speed
    start_time = time.time()
    solutions = solver.solve_batch(test_positions, test_orientations)
    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    speed = num_test / elapsed_time
    avg_time_per_pose = (elapsed_time / num_test) * 1000  # Convert to ms
    
    print(f"\nSpeed Results:")
    print(f"  Total time: {elapsed_time:.4f} seconds")
    print(f"  Average time per pose: {avg_time_per_pose:.2f} ms")
    print(f"  Throughput: {speed:.1f} poses/second")
    
    # Check GPU usage if available
    try:
        import torch
        print(f"\nCompute Device:")
        print(f"  GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    except:
        pass
    
    # Analyze solution statistics
    solutions_deg = np.degrees(solutions)
    print(f"\nSolution Statistics (degrees):")
    for i in range(6):
        print(f"  Joint {i+1}: min={solutions_deg[:,i].min():.2f}° "
              f"max={solutions_deg[:,i].max():.2f}° "
              f"std={solutions_deg[:,i].std():.2f}°")


# ============================================================================
# Example 6: Interactive Pose Input
# ============================================================================

def example_interactive_input():
    """
    Example 6: Get user input for poses and solve
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Interactive Pose Input")
    print("="*70)
    
    solver = EnhancedNNIKSolver()
    controller = RoboticArmIKController()
    
    print("\nEnter target pose parameters:")
    
    try:
        # Get position
        x = float(input("Position X (meters) [default 0.5]: ") or "0.5")
        y = float(input("Position Y (meters) [default 0.0]: ") or "0.0")
        z = float(input("Position Z (meters) [default 0.4]: ") or "0.4")
        
        # Get rotation (Euler angles)
        roll = float(input("Rotation Roll (degrees) [default 0]: ") or "0")
        pitch = float(input("Rotation Pitch (degrees) [default 0]: ") or "0")
        yaw = float(input("Rotation Yaw (degrees) [default 0]: ") or "0")
        
        # Convert to poses
        position = np.array([x, y, z], dtype=np.float32)
        orientation = controller._euler_to_quaternion(roll, pitch, yaw)
        
        # Solve
        result = controller.solve_ik(position, orientation)
        
        if result['success']:
            print(f"\n✓ IK Solution Found:")
            print(f"  Joint Angles (degrees): {result['joint_angles']}")
            print(f"  Joint Angles (radians): {result['joint_angles_rad']}")
        else:
            print(f"\n✗ IK Solving Failed: {result['error']}")
    
    except ValueError:
        print("Invalid input. Please enter numeric values.")


# ============================================================================
# Main: Run Examples
# ============================================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        '1': ('Basic IK Solving', example_basic_ik),
        '2': ('Batch IK Processing', example_batch_ik),
        '3': ('Gazebo Integration', example_gazebo_control),
        '4': ('Model Training', example_training),
        '5': ('Performance Analysis', example_performance_analysis),
        '6': ('Interactive Input', example_interactive_input),
    }
    
    print("="*70)
    print("NN IK Robot Controller - Examples")
    print("="*70)
    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}: {name}")
    print("  0: Run All Examples")
    print("  q: Quit")
    
    choice = input("\nSelect example (0-6, q to quit): ").strip()
    
    try:
        if choice == 'q':
            print("Exiting...")
            sys.exit(0)
        elif choice == '0':
            print("\nRunning all examples...")
            for key in sorted(examples.keys()):
                try:
                    examples[key][1]()
                    time.sleep(2)
                except Exception as e:
                    print(f"Error in example {key}: {e}")
        elif choice in examples:
            examples[choice][1]()
        else:
            print("Invalid selection")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
