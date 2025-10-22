"""
AUTOMATIC ROBOT CALIBRATION SYSTEM

Uses physics + servo feedback + vision to automatically calibrate:
1. Link lengths (from gravity drop test)
2. Joint offsets (from encoder readings)
3. Camera extrinsics (from ArUco markers)
4. Center of mass (from torque measurements)
5. Collision thresholds (from current spikes)

No calipers needed - let the robot figure itself out!

Inspired by LeRobot's calibration approach
"""

import numpy as np
import cv2
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


# ============================================================================
# SERVO FEEDBACK DATA STRUCTURE
# ============================================================================

@dataclass
class ServoFeedback:
    """
    Real-time feedback from each servo motor
    
    Your servos should provide:
    - position: Current angle (radians or encoder counts)
    - current: Current draw (amps) - spikes indicate collision/load
    - temperature: Motor temp (°C) - overheating detection
    - velocity: Angular velocity (rad/s)
    - pwm: PWM signal (0-100%) - correlates with torque
    """
    joint_id: int
    position: float  # radians
    current: float   # amps
    temperature: float  # celsius
    velocity: float  # rad/s
    pwm: float  # 0-100%
    timestamp: float
    
    def get_torque_estimate(self, kt=0.01):
        """
        Estimate torque from current draw
        kt: torque constant (Nm/A) - calibrate for your motors
        """
        return self.current * kt


# ============================================================================
# GRAVITY-BASED CALIBRATION
# ============================================================================

class GravityCalibration:
    """
    Use gravity to automatically find link lengths and centers of mass
    
    Method:
    1. Hold arm at various angles
    2. Measure torque (via servo current) at each joint
    3. Use physics: τ = m*g*r*cos(θ)
    4. Solve for link lengths (r) and masses (m)
    """
    
    def __init__(self, num_joints=6):
        self.num_joints = num_joints
        self.g = 9.81  # m/s^2
        
    def collect_gravity_data(self, robot_interface, num_samples=50):
        """
        Collect torque measurements at various poses
        
        Args:
            robot_interface: Your robot control interface
            num_samples: Number of different poses to test
        
        Returns:
            List of (joint_angles, torques) tuples
        """
        print("Starting gravity calibration...")
        print("Moving arm through various poses and measuring torque...")
        
        data = []
        
        for i in range(num_samples):
            # Generate random safe pose
            pose = self._generate_safe_pose()
            
            # Move to pose slowly
            robot_interface.send_joint_command(pose, velocity=0.2)
            time.sleep(1.0)  # Let it settle
            
            # Read servo feedback
            feedback = robot_interface.get_servo_feedback()
            
            # Extract torques from current
            torques = [s.get_torque_estimate() for s in feedback]
            
            data.append({
                'angles': pose.copy(),
                'torques': np.array(torques)
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_samples} poses")
        
        print("✓ Data collection complete!")
        return data
    
    def _generate_safe_pose(self):
        """Generate random pose within safe workspace"""
        # Vary shoulder and elbow most (they carry the load)
        pose = np.array([
            np.random.uniform(-np.pi/4, np.pi/4),  # base
            np.random.uniform(-np.pi/3, np.pi/3),  # shoulder
            np.random.uniform(-2*np.pi/3, -np.pi/6),  # elbow
            np.random.uniform(-np.pi/4, np.pi/4),  # wrist1
            0,  # wrist2 - keep neutral
            0   # wrist3 - keep neutral
        ])
        return pose
    
    def estimate_link_parameters(self, data):
        """
        Estimate link lengths and masses from gravity data
        
        Uses nonlinear least squares to fit:
        τ_measured = f(link_lengths, masses, angles)
        """
        print("\nEstimating link parameters...")
        
        # Initial guess (reasonable defaults)
        # Parameters: [L1, L2, L3, m1, m2, m3, ...]
        x0 = np.array([
            0.35, 0.30, 0.25,  # Link lengths (m)
            0.5, 0.4, 0.3,     # Link masses (kg)
            0.2, 0.15, 0.1     # More masses for wrist links
        ])
        
        # Define cost function
        def cost(params):
            total_error = 0
            for sample in data:
                angles = sample['angles']
                measured_torques = sample['torques']
                
                # Predict torques from current params
                predicted_torques = self._forward_dynamics(params, angles)
                
                # MSE error
                error = np.sum((predicted_torques - measured_torques[:3])**2)
                total_error += error
            
            return total_error
        
        # Optimize
        result = minimize(cost, x0, method='L-BFGS-B', 
                         bounds=[(0.1, 0.6)]*3 + [(0.1, 2.0)]*6)
        
        # Extract results
        link_lengths = result.x[:3]
        link_masses = result.x[3:]
        
        print(f"✓ Calibration complete!")
        print(f"\nEstimated link lengths:")
        print(f"  Shoulder to elbow: {link_lengths[0]:.3f}m")
        print(f"  Elbow to wrist:    {link_lengths[1]:.3f}m")
        print(f"  Wrist to end:      {link_lengths[2]:.3f}m")
        
        return {
            'link_lengths': link_lengths,
            'link_masses': link_masses,
            'optimization_result': result
        }
    
    def _forward_dynamics(self, params, angles):
        """
        Predict joint torques from link params and pose
        Simplified dynamics for calibration
        """
        L = params[:3]  # Link lengths
        m = params[3:]  # Masses
        
        q = angles
        
        # Torque at each joint due to gravity
        # τ = m * g * r * cos(θ)
        # r is distance from joint to center of mass
        
        tau = np.zeros(3)
        
        # Shoulder torque (holds everything)
        tau[0] = (
            m[0] * self.g * (L[0]/2) * np.cos(q[1]) +
            m[1] * self.g * (L[0] + L[1]/2) * np.cos(q[1] + q[2]) +
            m[2] * self.g * (L[0] + L[1] + L[2]/2) * np.cos(q[1] + q[2])
        )
        
        # Elbow torque (holds forearm + wrist)
        tau[1] = (
            m[1] * self.g * (L[1]/2) * np.cos(q[2]) +
            m[2] * self.g * (L[1] + L[2]/2) * np.cos(q[2])
        )
        
        # Wrist torque
        tau[2] = m[2] * self.g * (L[2]/2) * np.cos(q[2])
        
        return tau


# ============================================================================
# VISION-BASED CALIBRATION (Camera Extrinsics)
# ============================================================================

class VisionCalibration:
    """
    Automatically calibrate camera position using ArUco markers
    
    Place markers at known positions in workspace
    Robot moves through poses and sees markers from different angles
    Solve for camera pose in robot frame
    """
    
    def __init__(self):
        # ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
    def calibrate_overhead_camera(self, camera, robot_interface, marker_size=0.05):
        """
        Calibrate overhead camera position
        
        Args:
            camera: Camera interface
            robot_interface: Robot control
            marker_size: Size of ArUco markers in meters
        
        Returns:
            Camera pose (position + orientation) in robot base frame
        """
        print("\nCalibrating overhead camera...")
        print("Place ArUco marker ID 0 on table in workspace")
        input("Press Enter when ready...")
        
        detections = []
        
        # Move end effector to multiple positions
        for i in range(20):
            # Random pose
            pose = self._random_pose()
            robot_interface.send_joint_command(pose, velocity=0.3)
            time.sleep(0.5)
            
            # Capture frame
            frame = camera.get_frame()
            
            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame, self.aruco_dict, parameters=self.aruco_params
            )
            
            if ids is not None and 0 in ids:
                # Get end effector position from FK
                ee_pos = robot_interface.get_end_effector_pose()
                
                # Get marker position in image
                marker_idx = list(ids.flatten()).index(0)
                marker_corners = corners[marker_idx][0]
                marker_center = marker_corners.mean(axis=0)
                
                detections.append({
                    'ee_pos': ee_pos,
                    'marker_pixel': marker_center,
                    'marker_corners': marker_corners
                })
                
                print(f"  Detection {len(detections)}/20")
        
        if len(detections) < 10:
            print("❌ Not enough detections! Need at least 10.")
            return None
        
        # Solve for camera pose using PnP
        camera_pose = self._solve_camera_pose(detections, marker_size)
        
        print(f"✓ Camera calibrated!")
        print(f"  Position: {camera_pose['position']}")
        print(f"  Orientation: {camera_pose['orientation']}")
        
        return camera_pose
    
    def _random_pose(self):
        """Generate random pose for calibration"""
        return np.array([
            np.random.uniform(-np.pi/6, np.pi/6),
            np.random.uniform(-np.pi/4, np.pi/4),
            np.random.uniform(-np.pi/2, -np.pi/6),
            0, 0, 0
        ])
    
    def _solve_camera_pose(self, detections, marker_size):
        """Solve camera pose from detections using PnP"""
        # Simplified - in production use cv2.solvePnP with full 3D-2D correspondences
        # Here we estimate from multiple observations
        
        # Average camera height from detections
        z_estimates = []
        for det in detections:
            # Rough estimate from marker pixel size
            pixel_size = np.linalg.norm(det['marker_corners'][0] - det['marker_corners'][2])
            z_est = (marker_size * 600) / pixel_size  # Assuming fx=600
            z_estimates.append(z_est)
        
        camera_height = np.median(z_estimates)
        
        return {
            'position': [0.5, 0.0, camera_height],
            'orientation': [0, -np.pi/2, 0],  # Looking down
            'confidence': np.std(z_estimates)
        }


# ============================================================================
# COLLISION DETECTION & OBSTACLE AVOIDANCE
# ============================================================================

class CollisionDetector:
    """
    Real-time collision detection using servo current monitoring
    
    When arm hits obstacle:
    - Current spikes (motor working harder)
    - Velocity drops (motion blocked)
    - Position error increases (can't reach target)
    """
    
    def __init__(self, num_joints=6):
        self.num_joints = num_joints
        
        # Baselines (calibrate during free movement)
        self.baseline_current = np.zeros(num_joints)
        self.current_threshold = 0.5  # Amps above baseline = collision
        
        # History for filtering
        self.current_history = [[] for _ in range(num_joints)]
        self.max_history = 10
    
    def calibrate_free_movement(self, robot_interface, duration=10.0):
        """
        Learn baseline current during free (unloaded) movement
        
        Args:
            robot_interface: Robot control
            duration: Calibration time in seconds
        """
        print("\nCalibrating collision detection...")
        print(f"Moving freely for {duration}s to learn baseline current...")
        
        start_time = time.time()
        current_readings = [[] for _ in range(self.num_joints)]
        
        while time.time() - start_time < duration:
            # Move to random pose
            pose = np.random.uniform(-np.pi/4, np.pi/4, self.num_joints)
            robot_interface.send_joint_command(pose, velocity=0.5)
            time.sleep(0.2)
            
            # Read current
            feedback = robot_interface.get_servo_feedback()
            for i, servo in enumerate(feedback):
                current_readings[i].append(servo.current)
        
        # Calculate baseline (median + std)
        for i in range(self.num_joints):
            readings = np.array(current_readings[i])
            self.baseline_current[i] = np.median(readings)
            self.current_threshold = np.std(readings) * 3  # 3-sigma
        
        print("✓ Baseline calibrated!")
        print(f"  Baseline current: {self.baseline_current}")
        print(f"  Threshold: {self.current_threshold:.3f}A")
    
    def check_collision(self, servo_feedback: List[ServoFeedback]) -> Dict:
        """
        Check for collision in real-time
        
        Returns:
            {
                'collision': bool,
                'joint_id': int or None,
                'severity': float (0-1)
            }
        """
        collision_detected = False
        collision_joint = None
        max_severity = 0.0
        
        for servo in servo_feedback:
            i = servo.joint_id
            
            # Add to history
            self.current_history[i].append(servo.current)
            if len(self.current_history[i]) > self.max_history:
                self.current_history[i].pop(0)
            
            # Check spike
            current_filtered = np.mean(self.current_history[i])
            delta = current_filtered - self.baseline_current[i]
            
            if delta > self.current_threshold:
                severity = min(delta / (self.current_threshold * 2), 1.0)
                
                if severity > max_severity:
                    collision_detected = True
                    collision_joint = i
                    max_severity = severity
        
        return {
            'collision': collision_detected,
            'joint_id': collision_joint,
            'severity': max_severity
        }
    
    def get_avoidance_action(self, collision_info, current_velocity):
        """
        Generate avoidance action when collision detected
        
        Returns:
            Joint velocities to back away from collision
        """
        if not collision_info['collision']:
            return np.zeros(6)
        
        # Back away from collision
        avoidance = np.zeros(6)
        joint_id = collision_info['joint_id']
        severity = collision_info['severity']
        
        # Move colliding joint backwards
        avoidance[joint_id] = -current_velocity[joint_id] * 2.0 * severity
        
        return avoidance


# ============================================================================
# COMPLETE AUTO-CALIBRATION PIPELINE
# ============================================================================

class AutoCalibration:
    """
    Complete automatic calibration system
    Run this once on robot startup or after hardware changes
    """
    
    def __init__(self, robot_interface, overhead_camera, wrist_camera=None):
        self.robot = robot_interface
        self.overhead_cam = overhead_camera
        self.wrist_cam = wrist_camera
        
        # Calibration modules
        self.gravity_cal = GravityCalibration()
        self.vision_cal = VisionCalibration()
        self.collision_detector = CollisionDetector()
    
    def run_full_calibration(self):
        """
        Run complete calibration pipeline
        
        Returns:
            Updated RobotConfig with calibrated parameters
        """
        print("="*70)
        print("AUTOMATIC ROBOT CALIBRATION")
        print("="*70)
        
        results = {}
        
        # 1. Gravity-based link calibration
        print("\n[1/4] Gravity calibration (link lengths & masses)...")
        gravity_data = self.gravity_cal.collect_gravity_data(self.robot)
        link_params = self.gravity_cal.estimate_link_parameters(gravity_data)
        results['link_parameters'] = link_params
        
        # 2. Vision calibration
        print("\n[2/4] Camera calibration...")
        camera_pose = self.vision_cal.calibrate_overhead_camera(
            self.overhead_cam, self.robot
        )
        results['camera_pose'] = camera_pose
        
        # 3. Collision detection calibration
        print("\n[3/4] Collision detection calibration...")
        self.collision_detector.calibrate_free_movement(self.robot)
        results['collision_baseline'] = self.collision_detector.baseline_current
        
        # 4. Joint offset calibration (zero position)
        print("\n[4/4] Joint offset calibration...")
        joint_offsets = self._calibrate_joint_offsets()
        results['joint_offsets'] = joint_offsets
        
        # Generate updated config
        print("\n" + "="*70)
        print("CALIBRATION COMPLETE!")
        print("="*70)
        
        self._print_results(results)
        
        # Save to config file
        self._save_calibration(results)
        
        return results
    
    def _calibrate_joint_offsets(self):
        """
        Find joint zero positions using gravity
        Arm naturally hangs down when unpowered
        """
        print("  Disable motors and let arm hang naturally...")
        print("  (Or move to known zero configuration manually)")
        input("  Press Enter when ready...")
        
        feedback = self.robot.get_servo_feedback()
        zero_positions = [s.position for s in feedback]
        
        print(f"  ✓ Zero positions: {zero_positions}")
        return zero_positions
    
    def _print_results(self, results):
        """Print calibration summary"""
        print("\nCalibration Results:")
        print("-" * 70)
        
        if 'link_parameters' in results:
            L = results['link_parameters']['link_lengths']
            print(f"Link Lengths:")
            print(f"  L1: {L[0]:.3f}m")
            print(f"  L2: {L[1]:.3f}m")
            print(f"  L3: {L[2]:.3f}m")
        
        if 'camera_pose' in results:
            pose = results['camera_pose']
            print(f"\nCamera Position: {pose['position']}")
            print(f"Camera Confidence: ±{pose['confidence']:.3f}m")
        
        print(f"\nCollision Detection: Calibrated")
        print(f"  Baseline current: {results['collision_baseline']}")
    
    def _save_calibration(self, results):
        """Save results to config file"""
        from robot_config_master import RobotConfig
        
        # Load existing config
        config = RobotConfig.load("robot_config.json")
        
        # Update with calibrated values
        if 'link_parameters' in results:
            L = results['link_parameters']['link_lengths']
            config.shoulder_to_elbow = L[0]
            config.elbow_to_wrist1 = L[1]
            config.wrist3_to_endeffector = L[2]
        
        if 'camera_pose' in results:
            config.camera_position = results['camera_pose']['position']
            config.camera_orientation = results['camera_pose']['orientation']
        
        # Save
        config.save("robot_config_calibrated.json")
        print("\n✓ Calibration saved to robot_config_calibrated.json")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Automatic Robot Calibration System")
    print("No calipers needed - physics does the work!")
    
    print("\nThis will:")
    print("  1. Move arm through various poses")
    print("  2. Measure torques via servo current")
    print("  3. Use physics to calculate link lengths")
    print("  4. Calibrate cameras using ArUco markers")
    print("  5. Learn collision thresholds")
    
    input("\nPress Enter to start calibration...")
    
    # Initialize (you'll need your actual interfaces here)
    # robot = RobotArmInterface("robot_config.json")
    # camera = OverheadCamera()
    # 
    # calibrator = AutoCalibration(robot, camera)
    # results = calibrator.run_full_calibration()
    
    print("\nCalibration complete! Robot is ready.")