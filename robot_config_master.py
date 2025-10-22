"""
MASTER ROBOT CONFIGURATION FILE

This is the SINGLE SOURCE OF TRUTH for your robot's physical parameters.
Every other file loads from this!

CRITICAL: Measure these values PRECISELY on your actual hardware!
Use calipers, rulers, CAD measurements - whatever you have.
Getting these wrong means your robot reaches to the wrong position in space!

How to measure:
1. Joint-to-joint distances (center of rotation to center of rotation)
2. Link offsets (perpendicular distances between joint axes)
3. Joint limits (physical stops, encoder limits)
4. End effector offset (from last joint to pour point)

This took you 3 weeks as a kid - it's THE most important file!
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from pathlib import Path


# ============================================================================
# ROBOT CONFIGURATION
# ============================================================================

@dataclass
class RobotConfig:
    """
    Complete robot configuration
    
    All measurements in meters and radians unless specified!
    """
    
    # ========== IDENTIFICATION ==========
    robot_name: str = "WineBot-6DOF-v1"
    serial_number: str = "CHANGE_ME"
    
    # ========== KINEMATIC CHAIN ==========
    # Link lengths (meters) - MEASURE THESE ON YOUR ROBOT!
    # Distance from joint i to joint i+1 along the link
    
    # Example structure for 6DOF arm:
    # Joint 0 (base rotation) → Joint 1 (shoulder)
    base_to_shoulder_height: float = 0.100  # Z-height from base to shoulder
    
    # Joint 1 (shoulder) → Joint 2 (elbow)  
    shoulder_to_elbow: float = 0.300  # Length of upper arm
    
    # Joint 2 (elbow) → Joint 3 (wrist_1)
    elbow_to_wrist1: float = 0.250  # Length of forearm
    
    # Joint 3 (wrist_1) → Joint 4 (wrist_2)
    wrist1_to_wrist2: float = 0.080  # Wrist link 1
    
    # Joint 4 (wrist_2) → Joint 5 (wrist_3)
    wrist2_to_wrist3: float = 0.070  # Wrist link 2
    
    # Joint 5 (wrist_3) → End effector (pour point)
    wrist3_to_endeffector: float = 0.150  # Gripper + bottle offset
    
    # ========== DH PARAMETERS (Denavit-Hartenberg) ==========
    # Format: [theta, d, a, alpha] for each joint
    # theta: joint angle (variable for revolute joints)
    # d: link offset along previous z
    # a: link length along x
    # alpha: link twist about x
    
    dh_params: List[List[float]] = None  # Will be set in __post_init__
    
    # ========== JOINT LIMITS ==========
    # (min, max) in radians
    # CRITICAL: Set these to your PHYSICAL limits!
    joint_limits: Dict[str, Tuple[float, float]] = None
    
    # ========== JOINT PROPERTIES ==========
    joint_names: List[str] = None
    joint_types: List[str] = None  # 'revolute' or 'prismatic'
    
    # Max velocities (rad/s)
    max_joint_velocities: List[float] = None
    
    # Max accelerations (rad/s^2)
    max_joint_accelerations: List[float] = None
    
    # ========== WORKSPACE LIMITS ==========
    # Cartesian workspace bounds (meters)
    workspace_x_min: float = -0.6
    workspace_x_max: float = 0.6
    workspace_y_min: float = -0.6
    workspace_y_max: float = 0.6
    workspace_z_min: float = 0.0
    workspace_z_max: float = 0.8
    
    # ========== SAFETY PARAMETERS ==========
    emergency_stop_decel: float = 5.0  # rad/s^2
    collision_threshold: float = 0.05  # meters - min distance to obstacles
    
    # ========== COMMUNICATION ==========
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    
    # ========== CAMERA CALIBRATION ==========
    camera_position: List[float] = None  # [x, y, z] in robot base frame
    camera_orientation: List[float] = None  # [roll, pitch, yaw]
    camera_intrinsics: Dict = None  # fx, fy, cx, cy
    
    def __post_init__(self):
        """Initialize derived parameters"""
        
        # Joint names
        if self.joint_names is None:
            self.joint_names = [
                "base_rotation",
                "shoulder", 
                "elbow",
                "wrist_1",
                "wrist_2",
                "wrist_3"
            ]
        
        # Joint types
        if self.joint_types is None:
            self.joint_types = ["revolute"] * 6
        
        # Joint limits (CHANGE THESE TO YOUR ROBOT'S ACTUAL LIMITS!)
        if self.joint_limits is None:
            self.joint_limits = {
                "base_rotation": (-np.pi, np.pi),
                "shoulder": (-np.pi/2, np.pi/2),
                "elbow": (-2*np.pi/3, 0),
                "wrist_1": (-np.pi/2, np.pi/2),
                "wrist_2": (-np.pi/2, np.pi/2),
                "wrist_3": (-np.pi, np.pi)
            }
        
        # Max velocities (TUNE THESE!)
        if self.max_joint_velocities is None:
            self.max_joint_velocities = [2.0, 1.5, 2.0, 2.5, 2.5, 3.0]  # rad/s
        
        # Max accelerations
        if self.max_joint_accelerations is None:
            self.max_joint_accelerations = [5.0, 4.0, 5.0, 6.0, 6.0, 8.0]  # rad/s^2
        
        # Camera defaults
        if self.camera_position is None:
            self.camera_position = [0.5, 0.0, 1.5]  # Overhead
        if self.camera_orientation is None:
            self.camera_orientation = [0, -np.pi/2, 0]  # Looking down
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': 600.0,
                'fy': 600.0,
                'cx': 320.0,
                'cy': 240.0,
                'width': 640,
                'height': 480
            }
        
        # Build DH parameters from link lengths
        self._build_dh_parameters()
    
    def _build_dh_parameters(self):
        """
        Build DH parameter table from link lengths
        
        DH Convention: [theta, d, a, alpha]
        - theta: rotation about Z (joint angle for revolute)
        - d: translation along Z
        - a: translation along X
        - alpha: rotation about X
        """
        self.dh_params = [
            # Joint 0: Base rotation
            [0, self.base_to_shoulder_height, 0, np.pi/2],
            
            # Joint 1: Shoulder
            [0, 0, self.shoulder_to_elbow, 0],
            
            # Joint 2: Elbow
            [0, 0, self.elbow_to_wrist1, 0],
            
            # Joint 3: Wrist 1
            [0, 0, self.wrist1_to_wrist2, np.pi/2],
            
            # Joint 4: Wrist 2
            [0, 0, self.wrist2_to_wrist3, -np.pi/2],
            
            # Joint 5: Wrist 3 (end effector)
            [0, self.wrist3_to_endeffector, 0, 0]
        ]
    
    def get_total_reach(self) -> float:
        """Calculate maximum reach of robot"""
        return (self.shoulder_to_elbow + 
                self.elbow_to_wrist1 + 
                self.wrist1_to_wrist2 + 
                self.wrist2_to_wrist3 + 
                self.wrist3_to_endeffector)
    
    def save(self, filepath: str):
        """Save configuration to JSON"""
        config_dict = {
            'robot_name': self.robot_name,
            'serial_number': self.serial_number,
            'link_lengths': {
                'base_to_shoulder_height': self.base_to_shoulder_height,
                'shoulder_to_elbow': self.shoulder_to_elbow,
                'elbow_to_wrist1': self.elbow_to_wrist1,
                'wrist1_to_wrist2': self.wrist1_to_wrist2,
                'wrist2_to_wrist3': self.wrist2_to_wrist3,
                'wrist3_to_endeffector': self.wrist3_to_endeffector
            },
            'joint_limits': {k: list(v) for k, v in self.joint_limits.items()},
            'max_joint_velocities': self.max_joint_velocities,
            'max_joint_accelerations': self.max_joint_accelerations,
            'workspace_limits': {
                'x': [self.workspace_x_min, self.workspace_x_max],
                'y': [self.workspace_y_min, self.workspace_y_max],
                'z': [self.workspace_z_min, self.workspace_z_max]
            },
            'serial_port': self.serial_port,
            'baudrate': self.baudrate,
            'camera': {
                'position': self.camera_position,
                'orientation': self.camera_orientation,
                'intrinsics': self.camera_intrinsics
            }
        }
        
        Path(filepath).write_text(json.dumps(config_dict, indent=2))
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON"""
        config_dict = json.loads(Path(filepath).read_text())
        
        # Extract link lengths
        links = config_dict['link_lengths']
        
        config = cls(
            robot_name=config_dict['robot_name'],
            serial_number=config_dict['serial_number'],
            base_to_shoulder_height=links['base_to_shoulder_height'],
            shoulder_to_elbow=links['shoulder_to_elbow'],
            elbow_to_wrist1=links['elbow_to_wrist1'],
            wrist1_to_wrist2=links['wrist1_to_wrist2'],
            wrist2_to_wrist3=links['wrist2_to_wrist3'],
            wrist3_to_endeffector=links['wrist3_to_endeffector'],
            serial_port=config_dict['serial_port'],
            baudrate=config_dict['baudrate']
        )
        
        # Restore other parameters
        config.joint_limits = {k: tuple(v) for k, v in config_dict['joint_limits'].items()}
        config.max_joint_velocities = config_dict['max_joint_velocities']
        config.max_joint_accelerations = config_dict['max_joint_accelerations']
        
        workspace = config_dict['workspace_limits']
        config.workspace_x_min, config.workspace_x_max = workspace['x']
        config.workspace_y_min, config.workspace_y_max = workspace['y']
        config.workspace_z_min, config.workspace_z_max = workspace['z']
        
        cam = config_dict['camera']
        config.camera_position = cam['position']
        config.camera_orientation = cam['orientation']
        config.camera_intrinsics = cam['intrinsics']
        
        return config


# ============================================================================
# MEASUREMENT GUIDE
# ============================================================================

def print_measurement_guide():
    """Print guide for measuring your robot"""
    print("="*70)
    print("ROBOT MEASUREMENT GUIDE")
    print("="*70)
    print("""
HOW TO MEASURE YOUR ROBOT ACCURATELY:

1. BASE TO SHOULDER HEIGHT
   - Measure vertical distance from base plate to shoulder joint axis
   - Use calipers or ruler
   - Include any mounting plates!

2. SHOULDER TO ELBOW (Upper Arm)
   - Distance from shoulder rotation axis to elbow rotation axis
   - This is usually the longest link
   - Measure center-to-center

3. ELBOW TO WRIST (Forearm)
   - Distance from elbow axis to wrist_1 axis
   - Second longest link typically

4. WRIST LINKS
   - Measure each wrist joint axis to the next
   - These are usually shorter (5-10cm)

5. END EFFECTOR OFFSET
   - From last wrist joint to the POUR POINT (tip of bottle)
   - NOT just to gripper - to where wine actually comes out!
   - This changes if you swap grippers!

TIPS:
- Measure 3 times, average the results
- Use metric (meters) for consistency  
- Check your CAD if you have it
- Verify with forward kinematics after (move to known angles, measure actual position)

JOINT LIMITS:
- Manually move each joint to its limits
- Note the encoder/angle readings
- Add safety margin (5-10 degrees)
- Check for soft limits vs hard stops

VALIDATION:
After entering measurements, run the validation script to check:
- Forward kinematics matches reality
- Workspace is correct
- No impossible configurations
    """)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config(config: RobotConfig):
    """Validate robot configuration"""
    print("\n" + "="*70)
    print("VALIDATING ROBOT CONFIGURATION")
    print("="*70)
    
    errors = []
    warnings = []
    
    # Check link lengths are positive
    links = [
        config.base_to_shoulder_height,
        config.shoulder_to_elbow,
        config.elbow_to_wrist1,
        config.wrist1_to_wrist2,
        config.wrist2_to_wrist3,
        config.wrist3_to_endeffector
    ]
    
    for i, length in enumerate(links):
        if length <= 0:
            errors.append(f"Link {i} has non-positive length: {length}")
        if length > 1.0:
            warnings.append(f"Link {i} is very long (>1m): {length}m - is this correct?")
    
    # Check joint limits are valid
    for name, (lower, upper) in config.joint_limits.items():
        if lower >= upper:
            errors.append(f"Joint '{name}' has invalid limits: [{lower}, {upper}]")
    
    # Check reach
    total_reach = config.get_total_reach()
    print(f"\nTotal reach: {total_reach:.3f}m")
    
    if total_reach > 1.5:
        warnings.append(f"Very large reach: {total_reach:.3f}m - industrial robot?")
    if total_reach < 0.3:
        warnings.append(f"Very small reach: {total_reach:.3f}m - toy robot?")
    
    # Check workspace
    workspace_volume = (
        (config.workspace_x_max - config.workspace_x_min) *
        (config.workspace_y_max - config.workspace_y_min) *
        (config.workspace_z_max - config.workspace_z_min)
    )
    print(f"Workspace volume: {workspace_volume:.3f} m³")
    
    # Report
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("\n✅ Configuration looks good!")
    
    return len(errors) == 0


# ============================================================================
# EXAMPLE: YOUR ROBOT
# ============================================================================

def create_my_robot_config():
    """
    Create configuration for YOUR specific robot
    
    FILL IN YOUR ACTUAL MEASUREMENTS HERE!
    """
    
    config = RobotConfig(
        robot_name="MyWineBot-001",
        serial_number="SN-12345",
        
        # ===== MEASURE THESE ON YOUR ROBOT! =====
        base_to_shoulder_height=0.120,  # YOUR measurement
        shoulder_to_elbow=0.350,        # YOUR measurement
        elbow_to_wrist1=0.280,          # YOUR measurement
        wrist1_to_wrist2=0.085,         # YOUR measurement
        wrist2_to_wrist3=0.075,         # YOUR measurement
        wrist3_to_endeffector=0.180,    # YOUR measurement (to pour point!)
        
        # ===== YOUR JOINT LIMITS =====
        # Move each joint manually and note the limits!
        
        # ===== YOUR SERIAL PORT =====
        serial_port="/dev/ttyUSB0",     # Or COM3 on Windows
        baudrate=115200
    )
    
    return config


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ROBOT CONFIGURATION MASTER FILE")
    print("="*70)
    
    # Show measurement guide
    print_measurement_guide()
    
    input("\nPress Enter after you've measured your robot...")
    
    # Create config
    print("\nCreating robot configuration...")
    config = create_my_robot_config()
    
    # Validate
    if validate_config(config):
        # Save
        config.save("robot_config.json")
        print("\n✅ Configuration saved to robot_config.json")
        print("\nNow use this in all other files:")
        print("  from robot_config_master import RobotConfig")
        print("  config = RobotConfig.load('robot_config.json')")
    else:
        print("\n❌ Fix errors before saving!")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
1. Fill in YOUR measurements in create_my_robot_config()
2. Run this script to validate
3. Save to robot_config.json
4. Update all other code to load from this config
5. Test forward kinematics in simulation
6. Calibrate with real robot movements
7. Fine-tune if needed

Remember: This file is the SINGLE SOURCE OF TRUTH!
If anything is wrong here, EVERYTHING will be wrong!
    """)