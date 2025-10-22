"""
FLUID DYNAMICS FOR WINE POURING

Models the actual physics of liquid pouring:
1. Liquid stream trajectory (parabolic motion under gravity)
2. Flow rate based on bottle angle and remaining volume
3. Fill level detection in cup
4. Splash prediction
5. Optimal pour angle calculation

This is CRITICAL for accurate pouring - you can't just assume
the wine goes straight down!
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, List, Optional


# ============================================================================
# FLUID PROPERTIES
# ============================================================================

@dataclass
class FluidProperties:
    """Physical properties of the liquid being poured"""
    
    # Wine properties (at ~20°C)
    density: float = 990.0  # kg/m³ (slightly less than water)
    viscosity: float = 0.0015  # Pa·s (dynamic viscosity)
    surface_tension: float = 0.050  # N/m
    
    # Flow properties
    discharge_coefficient: float = 0.6  # Typical for bottle opening
    

@dataclass
class BottleState:
    """Current state of the wine bottle"""
    
    position: np.ndarray  # [x, y, z] bottle spout position
    orientation: np.ndarray  # [roll, pitch, yaw] bottle orientation
    tilt_angle: float  # Angle from vertical (radians)
    
    # Bottle geometry
    spout_diameter: float = 0.015  # 15mm typical wine bottle
    total_volume: float = 0.750  # 750ml bottle
    current_volume: float = 0.750  # How much wine left
    bottle_height: float = 0.30  # 30cm typical
    
    def get_liquid_height(self) -> float:
        """Calculate height of liquid surface in bottle"""
        fill_ratio = self.current_volume / self.total_volume
        return self.bottle_height * fill_ratio
    
    def get_spout_position(self) -> np.ndarray:
        """Get 3D position of bottle spout where wine exits"""
        # Apply rotation to get actual spout position
        # For now, simplified
        return self.position


# ============================================================================
# LIQUID STREAM PHYSICS
# ============================================================================

class LiquidStreamSimulator:
    """
    Simulate the trajectory and behavior of poured liquid
    
    Physics:
    - Liquid exits bottle as laminar flow (low Reynolds number)
    - Follows parabolic trajectory under gravity
    - Flow rate depends on: tilt angle, remaining volume, spout size
    - Stream breaks up into droplets at high speeds (we'll ignore for wine)
    """
    
    def __init__(self, fluid_props: FluidProperties = None):
        self.g = 9.81  # m/s²
        self.fluid = fluid_props or FluidProperties()
    
    def calculate_flow_rate(self, bottle: BottleState) -> float:
        """
        Calculate volumetric flow rate (m³/s)
        
        Uses Torricelli's law + discharge coefficient:
        Q = C_d * A * sqrt(2 * g * h * sin(θ))
        
        Where:
        - C_d: discharge coefficient
        - A: spout cross-sectional area
        - h: height of liquid above spout
        - θ: tilt angle from vertical
        """
        # Spout area
        A = np.pi * (bottle.spout_diameter / 2) ** 2
        
        # Effective height (depends on tilt)
        h = bottle.get_liquid_height()
        h_eff = h * np.sin(bottle.tilt_angle)
        
        # Handle edge case: nearly empty or vertical
        if h_eff < 0.001 or bottle.tilt_angle < 0.1:
            return 0.0
        
        # Torricelli's law
        v = np.sqrt(2 * self.g * h_eff)
        
        # Flow rate
        Q = self.fluid.discharge_coefficient * A * v
        
        return Q  # m³/s
    
    def simulate_stream_trajectory(
        self, 
        bottle: BottleState, 
        dt: float = 0.01,
        max_time: float = 1.0
    ) -> np.ndarray:
        """
        Simulate liquid stream trajectory from bottle to cup
        
        Returns:
            trajectory: [N, 3] array of (x, y, z) positions along stream
        """
        # Initial conditions
        spout_pos = bottle.get_spout_position()
        
        # Initial velocity (exit velocity from spout)
        flow_rate = self.calculate_flow_rate(bottle)
        spout_area = np.pi * (bottle.spout_diameter / 2) ** 2
        v_exit = flow_rate / spout_area if spout_area > 0 else 0
        
        # Velocity direction (based on bottle tilt)
        # Simplified: assume liquid exits tangent to bottle opening
        v_direction = np.array([
            np.sin(bottle.tilt_angle) * np.cos(bottle.orientation[2]),
            np.sin(bottle.tilt_angle) * np.sin(bottle.orientation[2]),
            -np.cos(bottle.tilt_angle)
        ])
        
        v_initial = v_exit * v_direction
        
        # Simulate trajectory (projectile motion)
        trajectory = []
        pos = spout_pos.copy()
        vel = v_initial.copy()
        
        t = 0
        while t < max_time and pos[2] > 0:  # Stop when hits ground
            trajectory.append(pos.copy())
            
            # Update velocity (gravity only, ignore air resistance for wine)
            vel[2] -= self.g * dt
            
            # Update position
            pos += vel * dt
            
            t += dt
        
        return np.array(trajectory)
    
    def find_landing_point(self, bottle: BottleState) -> Tuple[np.ndarray, float]:
        """
        Find where the liquid stream lands (cup surface level)
        
        Returns:
            landing_pos: [x, y, z] where stream hits
            time_of_flight: seconds until landing
        """
        trajectory = self.simulate_stream_trajectory(bottle)
        
        if len(trajectory) == 0:
            return bottle.get_spout_position(), 0.0
        
        # Find point where z ~= cup height (assume cups at z=0.45m)
        cup_height = 0.45
        
        for i, pos in enumerate(trajectory):
            if pos[2] <= cup_height:
                time_of_flight = i * 0.01  # dt = 0.01
                return pos, time_of_flight
        
        # If never reached cup height, return final position
        return trajectory[-1], len(trajectory) * 0.01
    
    def calculate_optimal_tilt(
        self, 
        bottle_pos: np.ndarray,
        cup_pos: np.ndarray,
        target_flow_rate: float = 0.00005  # 50 ml/s typical
    ) -> float:
        """
        Calculate optimal bottle tilt angle to hit the cup center
        with desired flow rate
        
        Returns:
            optimal_tilt: angle in radians
        """
        # Distance to cup
        horizontal_dist = np.linalg.norm(cup_pos[:2] - bottle_pos[:2])
        vertical_dist = bottle_pos[2] - cup_pos[2]
        
        # Use projectile motion to find angle
        # For given exit velocity v, angle θ that hits target at (x, y):
        # tan(θ) = (y + sqrt(y² + x²)) / x  (simplified)
        
        # Assume moderate flow velocity (~1 m/s)
        v = 1.0
        g = self.g
        
        # Angle to hit target (from projectile motion)
        x = horizontal_dist
        y = vertical_dist
        
        # Using trajectory equation
        discriminant = v**4 - g * (g * x**2 + 2 * y * v**2)
        
        if discriminant < 0:
            # Can't reach target at this velocity
            return np.pi / 4  # Default to 45 degrees
        
        theta = np.arctan((v**2 - np.sqrt(discriminant)) / (g * x))
        
        # Clamp to reasonable range (15° to 75° from vertical)
        theta = np.clip(theta, np.radians(15), np.radians(75))
        
        return theta


# ============================================================================
# FILL LEVEL DETECTION
# ============================================================================

class FillLevelDetector:
    """
    Detect how full the cup is using vision
    
    Methods:
    1. Color-based (detect wine surface)
    2. Edge detection (find meniscus)
    3. Volume estimation from depth
    """
    
    def __init__(self, cup_height_mm: float = 100):
        self.cup_height = cup_height_mm
        self.wine_color_lower = np.array([0, 50, 50])  # HSV lower bound
        self.wine_color_upper = np.array([15, 255, 255])  # HSV upper bound
    
    def detect_fill_level_color(self, image: np.ndarray, cup_bbox: Tuple) -> float:
        """
        Detect fill level using color segmentation
        
        Args:
            image: RGB image from overhead camera
            cup_bbox: (x, y, w, h) bounding box of cup
        
        Returns:
            fill_percentage: 0.0 to 1.0
        """
        x, y, w, h = cup_bbox
        cup_roi = image[y:y+h, x:x+w]
        
        # Convert to HSV
        hsv = cv2.cvtColor(cup_roi, cv2.COLOR_RGB2HSV)
        
        # Threshold for wine color (red/purple)
        mask = cv2.inRange(hsv, self.wine_color_lower, self.wine_color_upper)
        
        # Find highest wine pixel (lowest y-coordinate in image)
        wine_pixels = np.where(mask > 0)
        
        if len(wine_pixels[0]) == 0:
            return 0.0  # No wine detected
        
        # Wine surface is at lowest y (top of image)
        wine_surface_y = np.min(wine_pixels[0])
        
        # Calculate fill level
        cup_bottom_y = h
        fill_pixels = cup_bottom_y - wine_surface_y
        fill_percentage = fill_pixels / h
        
        return np.clip(fill_percentage, 0.0, 1.0)
    
    def detect_fill_level_depth(self, depth_image: np.ndarray, cup_bbox: Tuple) -> float:
        """
        Detect fill level using depth camera (if available)
        
        More accurate than color-based method
        """
        x, y, w, h = cup_bbox
        cup_roi = depth_image[y:y+h, x:x+w]
        
        # Find median depth in cup region
        # Wine surface will be higher (less depth) than empty cup bottom
        
        cup_depths = cup_roi[cup_roi > 0]  # Filter invalid depths
        
        if len(cup_depths) == 0:
            return 0.0
        
        # Surface depth
        surface_depth = np.percentile(cup_depths, 10)  # Top 10% = surface
        bottom_depth = np.percentile(cup_depths, 90)   # Bottom 10% = cup bottom
        
        # Calculate fill
        depth_diff = bottom_depth - surface_depth
        max_depth = self.cup_height / 1000.0  # Convert mm to m
        
        fill_percentage = depth_diff / max_depth
        
        return np.clip(fill_percentage, 0.0, 1.0)
    
    def estimate_volume(self, fill_percentage: float, cup_radius_mm: float = 35) -> float:
        """
        Estimate liquid volume in cup (ml)
        
        Assumes cylindrical cup
        """
        cup_radius_m = cup_radius_mm / 1000.0
        cup_height_m = self.cup_height / 1000.0
        
        # Volume of cylinder
        total_volume = np.pi * cup_radius_m**2 * cup_height_m
        current_volume = total_volume * fill_percentage
        
        # Convert to ml
        return current_volume * 1e6


# ============================================================================
# SPLASH PREDICTION
# ============================================================================

class SplashPredictor:
    """
    Predict if pouring will cause splashing
    
    Splashing occurs when:
    - Impact velocity too high (Weber number > critical)
    - Stream breaks up into droplets
    - Cup is too full (overflow risk)
    """
    
    def __init__(self):
        self.critical_weber = 80  # Typical for water/wine
        self.fluid = FluidProperties()
    
    def calculate_weber_number(
        self, 
        impact_velocity: float,
        droplet_diameter: float = 0.003  # 3mm typical
    ) -> float:
        """
        Weber number: We = ρ * v² * d / σ
        
        Where:
        - ρ: fluid density
        - v: impact velocity
        - d: characteristic length (droplet diameter)
        - σ: surface tension
        
        We > 80: Splashing likely
        We < 80: Smooth impact
        """
        We = (self.fluid.density * impact_velocity**2 * droplet_diameter) / \
             self.fluid.surface_tension
        
        return We
    
    def predict_splash(
        self, 
        bottle: BottleState,
        cup_fill_level: float
    ) -> Tuple[bool, float]:
        """
        Predict if current pour will cause splash
        
        Returns:
            will_splash: boolean
            splash_risk: 0.0 to 1.0 (confidence)
        """
        # Simulate stream
        simulator = LiquidStreamSimulator()
        landing_pos, time_of_flight = simulator.find_landing_point(bottle)
        
        # Calculate impact velocity
        impact_velocity = np.sqrt(2 * simulator.g * 
                                 (bottle.position[2] - landing_pos[2]))
        
        # Weber number
        We = self.calculate_weber_number(impact_velocity)
        
        # Check conditions
        will_splash = False
        risk_factors = []
        
        # Factor 1: High Weber number
        if We > self.critical_weber:
            will_splash = True
            risk_factors.append(('weber', (We - self.critical_weber) / self.critical_weber))
        
        # Factor 2: Cup too full
        if cup_fill_level > 0.85:
            will_splash = True
            risk_factors.append(('overfill', (cup_fill_level - 0.85) / 0.15))
        
        # Factor 3: High flow rate
        flow_rate = simulator.calculate_flow_rate(bottle)
        if flow_rate > 0.0001:  # 100 ml/s very fast
            risk_factors.append(('flow_rate', flow_rate / 0.0001))
        
        # Aggregate risk
        if risk_factors:
            splash_risk = np.mean([r[1] for r in risk_factors])
        else:
            splash_risk = We / self.critical_weber
        
        splash_risk = np.clip(splash_risk, 0.0, 1.0)
        
        return will_splash, splash_risk


# ============================================================================
# INTEGRATED POURING CONTROLLER
# ============================================================================

class FluidAwarePouringController:
    """
    Pouring controller that accounts for fluid dynamics
    
    This replaces simple circle overlap with physics-based control
    """
    
    def __init__(self):
        self.stream_sim = LiquidStreamSimulator()
        self.fill_detector = FillLevelDetector()
        self.splash_predictor = SplashPredictor()
        
        # Control parameters
        self.target_fill_level = 0.80  # Stop at 80% full
        self.max_flow_rate = 0.00008   # 80 ml/s max
    
    def compute_pour_command(
        self,
        bottle_state: BottleState,
        cup_position: np.ndarray,
        cup_image: np.ndarray,
        cup_bbox: Tuple
    ) -> Dict:
        """
        Compute optimal pouring action considering fluid dynamics
        
        Returns:
            {
                'tilt_angle': float,  # Desired bottle tilt
                'should_pour': bool,  # Whether to pour now
                'reason': str,        # Why pour/stop
                'landing_point': [x, y, z],  # Where stream lands
                'splash_risk': float  # 0-1
            }
        """
        # 1. Detect current fill level
        fill_level = self.fill_detector.detect_fill_level_color(
            cup_image, cup_bbox
        )
        
        # 2. Calculate where stream will land
        landing_point, tof = self.stream_sim.find_landing_point(bottle_state)
        
        # 3. Check if landing point is in cup
        cup_radius = 0.035  # 35mm
        distance_from_cup = np.linalg.norm(landing_point[:2] - cup_position[:2])
        is_aligned = distance_from_cup < cup_radius
        
        # 4. Predict splash
        will_splash, splash_risk = self.splash_predictor.predict_splash(
            bottle_state, fill_level
        )
        
        # 5. Decide action
        should_pour = False
        reason = ""
        
        if fill_level >= self.target_fill_level:
            reason = "Cup full"
        elif not is_aligned:
            reason = f"Not aligned (off by {distance_from_cup*1000:.1f}mm)"
        elif will_splash:
            reason = f"Splash risk too high ({splash_risk:.2f})"
        else:
            should_pour = True
            reason = "Pouring"
        
        # 6. Calculate optimal tilt angle
        optimal_tilt = self.stream_sim.calculate_optimal_tilt(
            bottle_state.position,
            cup_position,
            target_flow_rate=self.max_flow_rate
        )
        
        return {
            'tilt_angle': optimal_tilt,
            'should_pour': should_pour,
            'reason': reason,
            'landing_point': landing_point,
            'splash_risk': splash_risk,
            'fill_level': fill_level,
            'flow_rate': self.stream_sim.calculate_flow_rate(bottle_state),
            'is_aligned': is_aligned
        }
    
    def visualize_stream(
        self,
        image: np.ndarray,
        bottle_state: BottleState,
        camera_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Visualize predicted liquid stream on image
        Useful for debugging and visualization
        """
        # Simulate trajectory
        trajectory = self.stream_sim.simulate_stream_trajectory(bottle_state)
        
        # Project 3D trajectory to 2D image
        # (Requires camera calibration matrix)
        
        viz = image.copy()
        
        # Draw trajectory points
        for i in range(len(trajectory) - 1):
            # Project to image coordinates (simplified)
            # In production, use cv2.projectPoints with proper calibration
            pt1 = self._project_point(trajectory[i], camera_matrix)
            pt2 = self._project_point(trajectory[i + 1], camera_matrix)
            
            # Color gradient: green → yellow → red (based on velocity)
            color_ratio = i / len(trajectory)
            color = (
                int(255 * (1 - color_ratio)),  # R
                int(255 * color_ratio),         # G
                0                               # B
            )
            
            cv2.line(viz, tuple(pt1), tuple(pt2), color, 2)
        
        return viz
    
    def _project_point(self, point_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D image (simplified)"""
        # Simplified projection - use cv2.projectPoints in production
        # For overhead camera looking down
        x_img = int((point_3d[0] + 0.5) * 640)
        y_img = int((0.5 - point_3d[1]) * 480)
        return np.array([x_img, y_img])


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("WINE POURING FLUID DYNAMICS SIMULATION")
    print("="*70)
    
    # Create bottle state
    bottle = BottleState(
        position=np.array([0.5, 0.0, 0.7]),  # 70cm above table
        orientation=np.array([0, 0, 0]),
        tilt_angle=np.radians(45),  # 45 degree tilt
        current_volume=0.500  # Half full bottle
    )
    
    # Simulate stream
    print("\n1. Simulating liquid stream...")
    simulator = LiquidStreamSimulator()
    
    flow_rate = simulator.calculate_flow_rate(bottle)
    print(f"   Flow rate: {flow_rate * 1e6:.1f} ml/s")
    
    landing_point, tof = simulator.find_landing_point(bottle)
    print(f"   Landing point: {landing_point}")
    print(f"   Time of flight: {tof:.3f}s")
    
    # Optimal tilt
    cup_pos = np.array([0.5, 0.2, 0.45])
    optimal_tilt = simulator.calculate_optimal_tilt(
        bottle.position, cup_pos
    )
    print(f"   Optimal tilt: {np.degrees(optimal_tilt):.1f}°")
    
    # Splash prediction
    print("\n2. Predicting splash...")
    splash_pred = SplashPredictor()
    will_splash, risk = splash_pred.predict_splash(bottle, fill_level=0.5)
    print(f"   Will splash: {will_splash}")
    print(f"   Splash risk: {risk:.2f}")
    
    # Integrated controller
    print("\n3. Running fluid-aware controller...")
    controller = FluidAwarePouringController()
    
    # Dummy image for demo
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_bbox = (300, 200, 70, 100)
    
    command = controller.compute_pour_command(
        bottle, cup_pos, dummy_image, dummy_bbox
    )
    
    print(f"   Decision: {command['reason']}")
    print(f"   Should pour: {command['should_pour']}")
    print(f"   Fill level: {command['fill_level']:.1%}")
    print(f"   Tilt angle: {np.degrees(command['tilt_angle']):.1f}°")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("""
    - Wine doesn't go straight down - it follows a parabolic path!
    - Flow rate depends on bottle tilt AND how full it is
    - Need to check fill level to know when to stop
    - Splash happens when Weber number > 80
    - Optimal pour angle is ~30-45° from vertical
    
    This is WHY you need fluid dynamics - the simple circle IoU
    assumes the pour point and landing point are the same, but 
    they're not! The stream has trajectory!
    """)