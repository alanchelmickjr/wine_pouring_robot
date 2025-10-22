"""
WINE POURING VISUALIZATION - NO PYBULLET REQUIRED

Uses only matplotlib for visualization
Perfect if PyBullet won't install!

Shows:
- 3D trajectory plot
- Top-down IoU view
- Animated pour sequence
- All physics calculations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import time


class SimplifiedWineSimulation:
    """Wine pouring simulation without PyBullet"""
    
    def __init__(self):
        self.g = 9.81
        
        # Bottle state
        self.bottle_pos = np.array([0.5, 0.0, 0.7])
        self.bottle_tilt_angle = np.radians(35)  # How much to tilt
        self.bottle_direction = np.array([0, 1, 0])  # Which way to aim (toward cup!)
        self.is_pouring = False
        self.auto_aim = True  # AUTO-AIM ON BY DEFAULT!
        
        # Cup state
        self.cup_pos = np.array([0.5, 0.2, 0.45])
        self.cup_radius = 0.035
        self.fill_level = 0.0
        
        # Particles (simplified)
        self.particles = []
        self.max_particles = 100
    
    def calculate_flow_rate(self):
        """Calculate flow rate based on tilt"""
        if self.bottle_tilt_angle < np.radians(15):
            return 0.0
        
        # Simplified Torricelli
        h = 0.2 * np.sin(self.bottle_tilt_angle)
        A = np.pi * (0.015 / 2) ** 2
        v = np.sqrt(2 * self.g * h)
        Q = 0.6 * A * v  # discharge coefficient
        
        return Q
    
    def simulate_trajectory(self):
        """Calculate stream trajectory - FROM bottle TO cup!"""
        # Calculate direction from bottle to cup
        to_cup = self.cup_pos - self.bottle_pos
        horizontal_dir = to_cup[:2] / (np.linalg.norm(to_cup[:2]) + 1e-6)
        
        # Spout position (bottle opening)
        bottle_length = 0.15
        spout_offset = np.array([
            horizontal_dir[0] * bottle_length * np.sin(self.bottle_tilt_angle),
            horizontal_dir[1] * bottle_length * np.sin(self.bottle_tilt_angle),
            -bottle_length * np.cos(self.bottle_tilt_angle)
        ])
        spout_pos = self.bottle_pos + spout_offset
        
        # Initial velocity - aim TOWARD the cup!
        flow_rate = self.calculate_flow_rate()
        if flow_rate == 0:
            return np.array([spout_pos])
        
        spout_area = np.pi * (0.015 / 2) ** 2
        v_exit = flow_rate / spout_area
        
        # Velocity direction: tilt toward cup
        v_dir = np.array([
            horizontal_dir[0] * np.sin(self.bottle_tilt_angle),
            horizontal_dir[1] * np.sin(self.bottle_tilt_angle),
            -np.cos(self.bottle_tilt_angle)
        ])
        v_dir = v_dir / (np.linalg.norm(v_dir) + 1e-6)
        
        velocity = v_exit * v_dir
        
        # Simulate projectile motion
        trajectory = []
        pos = spout_pos.copy()
        vel = velocity.copy()
        dt = 0.01
        
        for _ in range(100):
            trajectory.append(pos.copy())
            vel[2] -= self.g * dt
            pos += vel * dt
            
            if pos[2] <= self.cup_pos[2]:
                break
        
        return np.array(trajectory)
    
    def find_landing_point(self):
        """Where does stream land?"""
        traj = self.simulate_trajectory()
        if len(traj) == 0:
            return self.bottle_pos
        return traj[-1]
    
    def calculate_iou(self):
        """IoU between landing point and cup"""
        landing = self.find_landing_point()
        
        # Distance between centers
        d = np.linalg.norm(landing[:2] - self.cup_pos[:2])
        r = self.cup_radius
        
        if d >= 2 * r:
            return 0.0
        if d <= 0:
            return 1.0
        
        # Approximate IoU
        return max(0, 1 - d / (2 * r))
    
    def calculate_optimal_tilt(self):
        """
        Calculate optimal tilt angle to hit the cup!
        Returns the tilt angle that makes the stream land in the cup.
        """
        # 3D distance to cup
        to_cup = self.cup_pos - self.bottle_pos
        horizontal_dist = np.linalg.norm(to_cup[:2])
        vertical_dist = to_cup[2]  # Negative since cup is below
        
        # Edge case: cup directly below
        if horizontal_dist < 0.01:
            return np.radians(20)  # Small tilt for straight down
        
        # Use projectile motion to find optimal angle
        # For a given velocity v, find angle θ that hits target
        v = 1.2  # Exit velocity (~1-1.5 m/s typical)
        g = self.g
        x = horizontal_dist
        y = -vertical_dist  # Make positive (falling down)
        
        # Projectile equation: tan(θ) = (v² ± √(v⁴ - g(gx² + 2yv²))) / (gx)
        v2 = v * v
        v4 = v2 * v2
        discriminant = v4 - g * (g * x * x - 2 * y * v2)
        
        if discriminant < 0:
            # Can't reach with this velocity - use max reasonable tilt
            return np.radians(45)
        
        # Two solutions - use the lower angle (shorter arc)
        sqrt_disc = np.sqrt(discriminant)
        angle1 = np.arctan((v2 - sqrt_disc) / (g * x))
        
        # Clamp to safe range
        angle = np.clip(angle1, np.radians(15), np.radians(65))
        
        return angle
    
    def update(self, dt=0.1):
        """Update simulation"""
        # AUTO-AIM: Calculate optimal tilt
        if self.auto_aim:
            self.bottle_tilt_angle = self.calculate_optimal_tilt()
        
        # Pour logic
        if self.is_pouring:
            iou = self.calculate_iou()
            if iou > 0.5:
                self.fill_level += 0.02 * dt
                self.fill_level = min(self.fill_level, 1.0)


class InteractiveVisualization:
    """Interactive matplotlib visualization"""
    
    def __init__(self):
        self.sim = SimplifiedWineSimulation()
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory
        self.ax_3d = self.fig.add_subplot(2, 2, 1, projection='3d')
        
        # Top-down view
        self.ax_top = self.fig.add_subplot(2, 2, 2)
        
        # Metrics
        self.ax_metrics = self.fig.add_subplot(2, 2, 3)
        
        # Controls info
        self.ax_info = self.fig.add_subplot(2, 2, 4)
        
        # Interactive controls
        self.setup_controls()
        
        # Disable matplotlib's default keybindings that conflict
        plt.rcParams['keymap.save'].remove('s')
        plt.rcParams['keymap.quit'].remove('q')
        plt.rcParams['keymap.fullscreen'] = []
        plt.rcParams['keymap.home'] = []
        plt.rcParams['keymap.back'] = []
        plt.rcParams['keymap.forward'] = []
        
        print("\n" + "="*60)
        print("WINE POURING VISUALIZATION (Matplotlib)")
        print("="*60)
        print("\n🎯 AUTO-AIM ENABLED (robot calculates perfect tilt!)")
        print("\nControls:")
        print("  UP/DOWN arrows - Manual tilt (disables auto-aim)")
        print("  A              - Toggle auto-aim")
        print("  ENTER/SPACE    - Toggle pouring")
        print("  R              - Reset")
        print("  ESC            - Quit")
        print("="*60 + "\n")
    
    def setup_controls(self):
        """Setup keyboard controls"""
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    
    def on_key(self, event):
        """Handle keyboard input"""
        if event.key == 'up':
            self.sim.auto_aim = False  # Manual control disables auto-aim
            self.sim.bottle_tilt_angle = min(self.sim.bottle_tilt_angle + np.radians(5), np.radians(75))
            print(f"Manual: Tilt {np.degrees(self.sim.bottle_tilt_angle):.1f}° (auto-aim OFF)")
        
        elif event.key == 'down':
            self.sim.auto_aim = False  # Manual control disables auto-aim
            self.sim.bottle_tilt_angle = max(self.sim.bottle_tilt_angle - np.radians(5), 0)
            print(f"Manual: Tilt {np.degrees(self.sim.bottle_tilt_angle):.1f}° (auto-aim OFF)")
        
        elif event.key == 'a':
            self.sim.auto_aim = not self.sim.auto_aim
            status = "ON 🎯" if self.sim.auto_aim else "OFF"
            print(f"Auto-aim: {status}")
        
        elif event.key in [' ', 'enter']:
            self.sim.is_pouring = not self.sim.is_pouring
            status = "POURING" if self.sim.is_pouring else "STOPPED"
            print(f"Status: {status}")
        
        elif event.key == 'r':
            self.sim.fill_level = 0.0
            self.sim.bottle_tilt_angle = np.radians(35)
            self.sim.is_pouring = False
            self.sim.auto_aim = True  # Reset to auto-aim
            print("Reset! (auto-aim ON)")
        
        elif event.key == 'escape':
            print("Quitting...")
            plt.close()
    
    def draw_3d_view(self):
        """Draw 3D trajectory view"""
        self.ax_3d.clear()
        
        # Trajectory
        traj = self.sim.simulate_trajectory()
        if len(traj) > 0:
            self.ax_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                           'b-', linewidth=3, label='Stream')
        
        # Bottle
        self.ax_3d.scatter(*self.sim.bottle_pos, c='green', s=300, 
                          marker='o', label='Bottle')
        
        # Cup
        self.ax_3d.scatter(*self.sim.cup_pos, c='red', s=300,
                          marker='^', label='Cup')
        
        # Landing point
        landing = self.sim.find_landing_point()
        self.ax_3d.scatter(*landing, c='yellow', s=200, 
                          marker='x', label='Landing')
        
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Stream Trajectory')
        self.ax_3d.legend()
        self.ax_3d.set_xlim(0.2, 0.8)
        self.ax_3d.set_ylim(-0.2, 0.4)
        self.ax_3d.set_zlim(0, 1)
    
    def draw_top_view(self):
        """Draw top-down IoU view"""
        self.ax_top.clear()
        
        # Trajectory
        traj = self.sim.simulate_trajectory()
        if len(traj) > 0:
            self.ax_top.plot(traj[:, 0], traj[:, 1], 'b-', 
                           linewidth=2, label='Stream')
        
        # Cup circle (green)
        cup_circle = Circle(
            (self.sim.cup_pos[0], self.sim.cup_pos[1]),
            self.sim.cup_radius,
            fill=False, color='green', linewidth=3, label='Cup'
        )
        self.ax_top.add_patch(cup_circle)
        
        # Landing circle (yellow)
        landing = self.sim.find_landing_point()
        landing_circle = Circle(
            (landing[0], landing[1]),
            self.sim.cup_radius,
            fill=False, color='yellow', linewidth=2, 
            linestyle='--', label='Landing'
        )
        self.ax_top.add_patch(landing_circle)
        
        # IoU overlap region (red if bad, green if good)
        iou = self.sim.calculate_iou()
        overlap_color = 'green' if iou > 0.6 else 'red'
        if iou > 0:
            overlap_circle = Circle(
                (landing[0], landing[1]),
                self.sim.cup_radius * iou,
                fill=True, color=overlap_color, alpha=0.3
            )
            self.ax_top.add_patch(overlap_circle)
        
        self.ax_top.set_xlabel('X (m)')
        self.ax_top.set_ylabel('Y (m)')
        self.ax_top.set_title(f'Top View - IoU: {iou:.2f}')
        self.ax_top.set_aspect('equal')
        self.ax_top.legend()
        self.ax_top.grid(True, alpha=0.3)
        self.ax_top.set_xlim(0.3, 0.7)
        self.ax_top.set_ylim(-0.1, 0.5)
    
    def draw_metrics(self):
        """Draw metrics panel"""
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        
        # Calculate metrics
        iou = self.sim.calculate_iou()
        flow_rate = self.sim.calculate_flow_rate()
        landing = self.sim.find_landing_point()
        distance = np.linalg.norm(landing[:2] - self.sim.cup_pos[:2])
        
        # Status color
        if iou > 0.8:
            status_color = 'green'
            status = "EXCELLENT"
        elif iou > 0.6:
            status_color = 'yellow'
            status = "GOOD"
        else:
            status_color = 'red'
            status = "POOR"
        
        # Display metrics
        auto_aim_status = "🎯 ON" if self.sim.auto_aim else "OFF"
        
        metrics_text = f"""
METRICS
{'='*30}

Auto-Aim:       {auto_aim_status}
Bottle Tilt:    {np.degrees(self.sim.bottle_tilt_angle):.1f}°
Flow Rate:      {flow_rate*1e6:.1f} ml/s

IoU:            {iou:.2f}
Fill Level:     {self.sim.fill_level:.1%}
Distance:       {distance*1000:.1f} mm

Status:         {status}
Pouring:        {'YES' if self.sim.is_pouring else 'NO'}
        """
        
        self.ax_metrics.text(0.1, 0.5, metrics_text,
                            fontsize=12, family='monospace',
                            verticalalignment='center')
        
        # Status indicator
        self.ax_metrics.add_patch(Circle((0.85, 0.8), 0.08, 
                                        color=status_color, alpha=0.5))
    
    def draw_info(self):
        """Draw controls info"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = """
CONTROLS
{'='*30}

A           - Toggle auto-aim
↑/↓         - Manual tilt
SPACE/ENTER - Pour on/off
R           - Reset
ESC         - Quit

AUTO-AIM 🎯
{'='*30}

Robot calculates perfect
tilt angle to hit cup!

Uses projectile motion:
tan(θ) = (v² ± √Δ) / gx

PHYSICS
{'='*30}

✓ Torricelli's law
✓ Parabolic trajectory
✓ IoU calculation
✓ Real-time auto-aim
        """
        
        self.ax_info.text(0.1, 0.5, info_text,
                         fontsize=10, family='monospace',
                         verticalalignment='center')
    
    def animate(self, frame):
        """Animation update"""
        self.sim.update(dt=0.1)
        
        self.draw_3d_view()
        self.draw_top_view()
        self.draw_metrics()
        self.draw_info()
    
    def run(self):
        """Run visualization"""
        ani = FuncAnimation(
            self.fig, self.animate,
            interval=100,  # 10 FPS
            blit=False
        )
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("WINE POURING VISUALIZATION")
    print("(Matplotlib - No PyBullet Required!)")
    print("="*60)
    
    viz = InteractiveVisualization()
    viz.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVisualization stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have matplotlib installed:")
        print("  pip install matplotlib numpy")
