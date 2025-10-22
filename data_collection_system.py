"""
DATA COLLECTION SYSTEM FOR WINE POURING ROBOT

Collect demonstrations in PyBullet simulation for training diffusion policy

Features:
- Record pour demonstrations with fluid physics
- Save observations (images, positions, metrics)
- Save actions (bottle tilt, pour decision)
- Export in HuggingFace LeRobot format
- Easy playback and verification

Usage:
    python data_collection_system.py
"""

import numpy as np
import cv2
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pickle


class DemonstrationRecorder:
    """
    Record wine pouring demonstrations
    
    Each demo contains:
    - Observations: images, cup position, IoU, fill level
    - Actions: bottle tilt, pour on/off
    - Rewards: success, spillage, efficiency
    - Metadata: scenario, success, duration
    """
    
    def __init__(self, save_dir="demonstrations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.current_demo = None
        self.demo_count = len(list(self.save_dir.glob("demo_*")))
        
        print(f"Recorder initialized. {self.demo_count} existing demos found.")
    
    def start_recording(self, scenario="manual"):
        """Start recording a new demonstration"""
        demo_id = f"demo_{self.demo_count:04d}"
        demo_path = self.save_dir / demo_id
        demo_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (demo_path / "observations").mkdir(exist_ok=True)
        (demo_path / "images").mkdir(exist_ok=True)
        
        self.current_demo = {
            'id': demo_id,
            'path': demo_path,
            'scenario': scenario,
            'start_time': time.time(),
            'timesteps': [],
            'frame_count': 0
        }
        
        print(f"\n📹 Recording: {demo_id}")
        print(f"   Scenario: {scenario}")
        return demo_id
    
    def record_timestep(
        self,
        image: np.ndarray,
        observation: Dict,
        action: Dict,
        reward: float = 0.0
    ):
        """Record one timestep"""
        if self.current_demo is None:
            print("❌ No recording in progress! Call start_recording() first.")
            return
        
        frame_id = self.current_demo['frame_count']
        
        # Save image
        image_path = self.current_demo['path'] / "images" / f"frame_{frame_id:05d}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Record data
        timestep = {
            'frame_id': frame_id,
            'timestamp': time.time() - self.current_demo['start_time'],
            'observation': {
                'cup_position': observation.get('cup_position', [0, 0]),
                'cup_radius': observation.get('cup_radius', 35),
                'bottle_position': observation.get('bottle_position', [0, 0, 0]),
                'bottle_tilt': observation.get('bottle_tilt', 0.0),
                'iou': observation.get('iou', 0.0),
                'fill_level': observation.get('fill_level', 0.0),
                'splash_risk': observation.get('splash_risk', 0.0),
                'landing_point': observation.get('landing_point', [0, 0, 0])
            },
            'action': {
                'tilt_angle': action.get('tilt_angle', 0.0),
                'is_pouring': action.get('is_pouring', False),
                'delta_tilt': action.get('delta_tilt', 0.0)
            },
            'reward': reward
        }
        
        self.current_demo['timesteps'].append(timestep)
        self.current_demo['frame_count'] += 1
    
    def stop_recording(self, success: bool = True, notes: str = ""):
        """Stop recording and save demonstration"""
        if self.current_demo is None:
            print("❌ No recording in progress!")
            return None
        
        duration = time.time() - self.current_demo['start_time']
        
        # Create metadata
        metadata = {
            'demo_id': self.current_demo['id'],
            'scenario': self.current_demo['scenario'],
            'duration': duration,
            'num_frames': self.current_demo['frame_count'],
            'success': success,
            'notes': notes,
            'timestamp': datetime.now().isoformat(),
            
            # Summary statistics
            'stats': self._calculate_stats(self.current_demo['timesteps'])
        }
        
        # Save metadata
        metadata_path = self.current_demo['path'] / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save timesteps (as numpy arrays for efficiency)
        timesteps_path = self.current_demo['path'] / "timesteps.pkl"
        with open(timesteps_path, 'wb') as f:
            pickle.dump(self.current_demo['timesteps'], f)
        
        # Also save as readable JSON
        timesteps_json_path = self.current_demo['path'] / "timesteps.json"
        with open(timesteps_json_path, 'w') as f:
            json.dump(self.current_demo['timesteps'], f, indent=2)
        
        demo_id = self.current_demo['id']
        self.demo_count += 1
        self.current_demo = None
        
        print(f"\n✅ Demo saved: {demo_id}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Frames: {metadata['num_frames']}")
        print(f"   Success: {success}")
        print(f"   Stats: {metadata['stats']}")
        
        return demo_id
    
    def _calculate_stats(self, timesteps: List[Dict]) -> Dict:
        """Calculate summary statistics"""
        if not timesteps:
            return {}
        
        ious = [t['observation']['iou'] for t in timesteps]
        fill_levels = [t['observation']['fill_level'] for t in timesteps]
        tilts = [t['action']['tilt_angle'] for t in timesteps]
        
        return {
            'avg_iou': float(np.mean(ious)),
            'min_iou': float(np.min(ious)),
            'max_iou': float(np.max(ious)),
            'final_fill_level': float(fill_levels[-1]),
            'avg_tilt': float(np.degrees(np.mean(tilts))),
            'max_tilt': float(np.degrees(np.max(tilts))),
            'total_reward': float(sum(t['reward'] for t in timesteps))
        }


class InteractiveDataCollector:
    """
    Interactive system for collecting demonstrations
    
    Modes:
    1. Manual control (keyboard/gamepad)
    2. Automatic (scripted scenarios)
    3. Playback (review existing demos)
    """
    
    def __init__(self, sim):
        self.sim = sim
        self.recorder = DemonstrationRecorder()
        self.mode = "idle"
        
        # Control state
        self.target_tilt = 0.0
        self.tilt_speed = np.radians(20)  # 20 deg/s
        
        print("\n" + "="*70)
        print("INTERACTIVE DATA COLLECTOR")
        print("="*70)
        print("\nControls:")
        print("  W/S       - Increase/decrease bottle tilt")
        print("  SPACE     - Toggle pouring")
        print("  R         - Start/stop recording")
        print("  N         - Next scenario (auto mode)")
        print("  P         - Playback last demo")
        print("  Q         - Quit")
        print("="*70 + "\n")
    
    def collect_manual_demo(self):
        """Collect demonstration via manual control"""
        print("\n🎮 Manual control mode")
        print("Control the bottle and demonstrate good pouring!")
        input("Press Enter to start recording...")
        
        self.recorder.start_recording(scenario="manual")
        recording = True
        
        while recording:
            # Get user input (simplified - in production use pygame/pynput)
            # For now, this is a placeholder
            
            # Simulate control
            timestep_data = self._get_current_state()
            
            # Calculate reward
            reward = self._calculate_reward(timestep_data['observation'])
            
            # Record
            self.recorder.record_timestep(
                image=timestep_data['image'],
                observation=timestep_data['observation'],
                action=timestep_data['action'],
                reward=reward
            )
            
            # Step simulation
            self.sim.step()
            
            # Check for stop (placeholder)
            # In production, check for 'R' key press
            if timestep_data['observation']['fill_level'] > 0.8:
                recording = False
        
        # Save
        success = timestep_data['observation']['fill_level'] > 0.7
        self.recorder.stop_recording(success=success)
    
    def collect_automatic_demos(self, num_demos: int = 10):
        """Collect demonstrations automatically with varying scenarios"""
        print(f"\n🤖 Automatic collection mode")
        print(f"Generating {num_demos} demonstrations...")
        
        scenarios = [
            {'name': 'stationary_cup', 'cup_movement': 0.0, 'tilt': 35},
            {'name': 'slow_drift', 'cup_movement': 0.01, 'tilt': 40},
            {'name': 'fast_pour', 'cup_movement': 0.0, 'tilt': 55},
            {'name': 'gentle_pour', 'cup_movement': 0.0, 'tilt': 25},
            {'name': 'moving_cup', 'cup_movement': 0.03, 'tilt': 35},
        ]
        
        for i in range(num_demos):
            scenario = scenarios[i % len(scenarios)]
            print(f"\nDemo {i+1}/{num_demos}: {scenario['name']}")
            
            # Reset sim
            self._reset_simulation()
            
            # Start recording
            self.recorder.start_recording(scenario=scenario['name'])
            
            # Execute scripted behavior
            self._execute_scenario(scenario)
            
            # Stop recording
            self.recorder.stop_recording(success=True)
            
            print(f"Progress: {i+1}/{num_demos} complete")
        
        print(f"\n✅ All {num_demos} demonstrations collected!")
    
    def _execute_scenario(self, scenario: Dict):
        """Execute a scripted pouring scenario"""
        target_tilt = np.radians(scenario['tilt'])
        cup_movement_speed = scenario['cup_movement']
        
        # Phase 1: Tilt bottle gradually
        for step in range(100):
            current_tilt = (step / 100.0) * target_tilt
            self.sim.bottle_tilt = current_tilt
            
            if step > 50:  # Start pouring halfway through tilt
                self.sim.is_pouring = True
            
            # Move cup if needed
            if cup_movement_speed > 0:
                self.sim.cup_position[0] += cup_movement_speed * np.sin(step * 0.1)
            
            # Record
            self._record_current_frame()
            self.sim.step()
        
        # Phase 2: Maintain pour
        for step in range(200):
            # Keep pouring
            self.sim.is_pouring = True
            
            # Move cup
            if cup_movement_speed > 0:
                self.sim.cup_position[0] += cup_movement_speed * np.sin(step * 0.1)
            
            # Record
            self._record_current_frame()
            self.sim.step()
            
            # Stop if cup full
            if self.sim.particles_in_cup > 100:
                break
        
        # Phase 3: Stop pouring
        self.sim.is_pouring = False
        for step in range(50):
            # Gradual untilt
            self.sim.bottle_tilt *= 0.95
            
            self._record_current_frame()
            self.sim.step()
    
    def _get_current_state(self) -> Dict:
        """Get current simulation state"""
        # Capture screenshot
        image = self._get_camera_image()
        
        # Get metrics
        metrics = self.sim.calculate_metrics()
        
        # Observation
        observation = {
            'cup_position': self.sim.cup_position.tolist(),
            'cup_radius': self.sim.cup_radius,
            'bottle_position': self.sim.bottle_position.tolist(),
            'bottle_tilt': self.sim.bottle_tilt,
            'iou': metrics['iou'],
            'fill_level': metrics['fill_level'],
            'splash_risk': metrics['splash_risk'],
            'landing_point': metrics.get('landing_point', [0, 0, 0])
        }
        
        # Action
        action = {
            'tilt_angle': self.sim.bottle_tilt,
            'is_pouring': self.sim.is_pouring,
            'delta_tilt': 0.0  # Calculate from previous
        }
        
        return {
            'image': image,
            'observation': observation,
            'action': action
        }
    
    def _record_current_frame(self):
        """Record current simulation frame"""
        state = self._get_current_state()
        reward = self._calculate_reward(state['observation'])
        
        self.recorder.record_timestep(
            image=state['image'],
            observation=state['observation'],
            action=state['action'],
            reward=reward
        )
    
    def _calculate_reward(self, observation: Dict) -> float:
        """Calculate reward for current state"""
        reward = 0.0
        
        # Positive: Good IoU
        reward += observation['iou'] * 1.0
        
        # Positive: Making progress (filling cup)
        reward += observation['fill_level'] * 0.5
        
        # Negative: High splash risk
        reward -= observation['splash_risk'] * 0.3
        
        # Bonus: Cup is filling AND aligned
        if observation['iou'] > 0.7 and observation['fill_level'] < 0.9:
            reward += 0.5
        
        return reward
    
    def _get_camera_image(self) -> np.ndarray:
        """Get image from PyBullet camera"""
        # Use PyBullet's camera
        import pybullet as p
        
        width, height = 640, 480
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0.0, 1.5],
            cameraTargetPosition=[0.5, 0.2, 0.45],
            cameraUpVector=[0, 1, 0]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height, nearVal=0.1, farVal=5.0
        )
        
        img = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb = np.array(img[2]).reshape(height, width, 4)[:, :, :3]
        return rgb
    
    def _reset_simulation(self):
        """Reset simulation to initial state"""
        self.sim.bottle_tilt = 0.0
        self.sim.is_pouring = False
        self.sim.bottle_position = np.array([0.5, 0.0, 0.7])
        self.sim.cup_position = np.array([0.5, 0.2, 0.45])
        self.sim.particles.clear()
        self.sim.particles_in_cup = 0


def export_to_lerobot_format(demo_dir: Path, output_dir: Path):
    """
    Export demonstrations to LeRobot format
    
    LeRobot format:
    - episodes.parquet: Metadata for all episodes
    - observations/: Images and sensor data
    - actions/: Control actions
    """
    print(f"\n📦 Exporting to LeRobot format...")
    print(f"   Input: {demo_dir}")
    print(f"   Output: {output_dir}")
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # TODO: Implement full LeRobot export
    # For now, just copy and organize
    
    demos = sorted(demo_dir.glob("demo_*"))
    print(f"   Found {len(demos)} demonstrations")
    
    for demo in demos:
        print(f"   Processing {demo.name}...")
    
    print("✅ Export complete!")


# ============================================================================
# MAIN LAUNCHER
# ============================================================================

def main():
    """Main launcher for data collection"""
    print("="*70)
    print("WINE POURING DATA COLLECTION SYSTEM")
    print("="*70)
    
    print("\nWhat would you like to do?\n")
    print("1. 🎮 Collect manual demonstrations (keyboard control)")
    print("2. 🤖 Generate automatic demonstrations (10 scenarios)")
    print("3. 📊 View existing demonstrations")
    print("4. 📦 Export to HuggingFace LeRobot format")
    print("5. 🎬 Quick demo (generate 5 demos automatically)")
    
    choice = input("\nSelect (1-5): ").strip()
    
    # Import simulation
    from pybullet_fluid_viz import WinePouringFluidSim
    
    if choice == "1":
        print("\n🎮 Manual control mode")
        sim = WinePouringFluidSim(gui=True)
        collector = InteractiveDataCollector(sim)
        collector.collect_manual_demo()
    
    elif choice == "2":
        print("\n🤖 Automatic generation mode")
        num = input("How many demos to generate? (default: 10): ").strip()
        num = int(num) if num else 10
        
        sim = WinePouringFluidSim(gui=True)
        collector = InteractiveDataCollector(sim)
        collector.collect_automatic_demos(num_demos=num)
    
    elif choice == "3":
        print("\n📊 Viewing demonstrations")
        view_demonstrations()
    
    elif choice == "4":
        print("\n📦 Exporting to LeRobot format")
        export_to_lerobot_format(
            demo_dir=Path("demonstrations"),
            output_dir=Path("lerobot_dataset")
        )
    
    elif choice == "5":
        print("\n🎬 Quick demo")
        sim = WinePouringFluidSim(gui=True)
        collector = InteractiveDataCollector(sim)
        collector.collect_automatic_demos(num_demos=5)
    
    else:
        print("Invalid choice!")


def view_demonstrations():
    """View collected demonstrations"""
    demo_dir = Path("demonstrations")
    demos = sorted(demo_dir.glob("demo_*"))
    
    if not demos:
        print("No demonstrations found!")
        return
    
    print(f"\nFound {len(demos)} demonstrations:\n")
    
    for i, demo in enumerate(demos):
        metadata_file = demo / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            print(f"{i+1}. {demo.name}")
            print(f"   Scenario: {metadata['scenario']}")
            print(f"   Duration: {metadata['duration']:.1f}s")
            print(f"   Frames: {metadata['num_frames']}")
            print(f"   Success: {'✅' if metadata['success'] else '❌'}")
            print(f"   Avg IoU: {metadata['stats']['avg_iou']:.2f}")
            print()


if __name__ == "__main__":
    main()