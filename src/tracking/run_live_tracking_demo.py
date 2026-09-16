"""
High-Quality Live Pygame Visualization Suite for Dynamic Target Tracking
=========================================================================
Renders a custom, high-resolution Pygame UI for live tracking demonstrations:
  - 25x25 grid with sleek dark-mode aesthetics
  - Dedicated Base Station (0,0) with charging station indicator
  - Dynamic Target with pulsing tracking radar ring
  - Individual Drone markers with real-time battery gauge bars above each drone
  - Live HUD displaying Step Counter, Battery Health, Target Distance & Status
  - Clean Pygame event loop with smooth exit handling

Usage:
  python src/tracking/run_live_tracking_demo.py --model rspo_selfheal
  python src/tracking/run_live_tracking_demo.py --model qmix_selfheal
"""

import os
import sys
import time
import argparse
import numpy as np
import torch as th
import pygame

# Ensure src directory is on sys.path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPYMARL_SRC = os.path.join(SRC_DIR, "epymarl", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EPYMARL_SRC not in sys.path:
    sys.path.insert(0, EPYMARL_SRC)

from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
from tracking.tracking_drift_wrapper import RandomDriftWrapper
from tracking.tracking_reward_wrapper import TrackingRewardWrapper

from tracking.evaluate_tracking_models import (
    load_rllib_policy,
    load_epymarl_policy,
    find_best_post_10m_checkpoint,
    RSPOModelV2,
    MAPPOModelVanilla,
)


class CustomTrackingPygameRenderer:
    """Sleek Pygame renderer for Dynamic Target Tracking demo."""
    
    # Palette
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_BASE = (0, 180, 216)
    COLOR_TARGET = (255, 75, 75)
    COLOR_TARGET_RADAR = (255, 75, 75, 60)
    COLOR_TEXT = (240, 240, 245)
    COLOR_PANEL_BG = (25, 32, 48)
    COLOR_ACCENT = (0, 230, 180)
    
    DRONE_COLORS = [
        (0, 210, 255),    # Drone 1: Cyan
        (255, 180, 0),    # Drone 2: Amber
        (180, 90, 255),   # Drone 3: Purple
        (50, 220, 100),   # Drone 4: Green
    ]

    def __init__(self, grid_size=25, cell_size=28, hud_height=90):
        pygame.init()
        pygame.font.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.hud_height = hud_height
        
        self.grid_width = grid_size * cell_size
        self.grid_height = grid_size * cell_size
        self.window_width = self.grid_width
        self.window_height = self.grid_height + hud_height
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("DSSE — Dynamic Target Tracking MARL Demo")
        
        self.font_large = pygame.font.SysFont("Helvetica", 20, bold=True)
        self.font_medium = pygame.font.SysFont("Helvetica", 14, bold=True)
        self.font_small = pygame.font.SysFont("Helvetica", 11)
        self.clock = pygame.time.Clock()

    def process_events(self):
        """Handle Pygame window events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def render(self, env_unwrapped, battery_wrapper, model_name, step, target_found):
        """Renders the entire game state to screen."""
        self.screen.fill(self.COLOR_BG)
        
        # 1. Draw Grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    self.hud_height + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # 2. Draw Base Station at (0,0)
        base_rect = pygame.Rect(
            0, self.hud_height, self.cell_size * 2, self.cell_size * 2
        )
        pygame.draw.rect(self.screen, (0, 70, 110), base_rect)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, 2)
        base_txt = self.font_small.render("BASE 🔋", True, self.COLOR_BASE)
        self.screen.blit(base_txt, (base_rect.x + 4, base_rect.y + 4))

        # 3. Draw Moving Target
        target_pos = None
        if hasattr(env_unwrapped, "person_position"):
            target_pos = env_unwrapped.person_position
        elif hasattr(env_unwrapped, "person_pos"):
            target_pos = env_unwrapped.person_pos

        if target_pos is not None:
            ty, tx = int(target_pos[0]), int(target_pos[1])
            cx = tx * self.cell_size + self.cell_size // 2
            cy = self.hud_height + ty * self.cell_size + self.cell_size // 2
            
            # Draw radar pulse
            pulse_radius = int(self.cell_size * 1.6)
            s = pygame.Surface((pulse_radius * 2, pulse_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 75, 75, 80), (pulse_radius, pulse_radius), pulse_radius)
            self.screen.blit(s, (cx - pulse_radius, cy - pulse_radius))
            
            # Draw target core
            pygame.draw.circle(self.screen, self.COLOR_TARGET, (cx, cy), self.cell_size // 3)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), self.cell_size // 6)

        # 4. Draw Drones & Battery Gauges
        drone_positions = getattr(env_unwrapped, "drone_positions", {})
        if hasattr(env_unwrapped, "get_drone_positions"):
            drone_positions = env_unwrapped.get_drone_positions()

        alive_count = 0
        batt_levels = []

        for idx, agent_name in enumerate(["drone0", "drone1", "drone2", "drone3"]):
            is_alive = battery_wrapper.alive.get(agent_name, True)
            batt = battery_wrapper.battery.get(agent_name, 125)
            batt_pct = max(0.0, min(100.0, (batt / 125.0) * 100.0))
            batt_levels.append(batt_pct)

            if agent_name in drone_positions:
                pos = drone_positions[agent_name]
                dy, dx = int(pos[0]), int(pos[1])
                
                # Offset overlapping drones slightly for visual clarity
                offset_x = (idx % 2 - 0.5) * 8
                offset_y = (idx // 2 - 0.5) * 8
                
                cx = int(dx * self.cell_size + self.cell_size // 2 + offset_x)
                cy = int(self.hud_height + dy * self.cell_size + self.cell_size // 2 + offset_y)
                
                color = self.DRONE_COLORS[idx] if is_alive else (120, 120, 120)

                # Draw Drone Body
                if is_alive:
                    alive_count += 1
                    pygame.draw.circle(self.screen, color, (cx, cy), self.cell_size // 2.5)
                    pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), self.cell_size // 5)
                    
                    # Drone ID label
                    lbl = self.font_small.render(f"D{idx+1}", True, (0, 0, 0))
                    self.screen.blit(lbl, (cx - 6, cy - 6))
                    
                    # Draw Battery Bar above drone
                    bar_w = int(self.cell_size * 0.9)
                    bar_h = 4
                    bx = cx - bar_w // 2
                    by = cy - self.cell_size // 2 - 6
                    
                    # Bar bg
                    pygame.draw.rect(self.screen, (40, 40, 40), (bx, by, bar_w, bar_h))
                    # Bar fill
                    fill_color = (50, 220, 100) if batt_pct > 40 else ((255, 200, 0) if batt_pct > 20 else (255, 50, 50))
                    fill_w = int(bar_w * (batt_pct / 100.0))
                    if fill_w > 0:
                        pygame.draw.rect(self.screen, fill_color, (bx, by, fill_w, bar_h))
                else:
                    # Crash marker 💥
                    pygame.draw.line(self.screen, (255, 50, 50), (cx - 8, cy - 8), (cx + 8, cy + 8), 3)
                    pygame.draw.line(self.screen, (255, 50, 50), (cx - 8, cy + 8), (cx + 8, cy - 8), 3)

        # 5. Draw Top HUD Panel
        panel_rect = pygame.Rect(0, 0, self.window_width, self.hud_height)
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.hud_height), (self.window_width, self.hud_height), 2)

        # HUD Text
        title_txt = self.font_large.render(f"MARL TRACKING DEMO: {model_name.upper()}", True, self.COLOR_ACCENT)
        self.screen.blit(title_txt, (14, 10))

        step_txt = self.font_medium.render(f"Step: {step}/750", True, self.COLOR_TEXT)
        self.screen.blit(step_txt, (14, 40))

        alive_txt = self.font_medium.render(f"Swarm Survival: {alive_count}/4 Alive", True, (50, 220, 100) if alive_count == 4 else (255, 180, 0))
        self.screen.blit(alive_txt, (170, 40))

        avg_b = np.mean(batt_levels) if batt_levels else 0.0
        batt_txt = self.font_medium.render(f"Avg Battery: {avg_b:.1f}%", True, (0, 210, 255))
        self.screen.blit(batt_txt, (360, 40))

        status_str = "🎯 TARGET INTERCEPTED!" if target_found else ("SEARCHING & TRACKING..." if step < 750 else "❌ ESCAPED")
        status_color = (50, 220, 100) if target_found else ((255, 75, 75) if step >= 750 else (255, 200, 0))
        status_txt = self.font_large.render(status_str, True, status_color)
        self.screen.blit(status_txt, (530, 25))

        pygame.display.flip()

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass


def make_tracking_env(grid_size=25, is_self_heal=True, fault_prob=0.0005):
    """Constructs non-rendered underlying env logic."""
    env = DroneSwarmSearch(
        grid_size=grid_size,
        drone_amount=4,
        person_amount=1,
        person_initial_position=(grid_size // 2, grid_size // 2),
        timestep_limit=750,
    )
    env = AllPositionsWrapper(env)
    battery_wrapper = BatteryStationWrapper(
        env,
        max_battery=125,
        depletion_rate=1,
        charge_rate=15,
        fault_prob=fault_prob
    )
    if not is_self_heal:
        battery_wrapper.COMPENSATION_BONUS = 0.0
        battery_wrapper.COMPENSATION_PENALTY = 0.0
        battery_wrapper.COMPENSATION_HORIZON = 0
    else:
        battery_wrapper.COMPENSATION_BONUS = 2.5
        battery_wrapper.COMPENSATION_PENALTY = -0.5
        battery_wrapper.COMPENSATION_HORIZON = 100

    env = RandomDriftWrapper(battery_wrapper, speed=1.0)
    env = TrackingRewardWrapper(env, prob_scale=5.0, step_penalty=-0.1, discovery_bonus=500.0)
    env = RetainDronePosWrapper(env, [(0, 0), (0, 1), (1, 0), (1, 1)])
    return env, battery_wrapper


def main():
    parser = argparse.ArgumentParser(description="Live Pygame Tracking Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="rspo_selfheal",
        choices=[
            "rspo_selfheal", "rspo_vanilla",
            "mappo_selfheal", "mappo_vanilla",
            "qmix_selfheal", "qmix_vanilla",
            "maa2c_selfheal", "maa2c_vanilla",
            "coma_selfheal", "coma_vanilla",
            "iql_selfheal", "iql_vanilla"
        ],
        help="Select model to visualize in live Pygame window"
    )
    parser.add_argument("--fps", type=int, default=20, help="Playback speed (FPS)")
    parser.add_argument("--seed", type=int, default=1000, help="Test seed")
    parser.add_argument("--no_faults", action="store_true", help="Disable random hardware faults")
    args = parser.parse_args()

    print("=" * 70)
    print(f"   LIVE PYGAME DEMO: DYNAMIC TARGET TRACKING ({args.model.upper()})")
    print("=" * 70)

    is_self_heal = "selfheal" in args.model
    fault_prob = 0.0 if args.no_faults else 0.0005

    model_config_map = {
        "rspo_vanilla": (os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*RSPO_V2_Tracking_Vanilla*/**/checkpoint_*"), "rllib", RSPOModelV2),
        "rspo_selfheal": (os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*RSPO_V2_Tracking_SelfHeal*/**/checkpoint_*"), "rllib", RSPOModelV2),
        "mappo_vanilla": (os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*MAPPO_Tracking_Vanilla*/**/checkpoint_*"), "rllib", MAPPOModelVanilla),
        "mappo_selfheal": (os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*MAPPO_Tracking_SelfHeal*/**/checkpoint_*"), "rllib", MAPPOModelVanilla),
        "qmix_vanilla": (os.path.join(SRC_DIR, "results/models/*qmix_tracking_vanilla*/**/agent.th"), "epymarl", None),
        "qmix_selfheal": (os.path.join(SRC_DIR, "results/models/*qmix_tracking_selfheal*/**/agent.th"), "epymarl", None),
        "maa2c_vanilla": (os.path.join(SRC_DIR, "results/models/*maa2c_tracking_vanilla*/**/agent.th"), "epymarl", None),
        "maa2c_selfheal": (os.path.join(SRC_DIR, "results/models/*maa2c_tracking_selfheal*/**/agent.th"), "epymarl", None),
        "coma_vanilla": (os.path.join(SRC_DIR, "results/models/*coma_tracking_vanilla*/**/agent.th"), "epymarl", None),
        "coma_selfheal": (os.path.join(SRC_DIR, "results/models/*coma_tracking_selfheal*/**/agent.th"), "epymarl", None),
        "iql_vanilla": (os.path.join(SRC_DIR, "results/models/*iql_tracking_vanilla*/**/agent.th"), "epymarl", None),
        "iql_selfheal": (os.path.join(SRC_DIR, "results/models/*iql_tracking_selfheal*/**/agent.th"), "epymarl", None),
    }

    pattern, framework, model_cls = model_config_map[args.model]
    ckpt_path = find_best_post_10m_checkpoint(pattern, framework, min_timesteps=10_000_000)

    if not ckpt_path:
        print(f"[!] Error: Could not find trained checkpoint for {args.model} matching {pattern}")
        sys.exit(1)

    print(f"Loading {framework.upper()} Checkpoint: {ckpt_path}\n")
    if framework == "rllib":
        policy_fn = load_rllib_policy(model_cls, ckpt_path, model_name=args.model)
    else:
        policy_fn = load_epymarl_policy(ckpt_path)

    env, battery_wrapper = make_tracking_env(grid_size=25, is_self_heal=is_self_heal, fault_prob=fault_prob)
    renderer = CustomTrackingPygameRenderer(grid_size=25, cell_size=28, hud_height=80)

    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    obs_dict, infos = env.reset(seed=args.seed)

    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    step = 0
    target_found = False

    print("Pygame Window opened! Press ESC or close window to stop.\n")

    try:
        running = True
        while running and env.agents and step < 750 and not target_found:
            step += 1
            
            running = renderer.process_events()
            if not running:
                break

            actions = policy_fn(obs_dict, env.agents)
            obs_dict, rewards, term, trunc, infos = env.step(actions)

            # Check target discovery
            for agent, r in rewards.items():
                if r >= 1.0 or (isinstance(infos.get(agent), dict) and infos.get(agent, {}).get("search_and_find", False)):
                    target_found = True
                    print(f"🎯 TARGET INTERCEPTED AT STEP {step}!")
                    break

            renderer.render(base_env, battery_wrapper, args.model, step, target_found)
            renderer.clock.tick(args.fps)

            if step % 25 == 0:
                alive_count = sum(1 for a in env.agents if obs_dict.get(a) is not None)
                print(f"  Step {step:3d}/750 | Alive Drones: {alive_count}/4 | Target Found: {target_found}")

        # Keep final state visible for 2 seconds
        if target_found:
            for _ in range(30):
                if not renderer.process_events():
                    break
                renderer.render(base_env, battery_wrapper, args.model, step, target_found)
                renderer.clock.tick(15)

    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"\nDemo error: {e}")
    finally:
        renderer.close()

    print(f"\nDemo Complete! Final Result: {'TARGET FOUND in ' + str(step) + ' steps' if target_found else 'TARGET ESCAPED (750 steps)'}")


if __name__ == "__main__":
    main()
