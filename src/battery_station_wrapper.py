import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium.spaces import Tuple, Box

class BatteryStationWrapper(BaseParallelWrapper):
    """
    Simulates battery constraints, charging stations, and attrition for running out of battery.
    Requires AllPositionsWrapper to be applied FIRST so `obs[agent][0]` holds the positions.

    When a drone dies (battery <= 0):
      - Its crash coordinate is recorded and broadcast to all surviving teammates
      - Surviving drones receive a compensation reward for moving towards the crash site
        (fades out over COMPENSATION_HORIZON steps)
      - The dead drone's position slots are set to -1.0 in teammates' observations
    """

    # Compensation reward settings (boosted so survivors actually react to crashes)
    COMPENSATION_BONUS = 2.5     # Reward for moving closer to a crash site
    COMPENSATION_PENALTY = -0.5  # Penalty for moving away from a crash site
    COMPENSATION_HORIZON = 100   # Steps after death before compensation reward fades to zero

    def __init__(self, env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.0005):
        super().__init__(env)
        self.max_battery = max_battery
        self.depletion_rate = depletion_rate
        self.charge_rate = charge_rate
        self.fault_prob = fault_prob  # Random failure probability per step per drone
        
        self.battery = {}
        self.alive = {}
        self.prev_dist = {}
        
        # Crash tracking
        self.crash_coords = {}       # agent -> (y, x) or None
        self.crash_step = {}         # agent -> step number when they died
        self.current_step = 0
        
        # Compensation tracking: distance of each alive agent to each crash site
        self.prev_crash_dist = {}    # (alive_agent, dead_agent) -> manhattan distance
        
        self.grid_size = getattr(self.env, "grid_size", getattr(self.env.unwrapped, "grid_size", 25))
        
        # Base station at top-left corner only (spawn location)
        self.charging_stations = [(0, 0)]

        # Ensure observation spaces reflect the new concatenated array sizes
        self.observation_spaces = {
            agent: self.observation_space(agent)
            for agent in self.env.possible_agents
        }
        
    def observation_space(self, agent):
        base_space = self.env.observation_space(agent)
        positions_box = base_space[0]
        matrix_box = base_space[1]
        
        n_agents = len(self.env.possible_agents)
        # positions (n_agents*2) + battery (n_agents) + crash_coords (n_agents*2) + nearest_station (2)
        new_shape = (positions_box.shape[0] + n_agents + n_agents * 2 + 2,)
        
        new_positions_box = Box(low=-1.0, high=1.0, shape=new_shape, dtype=np.float32)
        return Tuple((new_positions_box, matrix_box))

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        
        # FIX: DSSE overwrites reward_scheme in reset(). We must re-apply our fixed values.
        # This ensures reward_poc remains 0.0 and rewards stay flat/predictable.
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        base_env.reward_scheme.update({
            "default": -0.1,
            "exceed_timestep": 0.0,
            "search_cell": 5.0,
            "done": 500.0,
            "reward_poc": 0.0,
        })
        
        self.current_step = 0
        for agent in self.env.possible_agents:
            self.battery[agent] = self.max_battery
            self.alive[agent] = True
            self.prev_dist[agent] = 0
            self.crash_coords[agent] = None
            self.crash_step[agent] = None
            
        self.prev_crash_dist = {}
            
        for agent in infos:
            # DSSE's compute_infos() returns the same dict ref for all agents;
            # we must copy to avoid cross-agent overwrites.
            infos[agent] = dict(infos[agent])
            infos[agent]["alive"] = True
            infos[agent]["battery"] = self.max_battery
            infos[agent]["charging"] = False
            infos[agent]["stranded_event"] = False
            infos[agent]["crash_coords"] = None

        obs = self._append_battery_crash_and_mask_dead(obs)
        return obs, infos

    def _dist_to_nearest_station(self, pos):
        return min(abs(pos[0] - s[0]) + abs(pos[1] - s[1]) for s in self.charging_stations)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, actions):
        self.current_step += 1
        
        # Override actions for stranded drones (force SEARCH/IDLE = 8)
        filtered_actions = {}
        for agent in self.env.possible_agents:
            if agent in actions and self.alive.get(agent, False):
                filtered_actions[agent] = actions[agent]
            else:
                filtered_actions[agent] = 8
                
        obs, rewards, terminations, truncations, infos = self.env.step(filtered_actions)
        
        new_stranded = []
        
        # --- Battery logic & death detection ---
        for agent in self.env.possible_agents:
            if not self.alive.get(agent, False):
                rewards[agent] = 0.0
                continue
                
            if agent in obs:
                # The first two elements in the position vector are the agent's (y, x)
                pos_y, pos_x = obs[agent][0][0], obs[agent][0][1]
                pos = (int(pos_y), int(pos_x))
                
                # Measure distance to nearest station
                current_dist = self._dist_to_nearest_station(pos)
                
                # Check if at a charging station (distance 0)
                if current_dist == 0:
                    self.battery[agent] = min(self.max_battery, self.battery[agent] + self.charge_rate)
                    charging = True
                else:
                    self.battery[agent] -= self.depletion_rate
                    charging = False
                    
                    # Battery breadcrumb: only when critically low (< 30),
                    # nudge towards charging stations.
                    # Only active in self-healing mode (compensation_horizon > 0).
                    # In vanilla mode, drones learn battery management from death penalty alone.
                    if self.COMPENSATION_HORIZON > 0 and self.battery[agent] < 30:
                        urgency = 1.0 + (30 - self.battery[agent]) / 30.0  # Scales from 1.0 to 2.0
                        prev = self.prev_dist.get(agent, current_dist)
                        if current_dist < prev:
                            rewards[agent] += 0.3 * urgency
                        elif current_dist > prev:
                            rewards[agent] -= 0.3 * urgency
                            
                self.prev_dist[agent] = current_dist
                
                # Check for stranding/attrition
                if self.battery[agent] <= 0:
                    self.battery[agent] = 0
                    self.alive[agent] = False
                    self.crash_coords[agent] = pos
                    self.crash_step[agent] = self.current_step
                    new_stranded.append(agent)
                    rewards[agent] -= 50.0  # Large penalty to teach them to avoid dying!
                
                # --- Random Fault Injection ---
                # Even with full battery, a drone can randomly "malfunction" and crash.
                # This forces the policy to learn attrition handling.
                elif self.fault_prob > 0 and np.random.random() < self.fault_prob:
                    self.battery[agent] = 0
                    self.alive[agent] = False
                    self.crash_coords[agent] = pos
                    self.crash_step[agent] = self.current_step
                    new_stranded.append(agent)
                    # No death penalty for random faults (not the drone's fault)
                    rewards[agent] = 0.0
                    
                if isinstance(infos.get(agent), dict):
                    # DSSE's compute_infos() returns the same dict ref for all agents;
                    # we must copy to avoid cross-agent overwrites.
                    infos[agent] = dict(infos[agent])
                    infos[agent]["alive"] = self.alive[agent]
                    infos[agent]["battery"] = self.battery[agent]
                    infos[agent]["charging"] = charging
                    infos[agent]["stranded_event"] = agent in new_stranded
                    infos[agent]["crash_coords"] = self.crash_coords[agent]

        # --- Compensation reward for surviving drones moving towards crash sites ---
        for agent in self.env.possible_agents:
            if isinstance(infos.get(agent), dict):
                infos[agent] = dict(infos[agent])
            else:
                infos[agent] = {}
            infos[agent]["compensation_reward"] = 0.0

        for agent in self.env.possible_agents:
            if not self.alive.get(agent, False):
                continue
            if agent not in obs:
                continue
                
            pos_y, pos_x = obs[agent][0][0], obs[agent][0][1]
            agent_pos = (int(pos_y), int(pos_x))
            
            for dead_agent in self.env.possible_agents:
                crash_pos = self.crash_coords.get(dead_agent)
                if crash_pos is None:
                    continue
                    
                # Calculate fade factor: linearly decay from 1.0 to 0.0 over COMPENSATION_HORIZON steps
                steps_since_death = self.current_step - self.crash_step[dead_agent]
                
                # Safety check for disabled compensation (horizon=0)
                if self.COMPENSATION_HORIZON <= 0:
                    continue

                if steps_since_death > self.COMPENSATION_HORIZON:
                    continue  # Compensation period expired
                    
                fade = 1.0 - (steps_since_death / self.COMPENSATION_HORIZON)
                
                current_crash_dist = self._manhattan(agent_pos, crash_pos)
                key = (agent, dead_agent)
                prev = self.prev_crash_dist.get(key, current_crash_dist)
                
                comp = 0.0
                if current_crash_dist < prev:
                    comp = self.COMPENSATION_BONUS * fade
                elif current_crash_dist > prev:
                    comp = self.COMPENSATION_PENALTY * fade
                    
                rewards[agent] += comp
                infos[agent]["compensation_reward"] += comp
                self.prev_crash_dist[key] = current_crash_dist

        obs = self._append_battery_crash_and_mask_dead(obs)
        return obs, rewards, terminations, truncations, infos

    def _append_battery_crash_and_mask_dead(self, obs):
        all_agents = self.env.possible_agents
        n_agents = len(all_agents)
        
        # Build global normalized battery array
        batt_array = np.array([self.battery[a] / self.max_battery for a in all_agents], dtype=np.float32)
        
        # Build global crash coordinate array (normalized by grid_size, 0.0 if alive)
        crash_array = np.zeros(n_agents * 2, dtype=np.float32)
        for i, agent in enumerate(all_agents):
            if self.crash_coords[agent] is not None:
                crash_y, crash_x = self.crash_coords[agent]
                crash_array[2 * i] = crash_y / self.grid_size
                crash_array[2 * i + 1] = crash_x / self.grid_size
        
        for idx, agent in enumerate(all_agents):
            if agent not in obs:
                continue
                
            if not self.alive.get(agent, False):
                # Stranded drone sees zeroes
                positions, matrix = obs[agent]
                total_extra = n_agents + n_agents * 2 + 2  # battery + crash coords + station
                obs[agent] = (np.zeros(positions.shape[0] + total_extra, dtype=np.float32), np.zeros_like(matrix))
            else:
                positions, matrix = obs[agent]
                # PUSH: Normalize positions (0.0 to 1.0)
                masked_positions = (positions.astype(np.float32) / self.grid_size)
                
                # Mask out stranded drones in the observing agent's coords
                for dead_idx, dead_agent in enumerate(all_agents):
                    if not self.alive.get(dead_agent, False):
                        target_idx = dead_idx
                        # Adjust for AllPositionsWrapper's active index swap
                        if dead_idx == 0:
                            target_idx = idx
                        elif dead_idx == idx:
                            target_idx = 0
                            
                        masked_positions[2 * target_idx] = -1.0
                        masked_positions[2 * target_idx + 1] = -1.0
                
                # Reorder battery array to match AllPositionsWrapper swap logic (self is always index 0)
                agent_batt_array = batt_array.copy()
                agent_batt_array[[0, idx]] = agent_batt_array[[idx, 0]]
                
                # Reorder crash array with same swap logic
                agent_crash_array = crash_array.copy()
                # Swap the (y, x) pairs for indices 0 and idx
                agent_crash_array[[0, idx * 2]] = agent_crash_array[[idx * 2, 0]]
                agent_crash_array[[1, idx * 2 + 1]] = agent_crash_array[[idx * 2 + 1, 1]]
                
                # PUSH: Find nearest station and add its normalized coords
                pos_y, pos_x = positions[0], positions[1]
                stations = np.array(self.charging_stations)
                dists = np.sum(np.abs(stations - [pos_y, pos_x]), axis=1)
                nearest_station = stations[np.argmin(dists)]
                norm_station = nearest_station.astype(np.float32) / self.grid_size
                
                # Append battery array, crash coords, and station to positions
                final_positions = np.concatenate([
                    masked_positions, 
                    agent_batt_array, 
                    agent_crash_array,
                    norm_station
                ])
                obs[agent] = (final_positions, matrix)
                
        return obs
