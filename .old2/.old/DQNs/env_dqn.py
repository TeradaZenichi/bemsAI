import os
import json
import numpy as np
import pandas as pd
import gym
from gym import spaces

class EnergyEnv(gym.Env):
    """
    Energy environment with PV, load, BESS, and curriculum learning.
    Uses static action discretization and potential-based reward shaping
    to encourage battery charging when PV excess is available.
    """

    def __init__(
        self,
        data_dir: str = 'data',
        start_idx: int = 0,
        episode_length: int = 288,
        test: bool = False,
        observations: list = None,
        mode: str = None,
        discrete_actions: int = 501,
    ):
        super().__init__()
        # Load configuration
        config = json.load(open(os.path.join(data_dir, 'parameters.json')))
        self.params = config
        env_cfg = config['ENV']

        # BESS parameters
        bess_cfg = config['BESS']
        self.initial_soc = bess_cfg['SoC0']
        self.Emax = bess_cfg['Emax']
        self.Pmax_charge = bess_cfg['Pmax_c']
        self.Pmax_discharge = bess_cfg['Pmax_d']
        self.efficiency = bess_cfg['eff']
        self.dt = config['timestep'] / 60.0  # Convert minutes to hours

        # Static action levels
        self.discrete_actions = discrete_actions
        self.action_levels = np.linspace(
            -self.Pmax_discharge,
             self.Pmax_charge,
             self.discrete_actions,
             dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.discrete_actions)

        # EDS cost structure
        eds_cfg = config['EDS']
        self.PEDS_max = eds_cfg['Pmax']
        self.PEDS_min = eds_cfg['Pmin']
        self.cost_dict = eds_cfg.get('cost', {})

        # PV and load maxima
        self.PVmax = config['PV']['Pmax']
        self.Loadmax = config['Load']['Pmax']

        # Curriculum learning settings
        self.difficulty = float(env_cfg.get('difficulty', 0))
        self.test_mode = test
        self.episode_counter = 0

        # Load time series data
        data_mode = mode or ('test' if test else 'train')
        self.pv_series = pd.read_csv(
            os.path.join(data_dir, f'pv_5min_{data_mode}.csv'),
            index_col='timestamp', parse_dates=['timestamp']
        )['p_norm']
        self.load_series = pd.read_csv(
            os.path.join(data_dir, f'load_5min_{data_mode}.csv'),
            index_col='timestamp', parse_dates=['timestamp']
        )['p_norm']

        # Episode indices
        self.start_idx = start_idx
        self.current_idx = start_idx
        self.episode_length = episode_length
        self.end_idx = start_idx + episode_length
        self.soc = self.initial_soc
        self.done = False

        # Observation keys
        default_keys = [
            'pv', 'load', 'pmax', 'pmin', 'soc',
            'pv_excess', 'pv_charge',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'weekday'
        ]
        self.obs_keys = observations or default_keys
        sample_obs, _ = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=sample_obs.shape, dtype=np.float32
        )
        self.timestamps = list(self.pv_series.index)

    def new_training_episode(self, start_idx: int):
        """Start a new training episode at the given index."""
        self.start_idx = start_idx
        self.current_idx = start_idx
        self.end_idx = start_idx + self.episode_length
        self.soc = self.initial_soc
        self.done = False
        # Reset curriculum difficulty
        env_cfg = self.params['ENV']
        self.difficulty = float(env_cfg.get('difficulty', 0))
        self.episode_counter = 0
        return self.reset()

    def reset(self):
        """Reset environment state and apply curriculum/randomization logic."""
        env_cfg = self.params['ENV']
        if not self.test_mode:
            self.episode_counter += 1
            # Curriculum increment
            if (env_cfg.get('curriculum', 'False').upper() == 'TRUE' and
                self.episode_counter % int(env_cfg.get('curriculum_steps', 1)) == 0):
                inc = float(env_cfg.get('curriculum_increment', 0))
                mx = float(env_cfg.get('curriculum_max', 0))
                self.difficulty = min(self.difficulty + inc, mx)

            # Randomize initial SoC
            rand_cfg = env_cfg.get('randomize_observations', {})
            if (rand_cfg.get('soc','False').upper()=='TRUE' and
                env_cfg.get('randomize','False').upper()=='TRUE'):
                rng = 0.05 + self.difficulty * 0.95
                low, high = max(0,0.5-rng/2), min(1,0.5+rng/2)
                self.soc = np.random.uniform(low, high)
            else:
                self.soc = self.initial_soc

            # Randomize EDS capacity
            if (rand_cfg.get('eds','False').upper()=='TRUE' and
                env_cfg.get('randomize','False').upper()=='TRUE'):
                scale = 0.05 + self.difficulty
                fac = 1 + np.random.uniform(-scale, scale)
                self.PEDS_max = max(0, self.params['EDS']['Pmax'] * fac)
                self.PEDS_min = max(0, self.params['EDS']['Pmin'] * fac)
            else:
                self.PEDS_max = self.params['EDS']['Pmax']
                self.PEDS_min = self.params['EDS']['Pmin']

            # Randomize episode start index
            if (rand_cfg.get('idx','False').upper()=='TRUE' and
                env_cfg.get('randomize','False').upper()=='TRUE'):
                lim = int((0.2 + 0.6*self.difficulty) * 0.1 * len(self.pv_series))
                self.start_idx = np.random.randint(0, max(1, lim - self.episode_length))

        # Final reset of indices
        self.current_idx = self.start_idx
        self.end_idx = self.start_idx + self.episode_length
        self.done = False
        obs, _ = self._get_obs()
        return obs

    def _compute_limits(self, p_pv: float, p_load: float):
        """Compute max charge/discharge power given current SoC and imbalance."""
        phys_charge = self.PEDS_max + p_pv - p_load
        phys_discharge = max(0.0, p_load - p_pv)
        headroom = (1 - self.soc) * self.Emax / (self.efficiency * self.dt)
        availability = self.soc * self.Emax * self.efficiency / self.dt
        max_charge = max(0.0, min(self.Pmax_charge, phys_charge, headroom))
        max_discharge = max(0.0, min(self.Pmax_discharge, phys_discharge, availability))
        return max_charge, max_discharge

    def _update_soc(self, p: float):
        """Update SoC with efficiency losses and compute overflow/underflow penalty."""
        delta = (p * self.efficiency if p >= 0 else p / self.efficiency) * self.dt / self.Emax
        new_soc = self.soc + delta
        overflow = max(new_soc - 1.0, 0.0)
        underflow = max(-new_soc, 0.0)
        penalty = (overflow + underflow) * abs(p) * self.params['RL'].get('bess_penalty', 10.0) * self.dt
        self.soc = np.clip(new_soc, 0.0, 1.0)
        return penalty

    def _get_obs(self):
        """Construct observation vector and auxiliary info dict."""
        if self.current_idx >= len(self.pv_series):
            return np.zeros(len(self.obs_keys), dtype=np.float32), {}

        t = self.pv_series.index[self.current_idx]
        p_pv = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load = self.load_series.iloc[self.current_idx] * self.Loadmax
        nom = self.params.get('Pnom', self.PEDS_max + self.PVmax)

        max_c, max_d = self._compute_limits(p_pv, p_load)
        p_excess = max(p_pv - p_load, 0.0)

        obs = {key: 0.0 for key in self.obs_keys}
        obs.update({
            'pv': p_pv / nom,
            'load': p_load / nom,
            'pmax': max_c / nom,
            'pmin': max_d / nom,
            'soc': self.soc,
            'pv_excess': p_excess / nom,
            'pv_charge': 0.0,
            'hour_sin': np.sin(2 * np.pi * t.hour / 24),
            'hour_cos': np.cos(2 * np.pi * t.hour / 24),
            'day_sin': np.sin(2 * np.pi * (t.day - 1) / 31),
            'day_cos': np.cos(2 * np.pi * (t.day - 1) / 31),
            'month_sin': np.sin(2 * np.pi * (t.month - 1) / 12),
            'month_cos': np.cos(2 * np.pi * (t.month - 1) / 12),
            'weekday': t.weekday() / 6.0
        })

        flat_obs = np.array([obs[k] for k in self.obs_keys], dtype=np.float32)
        return flat_obs, obs

    def step(self, action: int):
        """Take action, compute reward with potential-based shaping, and advance state."""
        t = self.pv_series.index[self.current_idx]
        p_pv = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load = self.load_series.iloc[self.current_idx] * self.Loadmax

        # Compute limits and requested power
        max_c, max_d = self._compute_limits(p_pv, p_load)
        raw_req = self.action_levels[action]
        p_req = np.clip(raw_req, -max_d, max_c)

        # Energy cost (positive grid draw)
        grid_power = p_load - p_pv + p_req
        tariff = self.cost_dict.get(f"{t.hour:02d}:00", 0.4)
        energy_cost = max(grid_power, 0.0) * self.dt * tariff

        # Potential-based shaping when PV excess
        p_excess = max(p_pv - p_load, 0.0)
        indicator = 1.0 if p_excess > 0 else 0.0
        soc_before = self.soc
        overflow_penalty = self._update_soc(p_req)
        soc_after = self.soc

        k = self.params['RL'].get('potential_scale', 1.0)
        gamma = self.params['RL'].get('gamma', 1.0)
        Phi_t = k * soc_before * indicator
        Phi_tp1 = k * soc_after * indicator
        shaping = gamma * Phi_tp1 - Phi_t

        # Final reward
        reward = -energy_cost + shaping

        # Advance time step
        self.current_idx += 1
        self.done = self.current_idx >= self.end_idx

        obs, info = self._get_obs()
        info.update({
            'p_bess': p_req,
            'p_grid': grid_power,
            'energy_cost': energy_cost,
            'overflow_penalty': overflow_penalty,
            'Phi_t': Phi_t,
            'Phi_tp1': Phi_tp1,
            'shaping': shaping,
            'time': t
        })

        # Update pv_charge observation if present
        if 'pv_charge' in self.obs_keys:
            p_charge = max(min(p_req, p_excess), 0.0)
            nom = self.params.get('Pnom', self.PEDS_max + self.PVmax)
            idx = self.obs_keys.index('pv_charge')
            obs[idx] = p_charge / nom

        return obs, reward, self.done, info

    def render(self, mode='human'):
        """Print current SoC, PV and load norms."""
        t = self.current_idx
        print(
            f"SoC={self.soc:.2f}, ``PV_norm={self.pv_series.iloc[t]:.3f}, ``" +
            f"Load_norm={self.load_series.iloc[t]:.3f}"  # noqa: W605
        )
