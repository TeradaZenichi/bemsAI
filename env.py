import gym
from gym import spaces
import numpy as np
import pandas as pd
import json
import os

class EnergyEnvContinuous(gym.Env):
    """
    Continuous-action energy environment for BESS (Battery Energy Storage System) control.
    Models:
        - Dynamic power limits as a function of SoC [1,4]
        - Efficiency dependent on SoC and power [3,4]
        - Degradation (cycling & calendar aging) [2]
        - Power ramp rate limitation [1,4]
        - End-of-life (EOL) penalty [2,4]
    References:
    [1] IEC 62933-2-1:2017.
    [2] J. Neubauer et al., "Durability and Reliability of Large-Format Li-ion Batteries," NREL/TP-5400-65217, 2015.
    [3] Z. Rao, S. Wang, "A review of power battery thermal energy management," Renewable and Sustainable Energy Reviews, 2011.
    [4] Tesla Powerpack and BYD B-Box datasheets.
    """

    def __init__(self,
                 data_dir='data',
                 dataset='train',
                 start_idx=0,
                 episode_length=288,
                 observations=None,
                 mode='train'):

        super().__init__()
        assert mode in ['train', 'test'], "mode must be 'train' or 'test'"
        self.train_mode = (mode == 'train')
        self.test_mode  = (mode == 'train')

        # Load config from JSON
        cfg_path = os.path.join(data_dir, 'parameters.json')
        config   = json.load(open(cfg_path))
        self.params = config
        env_cfg = config['ENV']
        self.curriculum = env_cfg.get('curriculum', 'False').upper() == 'TRUE'
        self.curriculum_steps = int(env_cfg.get('curriculum_steps', 1))
        self.curriculum_inc = float(env_cfg.get('curriculum_increment', 0.0))
        self.curriculum_max = float(env_cfg.get('curriculum_max', 1.0))
        self.difficulty = float(env_cfg.get('difficulty', 0.0))
        self.episode_counter = 0

        self.randomize = env_cfg.get('randomize', 'False').upper() == 'TRUE'
        rand_cfg = env_cfg.get('randomize_observations', {})
        self.randomize_soc = rand_cfg.get('soc', 'False').upper() == 'TRUE'
        self.randomize_eds = rand_cfg.get('eds', 'False').upper() == 'TRUE'
        self.randomize_idx = rand_cfg.get('idx', 'False').upper() == 'TRUE'

        # Episode pointers
        self.start_idx      = start_idx
        self.episode_length = episode_length
        self.current_idx    = start_idx
        self.end_idx        = start_idx + episode_length

        # --- BESS Parameters from JSON ---
        bess = config['BESS']
        self.soc         = bess['SoC0']
        self.initial_soc = bess['SoC0']
        self.Emax_nom    = bess['Emax']      # kWh
        self.Emax        = bess['Emax']
        self.Pmax_c_nom  = bess['Pmax_c']
        self.Pmax_d_nom  = bess['Pmax_d']
        self.efficiency_nom = bess['eff']

        # SoC operational limits and target (for more realistic operation) [1,4]
        self.soc_min     = bess.get('soc_min', 0.10)
        self.soc_max     = bess.get('soc_max', 0.90)
        self.soc_target  = bess.get('soc_target', 0.0) # Optionally used in shaping/curriculum

        # Degradation parameters (all from JSON for reproducibility) [2]
        self.degradation_rate = bess.get('degradation_rate', 2e-4)
        self.calendar_degradation = bess.get('calendar_degradation', 5e-5)
        self.lifetime_threshold = bess.get('eol_percent', 0.7)           # EOL: % of initial capacity [2,4]
        self.eol_penalty = bess.get('eol_penalty', 100.0)                # Penalty for using EOL battery [2,4]

        # Power ramp rate (should be in JSON; fallback to 10% Pmax_c) [1,4]
        self.P_ramp = bess.get('P_ramp', 0.1 * self.Pmax_c_nom)

        # Counters for degradation logic [2]
        self.cycle_count = 0.0
        self.cycle_energy = 0.0
        self.age_days = 0

        # Simulation timestep (hours)
        self.dt = config['timestep'] / 60.0

        # System nominal power (for normalization)
        self.nom = config.get('Pnom', self.Emax + config['PV']['Pmax'])
        assert self.nom != 0, "Pnom is zero in parameters.json!"

        # Action space: parameterized
        self.action_space = spaces.Box(
            low = np.array([-self.Pmax_d_nom], dtype=np.float32),
            high= np.array([ self.Pmax_c_nom], dtype=np.float32),
            dtype=np.float32
        )

        # Load time series
        self.pv_series = pd.read_csv(
            os.path.join(data_dir, f'pv_5min_{dataset}.csv'),
            index_col='timestamp', parse_dates=['timestamp']
        )['p_norm']
        assert not self.pv_series.isna().any(), "pv_series contains NaN!"
        self.load_series = pd.read_csv(
            os.path.join(data_dir, f'load_5min_{dataset}.csv'),
            index_col='timestamp', parse_dates=['timestamp']
        )['p_norm']
        assert not self.load_series.isna().any(), "load_series contains NaN!"

        self.PVmax, self.Loadmax = config['PV']['Pmax'], config['Load']['Pmax']

        # Grid cost and limits
        eds_cfg = config['EDS']
        self.PEDS_max           = eds_cfg['Pmax']
        self.PEDS_min           = eds_cfg['Pmin']
        self.cost_dict          = eds_cfg.get('cost', {})
        self.grid_violation_coef= config.get('RL', {}).get('grid_violation_penalty', 0.0)

        # Observation space
        default_keys = [
            'pv','load','pmax','pmin','soc', 'tariff',
            'peds_max','peds_min','pv_excess','pv_charge',
            'hour_sin','hour_cos','day_sin','day_cos',
            'month_sin','month_cos','weekday'
        ]
        self.obs_keys = observations or default_keys
        sample_obs, _ = self._get_obs()
        self.observation_space = spaces.Box(
            low = np.full(sample_obs.shape, -np.inf, dtype=np.float32),
            high= np.full(sample_obs.shape,  np.inf, dtype=np.float32),
            dtype=np.float32
        )

        # Power at previous step (for ramp rate limit) [1,4]
        self.last_p_bess = 0.0

    # --- Dynamic maximum charge/discharge according to SoC ---
    # When SoC is near min or max, reduce max charge/discharge rates [1,4]
    def soc_limited_pmax_c(self, soc):
        if soc < self.soc_min or soc > self.soc_max:
            return 0.3 * self.Pmax_c_nom    # 30% of nominal at extremes [1,4]
        return self.Pmax_c_nom

    def soc_limited_pmax_d(self, soc):
        if soc < self.soc_min or soc > self.soc_max:
            return 0.3 * self.Pmax_d_nom
        return self.Pmax_d_nom

    # --- Efficiency model as function of SoC and Power ---
    # Efficiency is lower at SoC extremes and high current [3,4]
    def dynamic_efficiency(self, soc, p):
        soc_central = (self.soc_max + self.soc_min) / 2
        soc_range   = self.soc_max - self.soc_min
        soc_factor  = 0.9 - 0.2 * abs(soc - soc_central) / (soc_range / 2)    # [3,4]
        power_factor = 0.95 - 0.15 * (abs(p) / max(self.Pmax_c_nom, 1e-5))    # [3,4]
        eff = self.efficiency_nom * soc_factor * power_factor
        return np.clip(eff, 0.7, 0.98)   # [3,4] (could also be in JSON)

    # --- Battery degradation: cycling and calendar aging [2] ---
    def apply_degradation(self, p):
        # Accumulate energy moved; degrade by cycle [2]
        self.cycle_energy += abs(p * self.dt)
        if self.cycle_energy >= self.Emax_nom:
            self.cycle_count += 1
            self.cycle_energy -= self.Emax_nom
            self.Emax *= (1 - self.degradation_rate)
        # Calendar degradation per day [2]
        if self.current_idx == self.start_idx:
            self.age_days += 1
            self.Emax *= (1 - self.calendar_degradation)
        # Clamp to End-Of-Life threshold [2,4]
        if self.Emax < self.lifetime_threshold * self.Emax_nom:
            self.Emax = self.lifetime_threshold * self.Emax_nom

    def _reset_battery(self):
        """
        Resets battery aging and degradation: Emax, cycles, age, and cycled energy.
        Use at the beginning of training or when you want a new battery at env.reset().
        """
        self.Emax = self.Emax_nom
        self.cycle_count = 0.0
        self.age_days = 0
        self.cycle_energy = 0.0
        return 


    def reset(self, initial_soc=None):

        self._reset_battery()  # Reset battery state
        # Update curriculum difficulty
        if self.train_mode and self.curriculum:
            self.episode_counter += 1
            if self.episode_counter == self.curriculum_steps:
                self.difficulty = min(self.difficulty + self.curriculum_inc, self.curriculum_max)
                self.episode_counter = 0

        # Randomize start index if configured (window increases with difficulty)
        if self.train_mode and self.randomize and self.randomize_idx:
            lim = int((0.2 + 0.6 * self.difficulty) * 0.1 * len(self.pv_series))
            self.start_idx = np.random.randint(0, max(1, lim - self.episode_length))

        self.current_idx = self.start_idx
        self.end_idx     = self.start_idx + self.episode_length

        # Option to set SoC at reset
        if initial_soc is not None:
            self.initial_soc = float(np.clip(initial_soc, self.soc_min, self.soc_max))
        elif (self.train_mode and self.randomize and self.randomize_soc):
            rng = 0.05 + self.difficulty * 0.95
            low, high = max(self.soc_min, 0.5 - rng/2), min(self.soc_max, 0.5 + rng/2)
            self.initial_soc = np.random.uniform(low, high)

        self.soc = self.initial_soc

        # Randomize EDS limits if configured
        if self.train_mode and self.randomize and self.randomize_eds:
            scale = 0.05 + self.difficulty
            fac   = 1 + np.random.uniform(-scale, scale)
            self.PEDS_max = max(0, self.params['EDS']['Pmax'] * fac)
            self.PEDS_min = max(0, self.params['EDS']['Pmin'] * fac)
        else:
            self.PEDS_max = self.params['EDS']['Pmax']
            self.PEDS_min = self.params['EDS']['Pmin']

        # Power at previous step resets to 0
        self.last_p_bess = 0.0

        obs, _ = self._get_obs()
        return obs


    def _get_obs(self):
        if self.current_idx >= len(self.pv_series):
            return np.zeros(len(self.obs_keys), dtype=np.float32), {}
        t      = self.pv_series.index[self.current_idx]
        p_pv   = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load = self.load_series.iloc[self.current_idx] * self.Loadmax
        max_c, max_d = self._compute_limits(p_pv, p_load)
        p_excess     = max(p_pv - p_load, 0.0)
        obs = {k: 0.0 for k in self.obs_keys}
        obs.update({
            'pv':        p_pv/self.nom,
            'load':      p_load/self.nom,
            'tariff':    self.cost_dict[f"{t.hour:02d}:00"],
            'peds_max':  self.PEDS_max/self.nom,
            'peds_min':  self.PEDS_min/self.nom,
            'pmax':      max_c/self.nom,
            'pmin':      max_d/self.nom,
            'soc':       self.soc,
            'pv_excess': p_excess/self.nom,
            'pv_charge': 0.0,
            'hour_sin':  np.sin(2*np.pi*t.hour/24),
            'hour_cos':  np.cos(2*np.pi*t.hour/24),
            'day_sin':   np.sin(2*np.pi*(t.day-1)/31),
            'day_cos':   np.cos(2*np.pi*(t.day-1)/31),
            'month_sin': np.sin(2*np.pi*(t.month-1)/12),
            'month_cos': np.cos(2*np.pi*(t.month-1)/12),
            'weekday':   t.weekday()/6.0
        })
        return np.array([obs[k] for k in self.obs_keys], dtype=np.float32), obs

    def _compute_limits(self, p_pv, p_load):
        soc = self.soc
        max_c_soc = self.soc_limited_pmax_c(soc)
        max_d_soc = self.soc_limited_pmax_d(soc)
        phys_c = self.PEDS_max + p_pv - p_load
        phys_d = max(0.0, p_load - p_pv)
        head   = (self.soc_max - soc) * self.Emax / (self.efficiency_nom * self.dt)
        avail  = (soc - self.soc_min) * self.Emax * self.efficiency_nom / self.dt
        return (
            max(0.0, min(max_c_soc, phys_c, head)),
            max(0.0, min(max_d_soc, phys_d, avail))
        )

    def _update_soc(self, p):
        # Dynamic efficiency, SoC update, and SoC violation penalty [1,2,3,4]
        eff = self.dynamic_efficiency(self.soc, p)
        delta = (p*eff if p>=0 else p/eff) * self.dt / self.Emax
        new_soc = self.soc + delta
        overflow = max(new_soc - self.soc_max, 0.0)
        underflow = max(self.soc_min - new_soc, 0.0)
        self.soc = np.clip(new_soc, self.soc_min, self.soc_max)
        self.apply_degradation(p)
        bess_penalty = self.params.get('RL', {}).get('bess_penalty', 10.0)
        return (overflow + underflow) * abs(p) * bess_penalty * self.dt

    def step(self, action):
        t      = self.pv_series.index[self.current_idx]
        p_pv   = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load = self.load_series.iloc[self.current_idx] * self.Loadmax
        max_c, max_d = self._compute_limits(p_pv, p_load)

        # --- Power ramp rate constraint [1,4] ---
        raw_p_req = float(np.clip(action[0], -max_d, max_c))
        p_req = np.clip(raw_p_req,
                        self.last_p_bess - self.P_ramp,
                        self.last_p_bess + self.P_ramp)
        self.last_p_bess = p_req

        # --- Energy cost (RL reward) ---
        grid_p = p_load - p_pv + p_req
        tariff = self.cost_dict[f"{t.hour:02d}:00"]
        e_cost = max(grid_p, 0.0) * self.dt * tariff

        # --- Grid violation cost [1] ---
        ov = max(grid_p - self.PEDS_max, 0.0)
        ud = max(-self.PEDS_min - grid_p, 0.0)
        gv = (ov + ud) * self.grid_violation_coef * self.dt

        # --- Potential-based shaping (optional, for RL) ---
        k      = self.params.get('RL', {}).get('potential_scale', 1.0)
        gamma  = self.params.get('RL', {}).get('gamma', 1.0)
        ind    = float(p_pv > p_load)
        soc_b  = self.soc
        pen_bess = self._update_soc(p_req)
        soc_a  = self.soc
        shaping= gamma * (k * soc_a * ind) - (k * soc_b * ind)

        # --- End-of-life (EOL) penalty [2,4] ---
        eol_flag = (self.Emax <= self.lifetime_threshold * self.Emax_nom)
        reward = -e_cost + shaping - gv - pen_bess
        if eol_flag:
            reward -= self.eol_penalty

        self.current_idx += 1
        done = self.current_idx >= self.end_idx

        obs, info = self._get_obs()
        info.update({
            'p_bess':              p_req,
            'p_grid':              grid_p,
            'energy_cost':         e_cost,
            'grid_violation_cost': gv,
            'shaping':             shaping,
            'pen_bess':            pen_bess,
            'Emax':                self.Emax,
            'cycle_count':         self.cycle_count,
            'age_days':            self.age_days,
            'eol_flag':            eol_flag,
            'time':                t
        })
        if 'pv_charge' in self.obs_keys:
            p_ch = max(min(p_req, max(p_pv - p_load, 0.0)), 0.0)
            obs[self.obs_keys.index('pv_charge')] = p_ch / self.nom

        return obs, reward, done, info
