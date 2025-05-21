import gym
from gym import spaces
import numpy as np
import pandas as pd
import json
import os

class EnergyEnvContinuous(gym.Env):
    """
    Continuous-action energy environment for BESS control with curriculum learning 
    and optional randomization.
    Modes: 'train', 'test', 'eval'.
    """

    def __init__(self,
                 data_dir='data',
                 start_idx=0,
                 episode_length=288,
                 observations=None,
                 mode='train'):
        super().__init__()

        # Load configuration
        cfg_path = os.path.join(data_dir, 'parameters.json')
        config   = json.load(open(cfg_path))
        self.params = config
        env_cfg = config['ENV']

        # --- Initialize episode pointers before observation sampling ---
        self.start_idx      = start_idx
        self.episode_length = episode_length
        self.current_idx    = start_idx
        self.end_idx        = start_idx + episode_length

        # --- Initialize SOC ---
        bess = config['BESS']
        self.initial_soc = bess['SoC0']
        self.soc         = self.initial_soc

        # --- Curriculum and randomization flags ---
        self.test_mode       = (mode != 'train')
        self.eval_mode       = (mode == 'eval')
        self.curriculum      = env_cfg.get('curriculum', 'False').upper() == 'TRUE'
        self.curriculum_steps= int(env_cfg.get('curriculum_steps', 1))
        self.curriculum_inc  = float(env_cfg.get('curriculum_increment', 0.0))
        self.curriculum_max  = float(env_cfg.get('curriculum_max', 1.0))
        self.difficulty      = float(env_cfg.get('difficulty', 0.0))
        self.episode_counter = 0

        self.randomize       = env_cfg.get('randomize', 'False').upper() == 'TRUE'
        rand_cfg = env_cfg.get('randomize_observations', {})
        self.randomize_soc  = rand_cfg.get('soc','False').upper() == 'TRUE'
        self.randomize_eds  = rand_cfg.get('eds','False').upper() == 'TRUE'
        self.randomize_idx  = rand_cfg.get('idx','False').upper() == 'TRUE'

        # --- BESS and time step ---
        self.Emax       = bess['Emax']
        self.Pmax_c     = bess['Pmax_c']
        self.Pmax_d     = bess['Pmax_d']
        self.efficiency = bess['eff']
        self.dt         = config['timestep'] / 60.0  # convert minutes to hours

        # --- Nominal power and assert non-zero ---
        self.nom = config.get('Pnom', self.Emax + config['PV']['Pmax'])
        assert self.nom != 0, "Pnom está zero em parameters.json!"

        # --- Action space ---
        self.action_space = spaces.Box(
            low = np.array([-self.Pmax_d], dtype=np.float32),
            high= np.array([ self.Pmax_c], dtype=np.float32),
            dtype=np.float32
        )

        # --- Load time series ---
        split = 'train' if mode in ('train','eval') else 'test'
        self.pv_series = pd.read_csv(
            os.path.join(data_dir, f'pv_5min_{split}.csv'),
            index_col='timestamp', parse_dates=['timestamp']
        )['p_norm']
        assert not self.pv_series.isna().any(), "pv_series contém NaN!"

        self.load_series = pd.read_csv(
            os.path.join(data_dir, f'load_5min_{split}.csv'),
            index_col='timestamp', parse_dates=['timestamp']
        )['p_norm']
        assert not self.load_series.isna().any(), "load_series contém NaN!"

        self.PVmax, self.Loadmax = config['PV']['Pmax'], config['Load']['Pmax']

        # --- Grid cost structure ---
        eds_cfg = config['EDS']
        self.PEDS_max           = eds_cfg['Pmax']
        self.PEDS_min           = eds_cfg['Pmin']
        self.cost_dict          = eds_cfg.get('cost', {})
        self.grid_violation_coef= config['RL'].get('grid_violation_penalty', 10.0)

        # --- Observation space ---
        default_keys = [
            'pv','load','pmax','pmin','soc', 'tariff',
            'peds_max','peds_min','pv_excess','pv_charge',
            'hour_sin','hour_cos','day_sin','day_cos',
            'month_sin','month_cos','weekday'
        ]
        self.obs_keys = observations or default_keys
        sample_obs, _ = self._get_obs()  # safe: current_idx initialized
        self.observation_space = spaces.Box(
            low = np.full(sample_obs.shape, -np.inf, dtype=np.float32),
            high= np.full(sample_obs.shape,  np.inf, dtype=np.float32),
            dtype=np.float32
        )

    def reset(self):
        # Update curriculum difficulty
        if not self.test_mode and self.curriculum:
            self.episode_counter += 1
            if self.episode_counter == self.curriculum_steps:
                self.difficulty = min(self.difficulty + self.curriculum_inc, self.curriculum_max)
                self.episode_counter = 0

        # Randomize start index if configured
        if not self.test_mode and self.randomize and self.randomize_idx:
            lim = int((0.2 + 0.6*self.difficulty) * 0.1 * len(self.pv_series))
            self.start_idx = np.random.randint(0, max(1, lim - self.episode_length))
         
        self.current_idx = self.start_idx
        self.end_idx     = self.start_idx + self.episode_length

        # Randomize SOC if configured
        if not self.test_mode and self.randomize and self.randomize_soc:
            rng = 0.05 + self.difficulty * 0.95
            low, high = max(0, 0.5-rng/2), min(1, 0.5+rng/2)
            self.initial_soc = np.random.uniform(low, high)
        
        self.soc = self.initial_soc

        # Randomize EDS limits if configured
        if not self.test_mode and self.randomize and self.randomize_eds:
            scale = 0.05 + self.difficulty
            fac   = 1 + np.random.uniform(-scale, scale)
            self.PEDS_max = max(0, self.params['EDS']['Pmax'] * fac)
            self.PEDS_min = max(0, self.params['EDS']['Pmin'] * fac)
        else:
            self.PEDS_max = self.params['EDS']['Pmax']
            self.PEDS_min = self.params['EDS']['Pmin']

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
        phys_c = self.PEDS_max + p_pv - p_load
        phys_d = max(0.0, p_load - p_pv)
        head   = (1 - self.soc) * self.Emax / (self.efficiency * self.dt)
        avail  = self.soc * self.Emax * self.efficiency / self.dt
        return (
            max(0.0, min(self.Pmax_c, phys_c, head)),
            max(0.0, min(self.Pmax_d, phys_d, avail))
        )

    def _update_soc(self, p):
        delta   = (p*self.efficiency if p>=0 else p/self.efficiency) * self.dt / self.Emax
        new_soc = self.soc + delta
        overflow   = max(new_soc - 1.0, 0.0)
        underflow  = max(-new_soc, 0.0)
        self.soc   = np.clip(new_soc, 0.0, 1.0)
        return (overflow + underflow) * abs(p) * self.params['RL'].get('bess_penalty', 10.0) * self.dt

    def step(self, action):
        t      = self.pv_series.index[self.current_idx]
        p_pv   = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load = self.load_series.iloc[self.current_idx] * self.Loadmax
        max_c, max_d = self._compute_limits(p_pv, p_load)
        p_req        = float(np.clip(action[0], -max_d, max_c))

        # Energy cost
        grid_p = p_load - p_pv + p_req
        tariff = self.cost_dict[f"{t.hour:02d}:00"]
        e_cost = max(grid_p, 0.0) * self.dt * tariff

        # Grid violation cost
        ov = max(grid_p - self.PEDS_max, 0.0)
        ud = max(-self.PEDS_min - grid_p, 0.0)
        gv = (ov + ud) * self.grid_violation_coef * self.dt

        # Potential-based shaping
        k      = self.params['RL'].get('potential_scale', 0.0)
        γ      = self.params['RL'].get('gamma', 0.0)
        ind    = float(p_pv > p_load)
        soc_b  = self.soc
        self._update_soc(p_req)
        soc_a  = self.soc
        shaping= γ * (k * soc_a * ind) - (k * soc_b * ind)

        reward = -e_cost + shaping - gv

        # Advance pointer
        self.current_idx += 1
        done = self.current_idx >= self.end_idx

        obs, info = self._get_obs()
        info.update({
            'p_bess':              p_req,
            'p_grid':              grid_p,
            'energy_cost':         e_cost,
            'grid_violation_cost': gv,
            'shaping':             shaping,
            'time':                t
        })

        if 'pv_charge' in self.obs_keys:
            p_ch = max(min(p_req, max(p_pv - p_load, 0.0)), 0.0)
            obs[self.obs_keys.index('pv_charge')] = p_ch / self.nom

        return obs, reward, done, info
