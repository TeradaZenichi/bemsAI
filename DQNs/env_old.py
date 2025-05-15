import gym
import numpy as np
import pandas as pd
import json
import os
from gym import spaces

class EnergyEnv(gym.Env):
    """
    Energy environment with PV, load, BESS, and curriculum learning support.
    Supports two action discretization methods:
      - dynamic: bins adapt each step based on available headroom/availability
      - static: fixed bins from -Pmax_discharge to +Pmax_charge using discrete_actions

    Observations include normalized PV/load, SoC, pmax/pmin, time features (sin & cos),
    charging/discharging ratios, and weekday.
    """
    def __init__(
        self,
        data_dir='data',
        timestep_min=5,
        start_idx=0,
        episode_length=288,
        test=False,
        observations=None,
        data=None,
        discrete_charge_bins: int = 10,
        discrete_discharge_bins: int = 10,
        discrete_actions: int = 501,
    ):
        super().__init__()

        # Load parameters
        with open(os.path.join(data_dir, 'parameters.json'), 'r') as f:
            self.params = json.load(f)
        env_cfg = self.params['ENV']

        # Action discretization method: 'dynamic' or 'static'
        self.action_discretization = env_cfg.get('action_discretization', 'dynamic').lower()
        self.discrete_actions = discrete_actions

        # BESS parameters
        b = self.params['BESS']
        self.initial_soc    = b['SoC0']
        self.Emax           = b['Emax']
        self.Pmax_charge    = b['Pmax_c']
        self.Pmax_discharge = b['Pmax_d']
        self.eff            = b['eff']
        self.dt             = self.params['timestep'] / 60.0

        # Static discretization precompute
        if self.action_discretization == 'static':
            neg = -self.Pmax_discharge
            pos =  self.Pmax_charge
            self.static_levels = np.linspace(neg, pos, self.discrete_actions, dtype=np.float32)
            self.action_space = spaces.Discrete(self.discrete_actions)
        else:
            # Dynamic bins
            self.charge_bins    = discrete_charge_bins
            self.discharge_bins = discrete_discharge_bins
            self.action_space = spaces.Discrete(1 + self.charge_bins + self.discharge_bins)

        # EDS parameters
        eds = self.params['EDS']
        self.PEDS_max = eds['Pmax']
        self.PEDS_min = eds['Pmin']
        self.cost_dict = eds.get('cost', {})

        # PV & Load maxima
        self.PVmax   = self.params['PV']['Pmax']
        self.Loadmax = self.params['Load']['Pmax']

        # Curriculum settings
        self.difficulty      = float(env_cfg['difficulty'])
        self.test_mode       = test
        self.episode_counter = 0

        # Load time series
        mode = 'test' if test else 'train'
        if data is not None:
            mode = data
        self.pv_series = pd.read_csv(
            os.path.join(data_dir, f'pv_5min_{mode}.csv'),
            index_col='timestamp', parse_dates=['timestamp']
        )['p_norm']
        self.load_series = pd.read_csv(
            os.path.join(data_dir, f'load_5min_{mode}.csv'),
            index_col='timestamp', parse_dates=['timestamp']
        )['p_norm']

        # Episode indexing
        self.start_idx      = start_idx
        self.episode_length = episode_length
        self.current_idx    = start_idx
        self.end_idx        = start_idx + episode_length

        # Initial SoC and done flag
        self.soc  = self.initial_soc
        self.done = False

        # Observation keys
        self.obs_keys = observations or [
            'pv','load','pmax','pmin','soc',
            'hour_sin','hour_cos',
            'day_sin','day_cos',
            'month_sin','month_cos',
            'weekday',
            'charging_ratio','discharging_ratio'
        ]
        flat_obs, _ = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32
        )


    def new_training_episode(self, start_idx):
        self.start_idx       = start_idx
        self.current_idx     = start_idx
        self.end_idx         = start_idx + self.episode_length
        self.soc             = self.initial_soc
        self.done            = False
        self.difficulty      = float(self.params['ENV']['difficulty'])
        self.episode_counter = 0
        return self.reset()

    def reset(self):
        if not self.test_mode:
            self.episode_counter += 1
            env_cfg = self.params['ENV']
            # Curriculum increment
            if env_cfg.get('curriculum', 'False').upper() == 'TRUE' and \
               self.episode_counter % int(env_cfg['curriculum_steps']) == 0:
                inc = float(env_cfg['curriculum_increment'])
                mx  = float(env_cfg['curriculum_max'])
                self.difficulty = min(self.difficulty + inc, mx)

            # Randomize SoC
            rand_obs = env_cfg.get('randomize_observations', {})
            if rand_obs.get('soc','False').upper()=='TRUE' and env_cfg.get('randomize','False').upper()=='TRUE':
                rng = 0.05 + self.difficulty*0.95
                low, high = max(0, 0.5 - rng/2), min(1, 0.5 + rng/2)
                self.soc = np.random.uniform(low, high)
            else:
                self.soc = self.initial_soc

            # Randomize EDS
            if rand_obs.get('eds','False').upper()=='TRUE' and env_cfg.get('randomize','False').upper()=='TRUE':
                scale = 0.05 + self.difficulty
                fac = 1 + np.random.uniform(-scale, scale)
                self.PEDS_max = max(0, self.params['EDS']['Pmax'] * fac)
                self.PEDS_min = max(0, self.params['EDS']['Pmin'] * fac)
            else:
                self.PEDS_max = self.params['EDS']['Pmax']
                self.PEDS_min = self.params['EDS']['Pmin']

            # Randomize index
            if rand_obs.get('idx','False').upper()=='TRUE' and env_cfg.get('randomize','False').upper()=='TRUE':
                lim = int((0.2 + 0.6*self.difficulty) * 0.1 * len(self.pv_series))
                self.start_idx = np.random.randint(0, max(1, lim - self.episode_length))

            self.current_idx = self.start_idx
            self.end_idx     = self.start_idx + self.episode_length
            self.done        = False
        else:
            self.soc         = self.initial_soc
            self.current_idx = self.start_idx
            self.end_idx     = self.start_idx + self.episode_length
            self.done        = False

        flat_obs, _ = self._get_obs()
        return flat_obs

    def _action_to_p_bess(self, action: int, p_pv: float, p_load: float) -> float:
        """
        Converte índice de ação em potência BESS (kW), retorna raw_req
        """
        # Physical imbalance
        phys_max_charge    = self.PEDS_max + p_pv - p_load
        phys_max_discharge = max(0.0, p_load - p_pv)
        headroom     = (1.0 - self.soc) * self.Emax / (self.eff * self.dt)
        availability = self.soc * self.Emax * self.eff / self.dt
        max_charge    = max(0.0, min(self.Pmax_charge, phys_max_charge, headroom))
        max_discharge = max(0.0, min(self.Pmax_discharge, phys_max_discharge, availability))

        # Raw request
        if self.action_discretization == 'static':
            raw_req = float(self.static_levels[action])
        else:
            no_op = np.array([0.0], dtype=np.float32)
            charge_levels = (
                np.linspace(max_charge/self.charge_bins, max_charge, self.charge_bins, dtype=np.float32)
                if self.charge_bins>0 else np.array([],dtype=np.float32)
            )
            discharge_levels = (
                np.linspace(-max_discharge,
                            -max_discharge/self.discharge_bins,
                            self.discharge_bins,
                            dtype=np.float32)
                if self.discharge_bins>0 else np.array([],dtype=np.float32)
            )
            all_levels = np.concatenate([no_op, charge_levels, discharge_levels])
            raw_req = float(all_levels[action])
        return raw_req

    def _update_soc(self, p_bess: float) -> float:
        """
        Atualiza SoC e retorna penalidade overflow/underflow.
        """
        if p_bess >= 0:
            delta = (p_bess * self.eff * self.dt) / self.Emax
        else:
            delta = (p_bess / self.eff * self.dt) / self.Emax
        penalty = 0.0
        self.soc += delta
        if self.soc > 1.0:
            overflow = self.soc - 1.0
            penalty += overflow * abs(p_bess) * self.params['RL'].get('bess_penalty', 10.0)
            self.soc = 1.0
        if self.soc < 0.0:
            under = -self.soc
            penalty += under * abs(p_bess) * self.params['RL'].get('bess_penalty', 10.0)
            self.soc = 0.0
        return penalty * self.dt

    def _compute_cost(self, p_grid: float, hour_str: str):
        tarif = self.cost_dict.get(hour_str, 0.4)
        cost  = (p_grid * self.dt if p_grid > 0 else 0.0) * tarif
        return cost, tarif

    def _get_obs(self):
        if self.current_idx >= len(self.pv_series):
            return np.zeros(len(self.obs_keys), dtype=np.float32), {}
        t        = pd.to_datetime(self.pv_series.index[self.current_idx])
        pv_raw   = self.pv_series.iloc[self.current_idx]
        load_raw = self.load_series.iloc[self.current_idx]
        nom      = self.params.get('Pnom', self.PEDS_max + self.PVmax)
        p_pv     = pv_raw * self.PVmax
        p_load   = load_raw * self.Loadmax
        phys_charge    = self.PEDS_max + p_pv - p_load
        phys_discharge = p_load - p_pv
        max_charge     = max(0.0, min(self.Pmax_charge, phys_charge))
        max_discharge  = max(0.0, min(self.Pmax_discharge, phys_discharge))
        obs = {
            'pv':                np.clip(p_pv/nom, 0., 1.),
            'load':              np.clip(p_load/nom, 0., 1.),
            'pmax':              max_charge/nom,
            'pmin':              max_discharge/nom,
            'soc':               self.soc,
            'charging_ratio':    np.clip(phys_charge/nom, 0., 1.),
            'discharging_ratio': np.clip(phys_discharge/nom, 0., 1.),
            'hour_sin':          np.sin(2*np.pi * t.hour    / 24.0),
            'hour_cos':          np.cos(2*np.pi * t.hour    / 24.0),
            'day_sin':           np.sin(2*np.pi * (t.day-1) / 31.0),
            'day_cos':           np.cos(2*np.pi * (t.day-1) / 31.0),
            'month_sin':         np.sin(2*np.pi * (t.month-1)/ 12.0),
            'month_cos':         np.cos(2*np.pi * (t.month-1)/ 12.0),
            'weekday':           t.weekday() / 6.0,
        }
        missing = [k for k in self.obs_keys if k not in obs]
        if missing:
            raise KeyError(f"Missing obs keys: {missing}")
        flat = np.concatenate([np.atleast_1d(obs[k]).astype(np.float32).flatten() for k in self.obs_keys])
        return flat, obs

    def step(self, action: int):
        # 1) collect state
        t      = self.pv_series.index[self.current_idx]
        p_pv   = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load = self.load_series.iloc[self.current_idx] * self.Loadmax
        # 2) physical limits
        phys_max_charge    = self.PEDS_max + p_pv - p_load
        phys_max_discharge = max(0.0, p_load - p_pv)
        headroom     = (1.0 - self.soc) * self.Emax / (self.eff * self.dt)
        availability = self.soc * self.Emax * self.eff / self.dt
        max_charge    = max(0.0, min(self.Pmax_charge, phys_max_charge, headroom))
        max_discharge = max(0.0, min(self.Pmax_discharge, phys_max_discharge, availability))
        # 3) raw request
        raw_req = self._action_to_p_bess(action, p_pv, p_load)
        # 4) clamp + basic penalty
        bess_penalty    = self.params['RL'].get('bess_penalty', 10.0)
        extra_penalty   = self.params['RL'].get('pv_gt_load_penalty', 15.0)
        attempt_penalty = 0.0
        if raw_req > max_charge:
            excess = raw_req - max_charge
            attempt_penalty += excess * bess_penalty * self.dt
            p_req = max_charge
        elif raw_req < -max_discharge:
            excess = abs(raw_req) - max_discharge
            attempt_penalty += excess * bess_penalty * self.dt
            p_req = -max_discharge
        else:
            p_req = raw_req
        # 5) extra penalty if no demand
        if self.action_discretization == 'static' and p_pv > p_load and raw_req < 0:
            attempt_penalty += abs(raw_req) * extra_penalty * self.dt
        # 6) update SoC + overflow penalty
        K               = self.params['RL'].get('shaping_coefficient', 0.0)
        gamma           = self.params['RL'].get('gamma', 1.0)
        phi_s           = K * self.soc
        overflow_penalty = self._update_soc(p_req)
        phi_s_prime     = K * self.soc
        # 7) grid cost & reward
        grid_power     = p_load - p_pv + p_req
        energy_cost, tariff = self._compute_cost(grid_power, t.strftime("%H:00"))
        reward = (
            - energy_cost
            - overflow_penalty
            - attempt_penalty
            + (gamma * phi_s_prime - phi_s)
        )
        # 8) advance time
        self.current_idx += 1
        if self.current_idx >= self.end_idx:
            self.done = True
        # 9) return
        obs, info = self._get_obs()
        info.update({
            'p_bess':           p_req,
            'p_grid_power':     grid_power,
            'energy_cost':      energy_cost,
            'tariff':           tariff,
            'overflow_penalty': overflow_penalty,
            'attempt_penalty':  attempt_penalty,
            'time':             t
        })
        return obs, reward, self.done, info

    def render(self, mode='human'):
        print(f"SoC={self.soc:.2f}, PV(norm)={self.pv_series.iloc[self.current_idx]:.3f}, "
              f"Load(norm)={self.load_series.iloc[self.current_idx]:.3f}")


    def _update_soc(self, p_bess: float) -> float:
        """
        Atualiza o State of Charge dado p_bess (kW) e devolve
        penalização por overflow/underflow (kW·h * penalty_rate).
        """
        # converte potência em ΔSoC
        if p_bess >= 0:
            delta = (p_bess * self.eff * self.dt) / self.Emax
        else:
            delta = (p_bess / self.eff * self.dt) / self.Emax

        penalty = 0.0
        self.soc += delta

        # overflow
        if self.soc > 1.0:
            overflow = self.soc - 1.0
            penalty += overflow * abs(p_bess) * self.params['RL'].get('bess_penalty', 10.0)
            self.soc = 1.0

        # underflow
        if self.soc < 0.0:
            under = -self.soc
            penalty += under * abs(p_bess) * self.params['RL'].get('bess_penalty', 10.0)
            self.soc = 0.0

        return penalty * self.dt


    def _get_obs(self):
        # If beyond available data, return zeros
        if self.current_idx >= len(self.pv_series):
            return np.zeros(len(self.obs_keys), dtype=np.float32), {}

        t        = pd.to_datetime(self.pv_series.index[self.current_idx])
        pv_raw   = self.pv_series.iloc[self.current_idx]
        load_raw = self.load_series.iloc[self.current_idx]
        nom = self.params.get('Pnom', self.PEDS_max + self.PVmax)
        p_pv   = pv_raw * self.PVmax
        p_load = load_raw * self.Loadmax
        # recompute limits
        phys_max_charge    = self.PEDS_max + p_pv - p_load
        headroom           = (1.0 - self.soc) * self.Emax / (self.eff * self.dt)
        max_charge         = max(0.0, min(self.Pmax_charge, phys_max_charge, headroom))
        phys_max_discharge = max(0.0, p_load - p_pv)
        availability       = self.soc * self.Emax * self.eff / self.dt
        max_discharge      = max(0.0, min(self.Pmax_discharge, phys_max_discharge, availability))
        charging_ratio    = np.clip((self.PEDS_max + p_pv - p_load) / nom, 0.0, 1.0)
        discharging_ratio = np.clip((p_load - p_pv) / nom, 0.0, 1.0)
        pv_excess_norm    = np.clip(max(p_pv - p_load, 0.0) / nom, 0.0, 1.0)
        pv_norm           = np.clip(p_pv   / nom, 0.0, 1.0)
        load_norm         = np.clip(p_load / nom, 0.0, 1.0)
        obs = {
            'pv':               pv_norm,
            'load':             load_norm,
            'pmax':             max_charge / nom,
            'pmin':             max_discharge / nom,
            'soc':              self.soc,
            'charging_ratio':   charging_ratio,
            'discharging_ratio':discharging_ratio,
            'hour_sin':         np.sin(2*np.pi * t.hour/24.0),
            'hour_cos':         np.cos(2*np.pi * t.hour/24.0),
            'day_sin':          np.sin(2*np.pi * (t.day-1)/31.0),
            'day_cos':          np.cos(2*np.pi * (t.day-1)/31.0),
            'month_sin':        np.sin(2*np.pi * (t.month-1)/12.0),
            'month_cos':        np.cos(2*np.pi * (t.month-1)/12.0),
            'weekday':          t.weekday() / 6.0,
        }
        missing = [k for k in self.obs_keys if k not in obs]
        if missing:
            raise KeyError(f"Missing obs keys: {missing}")
        flat = np.concatenate([np.atleast_1d(obs[k]).astype(np.float32).flatten() for k in self.obs_keys])
        return flat, obs

    def render(self, mode='human'):
        print(f"SoC={self.soc:.2f}, PV(norm)={self.pv_series.iloc[self.current_idx]:.3f}, "
              f"Load(norm)={self.load_series.iloc[self.current_idx]:.3f}")
