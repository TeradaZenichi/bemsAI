import gym
import numpy as np
import pandas as pd
import json
import os
from gym import spaces

class EnergyEnv(gym.Env):
    """
    Energy environment with PV, load, BESS, and curriculum learning support.
    Actions discretized via bins for charging and discharging percentages,
    with a single no-op (zero) action. Charging is blocked when SoC == 1
    and discharging is blocked when SoC == 0.
    Reward consists only of grid energy cost and overflow/underflow penalty.
    Observations include charging_ratio and discharging_ratio.
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
        discrete_discharge_bins: int = 10
    ):
        super().__init__()

        # Load parameters
        with open(os.path.join(data_dir, 'parameters.json'), 'r') as f:
            self.params = json.load(f)

        # BESS parameters
        b = self.params['BESS']
        self.initial_soc    = b['SoC0']
        self.Emax           = b['Emax']
        self.Pmax_charge    = b['Pmax_c']
        self.Pmax_discharge = b['Pmax_d']
        self.eff            = b['eff']
        self.dt             = self.params['timestep'] / 60.0

        # EDS parameters
        eds = self.params['EDS']
        self.PEDS_max = eds['Pmax']
        self.PEDS_min = eds['Pmin']
        self.cost_dict = eds.get('cost', {})

        # PV & Load maxima
        self.PVmax   = self.params['PV']['Pmax']
        self.Loadmax = self.params['Load']['Pmax']

        # Curriculum settings
        self.difficulty      = float(self.params['ENV']['difficulty'])
        self.test_mode       = test
        self.episode_counter = 0

        # Discretization bins
        self.charge_bins    = discrete_charge_bins
        self.discharge_bins = discrete_discharge_bins

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

        # Initial SoC
        self.soc  = self.initial_soc
        self.done = False

        # Discrete action space
        self.action_space = spaces.Discrete(1 + self.charge_bins + self.discharge_bins)

        # Observation keys
        self.obs_keys = observations or [
            'pv','load','pmax','pmin','soc',
            'hour_sin','day_sin','month_sin','weekday',
            'charging_ratio','discharging_ratio'
        ]
        flat_obs, _ = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32
        )

    def _action_to_p_bess(self, action: int, p_pv: float, p_load: float) -> float:
        # Physical imbalance
        phys_max_charge    = self.PEDS_max + p_pv - p_load
        phys_max_discharge = max(0.0, p_load - p_pv)

        # Headroom & availability
        headroom     = (1.0 - self.soc) * self.Emax / (self.eff * self.dt)
        availability = self.soc * self.Emax * self.eff / self.dt

        # Combine limits
        max_charge    = max(0.0, min(self.Pmax_charge, phys_max_charge, headroom))
        max_discharge = max(0.0, min(self.Pmax_discharge, phys_max_discharge, availability))

        # Build discrete levels
        no_op = np.array([0.0], dtype=np.float32)
        charge_levels = (
            np.linspace(max_charge/self.charge_bins, max_charge, self.charge_bins, dtype=np.float32)
            if self.charge_bins > 0 else np.array([], dtype=np.float32)
        )
        discharge_levels = (
            np.linspace(-max_discharge, -max_discharge/self.discharge_bins, self.discharge_bins, dtype=np.float32)
            if self.discharge_bins > 0 else np.array([], dtype=np.float32)
        )

        all_levels = np.concatenate([no_op, charge_levels, discharge_levels])
        return float(all_levels[action])

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
            if env_cfg.get('curriculum','False').upper()=='TRUE' and \
               self.episode_counter % int(env_cfg['curriculum_steps']) == 0:
                inc = float(env_cfg['curriculum_increment'])
                mx  = float(env_cfg['curriculum_max'])
                self.difficulty = min(self.difficulty + inc, mx)
            # Randomize SoC
            rand_obs = env_cfg.get('randomize_observations', {})
            if rand_obs.get('soc','False').upper()=='TRUE' and env_cfg.get('randomize','False').upper()=='TRUE':
                rng = 0.05 + self.difficulty*0.95
                low, high = max(0,0.5-rng/2), min(1,0.5+rng/2)
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

    def _compute_cost(self, p_grid, hour_str):
        penalty = self.params['RL'].get('Penalty', 0.0)
        # se importação, custo normal
        if p_grid >= 0:
            cost = p_grid * self.dt * penalty
        else:
            # se injeção, não há receita — mas penalizamos o operador
            # usando a mesma tarifa (ou configure outra em parameters.json)
            cost = (-p_grid) * self.dt * penalty
        return cost, penalty

    def step(self, action: int):
        t        = self.pv_series.index[self.current_idx]
        p_pv     = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load   = self.load_series.iloc[self.current_idx] * self.Loadmax
        p_bess   = self._action_to_p_bess(action, p_pv, p_load)

        # ——— Shaping contínuo: potencial baseado em SoC ———
        K     = self.params['RL'].get('shaping_coefficient', 0.0)
        gamma = self.params['RL'].get('gamma', 1.0)
        phi_s = K * self.soc         # potencial antes da ação

        # aplica ação e atualiza SoC (overflow/underflow)
        overflow_penalty = self._update_soc(p_bess)
        phi_s_prime     = K * self.soc  # potencial depois da ação

        # potência na rede (pode ser negativa)
        grid_power = p_load - p_pv + p_bess

        # custo penaliza tanto importação quanto injeção
        energy_cost, tariff = self._compute_cost(grid_power, t.strftime("%H:00"))
        reward = -energy_cost - overflow_penalty

        # adiciona termo de shaping: γ·Φ(s') − Φ(s)
        reward += gamma * phi_s_prime - phi_s
        # ——————————————————————————————————————————————

        # avanço de tempo
        self.current_idx += 1
        if self.current_idx >= self.end_idx:
            self.done = True

        # próxima observação e metadados
        obs, info = self._get_obs()
        info.update({
            'p_bess':           p_bess,
            'p_grid_power':     grid_power,
            'energy_cost':      energy_cost,
            'tariff':           tariff,
            'overflow_penalty': overflow_penalty,
            'time':             t
        })
        return obs, reward, self.done, info


    def _update_soc(self, p_bess):
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

    def _compute_cost(self, p_grid, hour_str):
        tarif = self.cost_dict.get(hour_str, 0.4)
        cost  = (p_grid * self.dt if p_grid > 0 else 0.0) * tarif
        return cost, tarif

    def _get_obs(self):
        if self.current_idx >= len(self.pv_series):
            return np.zeros(len(self.obs_keys), dtype=np.float32), {}

        # timestamp and raw values
        t        = pd.to_datetime(self.pv_series.index[self.current_idx])
        pv_raw   = self.pv_series.iloc[self.current_idx]
        load_raw = self.load_series.iloc[self.current_idx]

        # nominal power for normalization: uses Pnom or sum of EDS limit + PVmax
        nom = self.params.get('Pnom', self.PEDS_max + self.PVmax)

        # compute absolute powers in kW
        p_pv   = pv_raw * self.PVmax
        p_load = load_raw * self.Loadmax

        # simplified charging_ratio normalized by nominal power
        phys_charge     = self.PEDS_max + p_pv - p_load
        charging_ratio  = np.clip(phys_charge / nom, 0.0, 1.0)

        # simplified discharging_ratio normalized by nominal power
        phys_discharge    = p_load - p_pv
        discharging_ratio = np.clip(phys_discharge / nom, 0.0, 1.0)

        # detect PV excess and normalize
        pv_excess       = max(p_pv - p_load, 0.0)
        pv_excess_norm  = np.clip(pv_excess / nom, 0.0, 1.0)

        pv_norm   = np.clip(p_pv   / nom, 0.0, 1.0)  # PV generation normalized
        load_norm = np.clip(p_load / nom, 0.0, 1.0)  # Load demand normalized

        # assemble observation dict
        obs = {
            'soc':               self.soc,            # fraction [0,1]
            'charging_ratio':    charging_ratio,     # normalized to Pnom
            'discharging_ratio': discharging_ratio,  # normalized to Pnom
            'pv_excess_norm':    pv_excess_norm,     # normalized PV surplus
            'pv':                pv_norm,           # normalized PV generation
            'load':              load_norm,         # normalized load demand
            'hour_sin':          np.sin(2*np.pi * t.hour    / 24.0),
            'hour_cos':          np.cos(2*np.pi * t.hour    / 24.0),
            'day_sin':           np.sin(2*np.pi * (t.day-1) / 31.0),
            'day_cos':           np.cos(2*np.pi * (t.day-1) / 31.0),
            'month_sin':         np.sin(2*np.pi * (t.month-1)/ 12.0),
            'month_cos':         np.cos(2*np.pi * (t.month-1)/ 12.0),
            'weekday':           t.weekday() / 6.0,
        }

        # flatten according to self.obs_keys
        flat = np.concatenate([
            np.atleast_1d(obs[k]).astype(np.float32).flatten()
            for k in self.obs_keys if k in obs
        ])

        return flat, obs


    def render(self, mode='human'):
        print(f"SoC={self.soc:.2f}, PV(norm)={self.pv_series.iloc[self.current_idx]:.3f}, "
              f"Load(norm)={self.load_series.iloc[self.current_idx]:.3f}")
