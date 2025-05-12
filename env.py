import gym
import numpy as np
import pandas as pd
import json, os
from gym import spaces

class EnergyEnv(gym.Env):
    """
    Ambiente de energia com PV, carga, BESS e suporte a curriculum learning.
    Modo estático: bins fixos de -Pmax_discharge a +Pmax_charge.
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
        discrete_actions: int = 501,
    ):
        super().__init__()
        # parâmetros
        with open(os.path.join(data_dir, 'parameters.json'),'r') as f:
            self.params = json.load(f)
        env_cfg = self.params['ENV']
        # BESS
        b = self.params['BESS']
        self.initial_soc    = b['SoC0']
        self.Emax           = b['Emax']
        self.Pmax_charge    = b['Pmax_c']
        self.Pmax_discharge = b['Pmax_d']
        self.eff            = b['eff']
        self.dt             = self.params['timestep']/60.0
        # static actions
        self.discrete_actions = discrete_actions
        self.static_levels    = np.linspace(-self.Pmax_discharge, self.Pmax_charge,
                                            self.discrete_actions, dtype=np.float32)
        self.action_space     = spaces.Discrete(self.discrete_actions)
        # EDS
        eds = self.params['EDS']
        self.PEDS_max = eds['Pmax']; self.PEDS_min = eds['Pmin']
        self.cost_dict = eds.get('cost',{})
        # PV & Load maxima
        self.PVmax   = self.params['PV']['Pmax']
        self.Loadmax = self.params['Load']['Pmax']
        # curriculum
        self.difficulty      = float(env_cfg.get('difficulty', 0))
        self.test_mode       = test
        self.episode_counter = 0
        # séries
        mode = 'test' if test else 'train'
        if data is not None:
            mode = data
        self.pv_series   = pd.read_csv(
            os.path.join(data_dir, f'pv_5min_{mode}.csv'),
            index_col='timestamp', parse_dates=['timestamp']
        )['p_norm']
        self.load_series = pd.read_csv(
            os.path.join(data_dir, f'load_5min_{mode}.csv'),
            index_col='timestamp', parse_dates=['timestamp']
        )['p_norm']
        # índices de episódio
        self.start_idx      = start_idx
        self.episode_length = episode_length
        self.current_idx    = start_idx
        self.end_idx        = start_idx + episode_length
        # estado inicial
        self.soc  = self.initial_soc
        self.done = False
        # chaves de observação
        default_keys = [
            'pv', 'load', 'pmax', 'pmin', 'soc',
            'pv_excess', 'pv_charge',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'weekday'
        ]
        self.obs_keys = observations or default_keys
        flat, _ = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, flat.shape, dtype=np.float32)
        self.timestamps = list(self.pv_series.index)

    def new_training_episode(self, start_idx):
        self.start_idx       = start_idx
        self.current_idx     = start_idx
        self.end_idx         = start_idx + self.episode_length
        self.soc             = self.initial_soc
        self.done            = False
        self.difficulty      = float(self.params['ENV'].get('difficulty', 0))
        self.episode_counter = 0
        return self.reset()

    def reset(self):
        env_cfg = self.params['ENV']
        if not self.test_mode:
            self.episode_counter += 1
            # curriculum increment
            if env_cfg.get('curriculum','False').upper() == 'TRUE' and \
               self.episode_counter % int(env_cfg.get('curriculum_steps',1)) == 0:
                inc = float(env_cfg.get('curriculum_increment',0))
                mx  = float(env_cfg.get('curriculum_max',0))
                self.difficulty = min(self.difficulty + inc, mx)
            # randomize SoC
            rand = env_cfg.get('randomize_observations',{})
            if rand.get('soc','False').upper()=='TRUE' and env_cfg.get('randomize','False').upper()=='TRUE':
                rng = 0.05 + self.difficulty*0.95
                low, high = max(0,0.5-rng/2), min(1,0.5+rng/2)
                self.soc = np.random.uniform(low, high)
            else:
                self.soc = self.initial_soc
            # randomize EDS
            if rand.get('eds','False').upper()=='TRUE' and env_cfg.get('randomize','False').upper()=='TRUE':
                scale = 0.05 + self.difficulty
                fac = 1 + np.random.uniform(-scale, scale)
                self.PEDS_max = max(0, self.params['EDS']['Pmax']*fac)
                self.PEDS_min = max(0, self.params['EDS']['Pmin']*fac)
            else:
                self.PEDS_max = self.params['EDS']['Pmax']
                self.PEDS_min = self.params['EDS']['Pmin']
            # randomize index
            if rand.get('idx','False').upper()=='TRUE' and env_cfg.get('randomize','False').upper()=='TRUE':
                lim = int((0.2+0.6*self.difficulty)*0.1*len(self.pv_series))
                self.start_idx = np.random.randint(0, max(1, lim - self.episode_length))
        # definir índices
        self.current_idx = self.start_idx
        self.end_idx     = self.start_idx + self.episode_length
        self.done        = False
        flat, _ = self._get_obs()
        return flat

    def _compute_limits(self, p_pv, p_load):
        phys_c = self.PEDS_max + p_pv - p_load
        phys_d = max(0, p_load - p_pv)
        head   = (1 - self.soc) * self.Emax / (self.eff * self.dt)
        avail  = self.soc * self.Emax * self.eff / self.dt
        max_c  = max(0, min(self.Pmax_charge, phys_c, head))
        max_d  = max(0, min(self.Pmax_discharge, phys_d, avail))
        return max_c, max_d

    def _update_soc(self, p):
        delta    = (p * self.eff if p >= 0 else p / self.eff) * self.dt / self.Emax
        new_soc  = self.soc + delta
        overflow = max(new_soc - 1, 0)
        under    = max(-new_soc, 0)
        pen      = (overflow + under) * abs(p) * self.params['RL'].get('bess_penalty',10.0) * self.dt
        self.soc  = np.clip(new_soc, 0, 1)
        return pen

    def _get_obs(self):
        if self.current_idx >= len(self.pv_series):
            return np.zeros(len(self.obs_keys), dtype=np.float32), {}
        t       = self.pv_series.index[self.current_idx]
        p_pv    = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load  = self.load_series.iloc[self.current_idx] * self.Loadmax
        nom     = self.params.get('Pnom', self.PEDS_max + self.PVmax)
        max_c, max_d = self._compute_limits(p_pv, p_load)
        p_excess = max(p_pv - p_load, 0)
        obs     = {
            'pv':         np.clip(p_pv/nom,0,1),
            'load':       np.clip(p_load/nom,0,1),
            'pmax':       max_c/nom,
            'pmin':       max_d/nom,
            'soc':        self.soc,
            'pv_excess':  p_excess/nom,
            'pv_charge':  0.0,
            'hour_sin':   np.sin(2*np.pi*t.hour/24),
            'hour_cos':   np.cos(2*np.pi*t.hour/24),
            'day_sin':    np.sin(2*np.pi*(t.day-1)/31),
            'day_cos':    np.cos(2*np.pi*(t.day-1)/31),
            'month_sin':  np.sin(2*np.pi*(t.month-1)/12),
            'month_cos':  np.cos(2*np.pi*(t.month-1)/12),
            'weekday':    t.weekday()/6
        }
        flat = np.array([obs.get(k,0.0) for k in self.obs_keys], dtype=np.float32)
        return flat, obs

    def step(self, action: int):
        t       = self.pv_series.index[self.current_idx]
        p_pv    = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load  = self.load_series.iloc[self.current_idx] * self.Loadmax
        max_c, max_d = self._compute_limits(p_pv, p_load)
        raw     = self.static_levels[action]
        p_req   = np.clip(raw, -max_d, max_c)
        # reward
        grid_p  = p_load - p_pv + p_req
        tariff  = self.cost_dict.get(f"{t.hour:02d}:00", 0.4)
        energy_c= max(grid_p,0) * self.dt * tariff
        p_excess= max(p_pv - p_load,0)
        p_charge= max(min(p_req,p_excess),0)
        pv_r    = self.params['RL'].get('pv_use_reward',0.1) * p_charge * self.dt
        pen_over= self._update_soc(p_req)
        reward  = -energy_c + pv_r
        # avançar
        self.current_idx += 1
        self.done = self.current_idx >= self.end_idx
        obs, info = self._get_obs()
        info.update({
            'p_bess': p_req, 'p_grid': grid_p,
            'energy_cost': energy_c, 'overflow_penalty': pen_over,
            'pv_charge': p_charge, 'pv_reward': pv_r,
            'time': t
        })
        # atualizar obs vetor
        if 'pv_charge' in self.obs_keys:
            idx = self.obs_keys.index('pv_charge')
            obs[idx] = p_charge/nom if (nom:=self.params.get('Pnom',self.PEDS_max+self.PVmax)) else obs[idx]
        return obs, reward, self.done, info

    def render(self, mode='human'):
        t=self.current_idx
        print(f"SoC={self.soc:.2f}, PV={self.pv_series.iloc[t]:.3f}, Load={self.load_series.iloc[t]:.3f}")
