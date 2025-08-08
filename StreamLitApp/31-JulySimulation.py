import sys
import os
import numpy as np
import torch
from enum import Enum
from pony.orm import db_session
import gymnasium as gym
from gymnasium import spaces

# Setup BASE_DIR and imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'simulation_envs', 'scripts', 'db'))
sys.path.insert(0, os.path.join(BASE_DIR, 'simulation_envs', 'scripts', 'core'))
sys.path.insert(0, os.path.join(BASE_DIR, 'simulation_envs', 'files'))

ACTUAL_DB_AVAILABLE = True

try:
    from simulation_envs.scripts.db.readModelDB import get_database, createMetabolome, createBacteria
    from simulation_envs.scripts.core.mainClasses import Microbiome, Pulse, Reactor

    DATABASEFOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', 'backend'))
    DATABASENAME = 'kombucha.sqlite3'
    DB_PATH = os.path.join(DATABASEFOLDER, DATABASENAME)

    if os.path.exists(DB_PATH):
        DB = get_database(DB_PATH)
    else:
        print(f"Warning: Database file not found at {DB_PATH}")
        ACTUAL_DB_AVAILABLE = False

except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    ACTUAL_DB_AVAILABLE = False
    DB = None


@db_session
def get_all_species_ids(db):
    return [row[0] for row in db.execute("SELECT id FROM species")]


# Simulation parameters
param_max_steps = 5
param_dilution = 0.5
param_volume = 100
param_max_dilution = 60
param_unit_change = 0.01
param_target_ethanol = 10.0
param_simul_time = 1
param_simul_steps = 5
param_reward_scale = 100

ACETATE_IDX = 0  # index for acetate metabolite


def getpH(metabolome_concentration):
    pH = min(10, max(3, -0.0491 * metabolome_concentration[ACETATE_IDX] + 6.6369))
    return pH


@db_session
def getReactor(simulTime=1, SimulSteps=100, dilution=0.0, volume=100):
    if not ACTUAL_DB_AVAILABLE or DB is None:
        raise RuntimeError("Database or classes not available.")

    metabolome = createMetabolome(DB, mediaName='kombucha_media', pHFunc=getpH)
    metabolome_feed = createMetabolome(DB, mediaName='kombucha_media', pHFunc=getpH)

    species_ids = get_all_species_ids(DB)
    microbio_dict = {}
    microbio_feed_dict = {}

    for sp_id in species_ids:
        try:
            microbio_dict[sp_id] = createBacteria(DB, speciesID=sp_id, mediaName='kombucha_media')
            feed_bac = createBacteria(DB, speciesID=sp_id, mediaName='kombucha_media')
            for subp in feed_bac.subpopulations.values():
                subp.count = 0
            microbio_feed_dict[sp_id] = feed_bac
        except Exception as e:
            print(f"Warning: Could not create bacteria for species '{sp_id}': {e}")

    microbiome = Microbiome(microbio_dict)
    microbiome_feed = Microbiome(microbio_feed_dict)

    pulse = Pulse(metabolome_feed, microbiome_feed, 0, simulTime, SimulSteps, 0, 0, dilution, dilution)
    reactor = Reactor(microbiome, metabolome, [pulse], volume)
    return reactor
from pony.orm import db_session

@db_session
def run_direct_reactor_simulation(subpop_dict, media_dict, volume=15):
    """
    Direct reactor simulation using user-provided subpopulation counts and media concentrations.

    Args:
        subpop_dict (dict): {subpop_name: count}
        media_dict (dict): {metabolite_name: concentration}
        volume (float): Reactor volume (default 15)

    Returns:
        fig: plotly figure object
    """

    if not ACTUAL_DB_AVAILABLE or DB is None:
        raise RuntimeError("Database not available. Cannot run direct simulation.")

    # Create metabolome with pH modeling
    metabolome = createMetabolome(DB, mediaName='kombucha_media', pHFunc=getpH)

    # Update metabolite concentrations based on user input
    for met_name, conc in media_dict.items():
        if met_name in metabolome.metD:
            metabolome.metD[met_name].concentration = conc
        else:
            print(f"Warning: Metabolite '{met_name}' not found in kombucha_media. Skipping.")
    print([metabolome.metD[i].concentration for i in metabolome.metD])
    # Dynamically load all species and create the microbiome
    species_ids = get_all_species_ids(DB)
    microbio_dict = {}

    for sp_id in species_ids:
        try:
            bac = createBacteria(DB, speciesID=sp_id, mediaName='kombucha_media')
            microbio_dict[sp_id] = bac
        except Exception as e:
            print(f"Warning: Failed to create bacteria for species '{sp_id}': {e}")

    microbiome = Microbiome(microbio_dict)

    # Update subpopulation counts from input
    for subpop_name, count in subpop_dict.items():
        if subpop_name in microbiome.subpopD:
            microbiome.subpopD[subpop_name].count = count
        else:
            print(f"Warning: Subpopulation '{subpop_name}' not found in microbiome. Skipping.")

    # Create feed with same species but 0 counts
    metabolome_feed = createMetabolome(DB, mediaName='kombucha_media', pHFunc=getpH)
    microbio_feed_dict = {}

    for sp_id in species_ids:
        try:
            feed_bac = createBacteria(DB, speciesID=sp_id, mediaName='kombucha_media')
            for subp in feed_bac.subpopulations.values():
                subp.count = 0
            microbio_feed_dict[sp_id] = feed_bac
        except Exception as e:
            print(f"Warning: Could not prepare feed bacteria for '{sp_id}': {e}")

    microbiome_feed = Microbiome(microbio_feed_dict)

    # Create Pulse and Reactor
    pulse = Pulse(metabolome_feed, microbiome_feed, 0, 600, 1000, 0, 0, 0.0, 0.0)
    reactor = Reactor(microbiome, metabolome, [pulse], volume)
    #print(reactor.microbiome.subpopD['bb1'].feedingTerms[0].yields)
    #['acetate', 'ethanol', 'fructose', 'glucose', 'sucrose']
    # Run simulation
    reactor.simulate()

    # Return plot
    return reactor.makePlots(return_fig=True)


class KombuchaGymAction(Enum):
    INCREASE = 0
    DECREASE = 1


class KombuchaGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    @db_session
    def _get_species_ids(self, db):
        return [row[0] for row in db.execute("SELECT id FROM species")]

    def _zero_count_bacteria(self, sp_id):
        try:
            bac = createBacteria(DB, speciesID=sp_id, mediaName='kombucha_media')
            for subp in bac.subpopulations.values():
                subp.count = 0
            return bac
        except Exception as e:
            print(f"Warning: Could not create zero-count bacteria for '{sp_id}': {e}")
            return None

    def __init__(self, max_steps=param_max_steps, reward_scale=param_reward_scale,
                 simulTime=param_simul_time, SimulSteps=param_simul_steps,
                 dilution=param_dilution, volume=param_volume,
                 param_target_ethanol=param_target_ethanol,
                 param_unit_change=param_unit_change,
                 render_mode=None, max_dilution=param_max_dilution):
        super().__init__()
        self.current_step = 0
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.render_mode = render_mode
        self.max_dilution = max_dilution
        self.unit_change = param_unit_change
        self.simulTime = simulTime
        self.SimulSteps = SimulSteps
        self.dilution = dilution
        self.volume = volume
        self.param_target_ethanol = param_target_ethanol

        if not ACTUAL_DB_AVAILABLE or DB is None:
            raise RuntimeError("Cannot create KombuchaGym: Database or classes not available")

        self.metabolome_feed = createMetabolome(DB, mediaName='kombucha_media', pHFunc=getpH)
        species_ids = self._get_species_ids(DB)
        self.microbiome_feed = Microbiome({sp_id: self._zero_count_bacteria(sp_id) for sp_id in species_ids})
        for subpop in self.microbiome_feed.subpopD.values():
            subpop.count = 0

        self.reactor = getReactor(self.simulTime, self.SimulSteps, self.dilution, self.volume)
        self.reactor.simulate()

        self.action_space = spaces.Discrete(2)
        obs = self._get_observation()
        self.observation_space = spaces.Box(
            low=np.zeros_like(obs, dtype=np.float32),
            high=np.full_like(obs, np.inf, dtype=np.float32),
            dtype=np.float32
        )
        self.episode_logs = []

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.dilution = param_dilution
        self.reactor = getReactor(simulTime=self.simulTime, SimulSteps=self.SimulSteps, dilution=0.0, volume=self.volume)
        self.reactor.simulate()
        self.episode_logs = []
        return self._get_observation(), {}

    def _get_observation(self):
        try:
            states = self.reactor.get_states()
        except AttributeError:
            metabolite_states = list(self.reactor.metabolome.concentration)
            bacterial_states = [subpop.count for subpop in self.reactor.microbiome.subpopD.values()]
            states = metabolite_states + bacterial_states
        return np.array(states, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        action_enum = KombuchaGymAction(action)

        if action_enum == KombuchaGymAction.INCREASE:
            self.dilution = min(self.max_dilution, self.dilution + self.unit_change)
        elif action_enum == KombuchaGymAction.DECREASE:
            self.dilution = max(0, self.dilution - self.unit_change)

        pulse = [Pulse(self.metabolome_feed, self.microbiome_feed, 0, self.simulTime,
                       self.SimulSteps, 0, 0, self.dilution, self.dilution)]
        self.reactor = Reactor(self.reactor.microbiome, self.reactor.metabolome, pulse, self.volume)
        self.reactor.simulate()

        subpop_counts = {k: v.count for k, v in self.reactor.microbiome.subpopD.items()}
        current_pH = getpH(self.reactor.metabolome.concentration)
        metabolite_conc = dict(zip(self.reactor.metabolome.metabolites, self.reactor.metabolome.concentration))

        episode_info = {
            'step': self.current_step,
            'subpop_counts': subpop_counts,
            'pH': current_pH,
            'metabolite_concentrations': metabolite_conc,
            'dilution': self.dilution
        }
        self.episode_logs.append(episode_info)

        reward = self.reward_function() * self.reward_scale
        done = self.current_step >= self.max_steps
        obs = self._get_observation()

        info = {'logs': episode_info}
        return obs, reward, done, False, info

    def reward_function(self):
        try:
            ethanol_idx = self.reactor.metabolome.metabolites.index('ethanol')
            ethanol_conc = self.reactor.metabolome.concentration[ethanol_idx]
            deviation = abs(ethanol_conc - self.param_target_ethanol)
            reward = max(0.0, 1 - deviation / self.param_target_ethanol)
            return reward
        except (ValueError, IndexError):
            return 0.0

    def render(self):
        pass


def main(param_max_steps=5, param_dilution=0.5, param_volume=100,
         param_reward_scale=100, param_max_dilution=60, param_unit_change=0.01, **kwargs):

    env = KombuchaGym(
        max_steps=param_max_steps,
        max_dilution=param_max_dilution,
        dilution=param_dilution,
        volume=param_volume,
        reward_scale=param_reward_scale,
        param_unit_change=param_unit_change,
        **kwargs
    )

    obs, info = env.reset()

    logs = []

    for i in range(param_max_steps):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)

        step_data = {
            "step": i + 1,
            "reward": reward,
            "done": done,
            "dilution": info['logs']['dilution'],
            "pH": info['logs']['pH'],
        }

        metabolites = info['logs']['metabolite_concentrations']
        for met, val in metabolites.items():
            step_data[met] = val

        subpop_counts = info['logs']['subpop_counts']
        for sp_id, count in subpop_counts.items():
            step_data[f"subpop_{sp_id}"] = count

        logs.append(step_data)

        if done:
            break

    return logs


if __name__ == "__main__":
    results = main()
    print(f"Simulation finished with {len(results)} steps.")