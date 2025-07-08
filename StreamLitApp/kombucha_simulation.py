import sys
import os
import random
import copy
import sqlite3
import chardet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

import gymnasium as gym
from gymnasium import spaces
from enum import Enum

from tqdm import tqdm
import plotly.io as pio
pio.renderers.default = 'colab'

# Dynamically add script folders to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR =", BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'simulation_envs', 'scripts', 'db'))
sys.path.insert(0, os.path.join(BASE_DIR, 'simulation_envs', 'scripts', 'core'))
sys.path.insert(0, os.path.join(BASE_DIR, 'simulation_envs', 'files'))

ACTUAL_DB_AVAILABLE = True

try:
    from readModelDB import get_database, createMetabolome, createBacteria
    from mainClasses import Microbiome, Pulse, Reactor
    
    # Load database
    #DATABASEFOLDER = os.path.join(BASE_DIR, 'simulation_envs', 'files', 'db_tables')
    #DATABASENAME = 'kombuchaDB_colab.sqlite3'
    DATABASEFOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', 'backend'))
    DATABASENAME = 'microbiomeglobal.sqlite3'
    
    if os.path.exists(os.path.join(DATABASEFOLDER, DATABASENAME)):
        DB = get_database(os.path.join(DATABASEFOLDER, DATABASENAME))
    else:
        print(f"Warning: Database file not found at {os.path.join(DATABASEFOLDER, DATABASENAME)}")
        ACTUAL_DB_AVAILABLE = False
        
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    ACTUAL_DB_AVAILABLE = False

# Simulation parameters
param_max_steps = 500
param_dilution = 0.5
param_volume = 100
param_max_dilution = 60
param_unit_change = 0.01
param_target_ethanol = 10.0
param_simul_time = 1
param_simul_steps = 100
param_reward_scale = 100
TRAINED_NN_PTH = 'q_network_final.pth'

ACETATE_IDX = 0

def getpH(metabolome_concentration):
    """Add a pH function"""
    pH = min(10, max(3, -0.0491*metabolome_concentration[ACETATE_IDX] + 6.6369))
    return pH

def getReactor(simulTime=1, SimulSteps=100, dilution=0.0, volume=100):
    """Create reactor with actual classes if available"""
    if not ACTUAL_DB_AVAILABLE or DB is None:
        raise RuntimeError("Database or classes not available. Please check your paths and database setup.")
    
    metabolome = createMetabolome(DB, mediaName='kombucha_media', pHFunc=getpH)
    metabolome_feed = createMetabolome(DB, mediaName='kombucha_media', pHFunc=getpH)

    microbiome = Microbiome({'bb': createBacteria(DB, speciesID='bb', mediaName='kombucha_media'),
                           'ki': createBacteria(DB, speciesID='ki', mediaName='kombucha_media')})
    microbiome_feed = Microbiome({'bb': createBacteria(DB, speciesID='bb', mediaName='kombucha_media'),
                                'ki': createBacteria(DB, speciesID='ki', mediaName='kombucha_media')})
    
    for i in microbiome_feed.subpopD:
        microbiome_feed.subpopD[i].count = 0

    # Convert dilution to int if needed (fix for type mismatch)
    dilution_param = int(dilution) if isinstance(dilution, float) and dilution.is_integer() else dilution
    
    pulse = Pulse(metabolome_feed, microbiome_feed, 0, simulTime, SimulSteps, 0, 0, dilution_param, dilution_param)
    reactor = Reactor(microbiome, metabolome, [pulse], volume)
    return reactor

def reward_function(reactor):
    target_ethanol = param_target_ethanol

    try:
        print(f"[DEBUG] metabolites: {reactor.metabolome.metabolites}")
        ethanol_idx = reactor.metabolome.metabolites.index('ethanol')
        ethanol_conc = reactor.metabolome.concentration[ethanol_idx]
        print(f"[DEBUG] ethanol concentration: {ethanol_conc}")
        # Normalize deviation and invert: max reward = 1.0 at target
        deviation = abs(ethanol_conc - target_ethanol)
        reward = max(0.0, 1 - deviation / target_ethanol)  # between 0 and 1

        return reward
    except (ValueError, IndexError):
        
        return 0.0  # lowest reward if ethanol not found

class KombuchaGymAction(Enum):
    INCREASE = 0
    DECREASE = 1

class KombuchaGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(self, 
                 max_steps=240,
                 reward_scale=100,
                 simulTime=1,
                 SimulSteps=100,
                 dilution=0.5,
                 volume=100,
                 render_mode=None,
                 max_dilution=60):
        super(KombuchaGym, self).__init__()
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
        
        if not ACTUAL_DB_AVAILABLE or DB is None:
            raise RuntimeError("Cannot create KombuchaGym: Database or classes not available")
        
        self.metabolome_feed = createMetabolome(DB, mediaName='kombucha_media', pHFunc=getpH)
        self.microbiome_feed = Microbiome({'bb': createBacteria(DB, speciesID='bb', mediaName='kombucha_media'),
                                          'ki': createBacteria(DB, speciesID='ki', mediaName='kombucha_media')})
        for i in self.microbiome_feed.subpopD:
            self.microbiome_feed.subpopD[i].count = 0
            
        self.reactor = getReactor(self.simulTime, self.SimulSteps, self.dilution, self.volume)
        self.reactor.simulate()
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0]*12, dtype=np.float32),
            high=np.array([np.inf]*12, dtype=np.float32),
            shape=(12,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        self.current_step = 0
        self.dilution = param_dilution
        self.reactor = getReactor(simulTime=1, SimulSteps=100, dilution=0.0, volume=100)
        self.reactor.simulate()
        self.episode_logs = []
        return self._get_observation(), {}

    def _get_observation(self):
        """Get observation - this calls reactor.get_states() which should return 12 values"""
        if hasattr(self.reactor, 'get_states'):
            states = self.reactor.get_states()
        else:
            # Fallback: manually construct states if get_states method doesn't exist
            metabolite_states = list(self.reactor.metabolome.concentration)
            bacterial_states = [self.reactor.microbiome.subpopD[species].count for species in self.reactor.microbiome.subpopD]
            states = metabolite_states + bacterial_states
            
        print(f"[DEBUG] get_states() returned shape: {np.array(states).shape}")
        print(f"[DEBUG] get_states() values: {states}")
        return np.array(states, dtype=np.float32)

    def step(self, action):
        
        self.current_step += 1

        action_enum = KombuchaGymAction(action)
        if action_enum == KombuchaGymAction.INCREASE:
            self.dilution = min(self.max_dilution, self.dilution + self.unit_change)
        elif action_enum == KombuchaGymAction.DECREASE:
            self.dilution = max(0, self.dilution - self.unit_change)
            
        microbiome = self.reactor.microbiome
        metabolome = self.reactor.metabolome
        
        # Fix type mismatch for dilution parameter
        dilution_param = self.dilution
        if hasattr(Pulse, '__init__'):
            # Check if Pulse expects int for dilution parameter
            pulse = [Pulse(self.metabolome_feed, self.microbiome_feed, 0, self.simulTime, 
                          self.SimulSteps, 0, 0, dilution_param, dilution_param)]
        else:
            pulse = []
            
        self.reactor = Reactor(microbiome, metabolome, pulse, self.volume)
        self.reactor.simulate()
        print(f"[DEBUG] metabolites: {self.reactor.metabolome.metabolites}")
        if 'ethanol' in self.reactor.metabolome.metabolites:
            ethanol_idx = self.reactor.metabolome.metabolites.index('ethanol')
            ethanol_conc = self.reactor.metabolome.concentration[ethanol_idx]
            print(f"[DEBUG] ethanol concentration: {ethanol_conc}")
        else:
            print("[DEBUG] ethanol not found in metabolites")


        subpop_counts = {k: v.count for k, v in self.reactor.microbiome.subpopD.items()}
        current_pH = getpH(self.reactor.metabolome.concentration)
        metabolite_conc = dict(zip(self.reactor.metabolome.metabolites, self.reactor.metabolome.concentration))
        dilution_level = self.dilution

        episode_info = {
            'step': self.current_step,
            'subpop_counts': subpop_counts,
            'pH': current_pH,
            'metabolite_concentrations': metabolite_conc,
            'dilution': dilution_level
        }
        self.episode_logs.append(episode_info)

        observation = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = False
        info = episode_info
        return observation, reward, terminated, truncated, info

    def _get_reward(self):
        """Get reward"""
        return reward_function(self.reactor)

    def _is_terminated(self):
        """Check if terminated"""
        return self.current_step >= self.max_steps

    def render(self):
        """Render environment"""
        if self.render_mode == "human":
            if hasattr(self.reactor, 'makePlots'):
                self.reactor.makePlots()
            else:
                self._create_plots()
        elif self.render_mode == "rgb_array":
            if hasattr(self.reactor, 'makePlots'):
                return self.reactor.makePlots()
            else:
                return self._create_plots()

    def _create_plots(self):
        """Create basic plots if makePlots method is not available"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot episode logs if available
        if hasattr(self, 'episode_logs') and self.episode_logs:
            steps = [log['step'] for log in self.episode_logs]
            pHs = [log['pH'] for log in self.episode_logs]
            dilutions = [log['dilution'] for log in self.episode_logs]
            
            axes[0, 0].plot(steps, pHs, 'b-o')
            axes[0, 0].set_title('pH Evolution')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('pH')
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(steps, dilutions, 'r-o')
            axes[0, 1].set_title('Dilution Rate')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Dilution Rate')
            axes[0, 1].grid(True)
            
            # Extract ethanol concentrations
            ethanol_concs = [log['metabolite_concentrations'].get('ethanol', 0) 
                           for log in self.episode_logs]
            axes[1, 0].plot(steps, ethanol_concs, 'g-o')
            axes[1, 0].set_title('Ethanol Concentration')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Ethanol (g/L)')
            axes[1, 0].grid(True)
            
            # Extract bacterial populations
            bb_counts = [log['subpop_counts'].get('bb', 0) for log in self.episode_logs]
            ki_counts = [log['subpop_counts'].get('ki', 0) for log in self.episode_logs]
            axes[1, 1].plot(steps, bb_counts, 'purple', label='bb', marker='o')
            axes[1, 1].plot(steps, ki_counts, 'orange', label='ki', marker='s')
            axes[1, 1].set_title('Bacterial Populations')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig

class PreprocessEnv(gym.ObservationWrapper):
    """Preprocessing wrapper - exact copy from your original code"""
    def __init__(self, env):
        super(PreprocessEnv, self).__init__(env)
        
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return self._obs_to_flat_tensor(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = torch.tensor(reward).view(1, -1).float()
        terminated = torch.tensor(terminated).view(1, -1).int()
        truncated = torch.tensor(truncated).view(1, -1).int()
        return self._obs_to_flat_tensor(obs), reward, terminated, truncated, info

    def _obs_to_flat_tensor(self, obs):
        grid_tensor = torch.tensor(obs).flatten()
        return torch.cat([grid_tensor]).unsqueeze(dim=0).float()

def debug_reactor_states():
    if not ACTUAL_DB_AVAILABLE or DB is None:
        print("Cannot debug: Database not available")
        return
    
    print("=== DEBUGGING REACTOR STATES ===")
    reactor = getReactor()
    reactor.simulate()
    
    print(f"Reactor type: {type(reactor)}")
    print(f"Has get_states method: {hasattr(reactor, 'get_states')}")
    
    if hasattr(reactor, 'get_states'):
        states = reactor.get_states()
        print(f"get_states() returns type: {type(states)}")
        print(f"get_states() shape: {np.array(states).shape if hasattr(states, '__len__') else 'Not array-like'}")
        print(f"get_states() values: {states}")
    else:
        print("Reactor does not have get_states method!")
        print("Available methods:", [method for method in dir(reactor) if not method.startswith('_')])
        
        # Check metabolome and microbiome
        print(f"\nMetabolome metabolites: {reactor.metabolome.metabolites}")
        print(f"Metabolome concentrations shape: {np.array(reactor.metabolome.concentration).shape}")
        print(f"Microbiome subpopulations: {list(reactor.microbiome.subpopD.keys())}")
        
        # Try to construct 12 states manually
        metabolite_states = reactor.metabolome.concentration
        bacterial_states = [reactor.microbiome.subpopD[species].count for species in reactor.microbiome.subpopD]
        all_states = list(metabolite_states) + bacterial_states
        print(f"Manual states construction: {len(all_states)} states")
        print(f"Manual states: {all_states}")

def main(**kwargs):
    """Main simulation function"""
    # Update parameters if provided
    global param_max_steps, param_dilution, param_target_ethanol
    if 'param_max_steps' in kwargs:
        param_max_steps = int(kwargs['param_max_steps'])
    if 'param_dilution' in kwargs:
        param_dilution = float(kwargs['param_dilution'])
    if 'param_target_ethanol' in kwargs:
        param_target_ethanol = float(kwargs['param_target_ethanol'])
    
    print("Starting Kombucha Microbial Simulation...")
    print(f"Database available: {ACTUAL_DB_AVAILABLE}")
    print(f"Database loaded: {DB is not None}")
    
    if not ACTUAL_DB_AVAILABLE:
        print("\n ERROR: Actual database classes not available!")
        print("Please ensure:")
        print("1. simulation_envs repository is cloned")
        print("2. Database is created and populated")
        print("3. Paths in the script are correct")
        return None
    
    if DB is None:
        print("\n ERROR: Database not found!")
        print(f"Looking for database at: {os.path.join(DATABASEFOLDER, DATABASENAME)}")
        return None
    
    print(f"Parameters: max_steps={param_max_steps}, dilution={param_dilution}, target_ethanol={param_target_ethanol}")
    
    # Debug reactor states first
    debug_reactor_states()
    
    try:
        # Create environment
        env = KombuchaGym(max_steps=min(param_max_steps, 20))  
        
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Run simulation
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")
        
        episode_data = []
        done = False
        step = 0
        
        while not done and step < min(param_max_steps, 20):
            # Random action for demonstration
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"Step {step+1}: Action={action}, Reward={reward:.4f}, pH={info['pH']:.2f}")
            
            episode_data.append({
                'step': step+1,
                'action': action,
                'reward': reward,
                'pH': info['pH'],
                'dilution': info['dilution'],
                'ethanol': info['metabolite_concentrations']['ethanol']
            })
            
            step += 1
        
        # Create summary plot
        if episode_data:
            df = pd.DataFrame(episode_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            axes[0, 0].plot(df['step'], df['pH'], 'b-o')
            axes[0, 0].set_title('pH Evolution')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('pH')
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(df['step'], df['ethanol'], 'g-o')
            axes[0, 1].set_title('Ethanol Concentration')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Ethanol (g/L)')
            axes[0, 1].grid(True)
            
            axes[1, 0].plot(df['step'], df['dilution'], 'r-o')
            axes[1, 0].set_title('Dilution Rate')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Dilution Rate')
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(df['step'], df['reward'], 'm-o')
            axes[1, 1].set_title('Reward')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.show()
        
        print("Simulation completed!")
        return episode_data
        
    except Exception as e:
        print(f"❌ ERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()