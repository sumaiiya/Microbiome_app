import sys
import os
from turtle import st
import numpy as np
import torch
from enum import Enum
from pony.orm import db_session
import gymnasium as gym
from gymnasium import spaces
from scipy.special import gammaln
from typing import Dict, List, Union, Any, Optional
import traceback

# Setup BASE_DIR and imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'simulation_envs', 'scripts', 'db'))
sys.path.insert(0, os.path.join(BASE_DIR, 'simulation_envs', 'scripts', 'core'))
sys.path.insert(0, os.path.join(BASE_DIR, 'simulation_envs', 'files'))

ACTUAL_DB_AVAILABLE = True

try:
    from simulation_envs.scripts.db.readModelDB import get_database, createMetabolome, createBacteria
    from simulation_envs.scripts.core.mainClasses import Microbiome, Pulse, Reactor

    # Try multiple possible database locations
    DATABASENAME = 'kombucha.sqlite3'
    
    possible_paths = [
        # Original path (backend folder one level up)
        os.path.abspath(os.path.join(BASE_DIR, '..', 'backend', DATABASENAME)),
        # Same directory as the script
        os.path.join(BASE_DIR, DATABASENAME),
        # Backend folder in same directory
        os.path.join(BASE_DIR, 'backend', DATABASENAME),
        # Parent directory
        os.path.abspath(os.path.join(BASE_DIR, '..', DATABASENAME)),
        # Two levels up (in case structure is different)
        os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'backend', DATABASENAME)),
        # Common database locations
        os.path.join(BASE_DIR, 'database', DATABASENAME),
        os.path.join(BASE_DIR, 'data', DATABASENAME),
        os.path.join(BASE_DIR, 'db', DATABASENAME)
    ]
    
    print(f"🔍 Searching for database '{DATABASENAME}' in possible locations:")
    DB_PATH = None
    DB = None
    
    for path in possible_paths:
        print(f"   Checking: {path}")
        if os.path.exists(path):
            print(f"   ✅ Found database at: {path}")
            DB_PATH = path
            try:
                DB = get_database(DB_PATH)
                print(f"   ✅ Successfully connected to database")
                break
            except Exception as e:
                print(f"   ❌ Failed to connect to database: {e}")
                continue
        else:
            print(f"   ❌ Not found")
    
    if DB_PATH is None or DB is None:
        print(f"❌ Database '{DATABASENAME}' not found in any of the expected locations")
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"📁 Script directory (BASE_DIR): {BASE_DIR}")
        print(f"📁 Contents of BASE_DIR: {os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else 'Directory not accessible'}")
        
        # Check parent directory contents
        parent_dir = os.path.abspath(os.path.join(BASE_DIR, '..'))
        print(f"📁 Parent directory: {parent_dir}")
        if os.path.exists(parent_dir):
            print(f"📁 Contents of parent directory: {os.listdir(parent_dir)}")
        
        ACTUAL_DB_AVAILABLE = False
    else:
        print(f"✅ Database successfully loaded from: {DB_PATH}")

except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print(f"📁 Python path includes:")
    for path in sys.path[:10]:  # Show first 10 paths
        print(f"   {path}")
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
    """
    Safe pH calculation with bounds checking
    """
    if len(metabolome_concentration) <= ACETATE_IDX:
        return 6.5  # Default pH if acetate not available
    
    acetate_conc = max(0, metabolome_concentration[ACETATE_IDX])  # Ensure non-negative
    pH = min(10, max(3, -0.0491 * acetate_conc + 6.6369))
    return pH

    beta = max(beta, 1e-6)    # Avoid division by zero
    
    try:
        # Calculate with numerical safety
        log_result = alpha * np.log(beta) - gammaln(alpha) + (alpha - 1) * np.log(x) - beta * x
        
        # Clip extreme values to prevent overflow/underflow
        log_result = np.clip(log_result, -700, 700)  # Safe range for exp
        
        result = np.exp(log_result)
        
        # Final safety check
        if np.isnan(result) or np.isinf(result):
            return 1e-12
        
        return result
        
    except (ValueError, OverflowError, FloatingPointError):
        return 1e-12

def safe_gammaD(x, alpha, beta):
    # Ensure x, alpha, beta are positive to avoid log(0) or negative values
    x_safe = np.maximum(x, 1e-12)
    alpha_safe = np.maximum(alpha, 1e-12)
    beta_safe = np.maximum(beta, 1e-12)
    # Compute gamma distribution PDF safely
    return np.exp(alpha_safe*np.log(beta_safe) - gammaln(alpha_safe) + (alpha_safe - 1)*np.log(x_safe) - beta_safe*x_safe)

# Apply the monkey patch after importing
if ACTUAL_DB_AVAILABLE:
    try:
        from simulation_envs.scripts.core.mainClasses import Subpopulation
        Subpopulation.gammaD = staticmethod(safe_gammaD)
        print("✅ Applied numerical safety patch to gammaD function")
    except ImportError:
        print("⚠️ Could not apply gammaD patch - mainClasses not available")


@db_session
def getReactor(simulTime=1, SimulSteps=5, dilution=0.0, volume=100):
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


@db_session
def run_direct_reactor_simulation(subpop_dict, media_dict, volume=15):
    """
    Direct reactor simulation using user-provided subpopulation counts and media concentrations.
    """
    if not ACTUAL_DB_AVAILABLE or DB is None:
        raise RuntimeError("Database not available. Cannot run direct simulation.")

    # Create metabolome with pH modeling
    metabolome = createMetabolome(DB, mediaName='kombucha_media', pHFunc=getpH)

    # Update metabolite concentrations based on user input
    for met_name, conc in media_dict.items():
        if met_name in metabolome.metD:
            metabolome.metD[met_name].concentration = max(0, conc)  # Ensure non-negative
        else:
            print(f"Warning: Metabolite '{met_name}' not found in kombucha_media. Skipping.")

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
            microbiome.subpopD[subpop_name].count = max(0, count)  # Ensure non-negative
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

    # Create Pulse and Reactor with safer parameters
    pulse = Pulse(metabolome_feed, microbiome_feed, 0, 600, 1000, 0, 0, 0.0, 0.0)
    reactor = Reactor(microbiome, metabolome, [pulse], volume)

    # Run simulation with error handling
    try:
        reactor.simulate()
        return reactor.makePlots(return_fig=True)
    except Exception as e:
        print(f"Error during simulation: {e}")
        raise


@db_session
def simple_reactor_test(simulTime=1, SimulSteps=5, dilution=0.5, volume=100, max_steps=3):
    """
    Simple reactor test with enhanced error handling and numerical stability
    """
    if not ACTUAL_DB_AVAILABLE or DB is None:
        raise RuntimeError("Database not available. Cannot run simple test.")
    
    print(f"🔍 Creating simple reactor test with params: simulTime={simulTime}, SimulSteps={SimulSteps}, dilution={dilution}, volume={volume}")
    
    try:
        # Create a basic reactor with safer parameters
        reactor = getReactor(simulTime=simulTime, SimulSteps=SimulSteps, dilution=dilution, volume=volume)
        
        # Validate initial state
        initial_states = reactor.get_states()
        if np.any(np.isnan(initial_states)) or np.any(np.isinf(initial_states)):
            print("⚠️ Warning: Invalid initial states detected, applying corrections...")
            # Fix any NaN or inf values
            initial_states = np.nan_to_num(initial_states, nan=0.0, posinf=1e6, neginf=0.0)
            reactor.update_states(initial_states)
        
        # Run simulation with error handling
        print("🔍 Running reactor simulation...")
        reactor.simulate()
        print("✅ Reactor simulation completed")
        
        # Extract results into a simple format
        results = []
        
        # Get final state data
        metabolite_names = reactor.metabolome.metabolites
        subpop_names = reactor.microbiome.subpops
        
        # Create data for each step (simulated progression)
        for step in range(max_steps):
            step_data = {
                'step': step,
                'pH': reactor.metabolome.pH,
                'volume': reactor.volume,
                'dilution': dilution
            }
            
            # Add metabolite concentrations with safety checks
            for i, met_name in enumerate(metabolite_names):
                conc = reactor.metabolome.metD[met_name].concentration
                # Ensure finite values
                if np.isnan(conc) or np.isinf(conc):
                    conc = 0.0
                step_data[f'metabolite_{met_name}'] = conc
            
            # Add subpopulation counts with safety checks
            for i, subpop_name in enumerate(subpop_names):
                count = reactor.microbiome.subpopD[subpop_name].count
                # Ensure finite, non-negative values
                if np.isnan(count) or np.isinf(count) or count < 0:
                    count = 0.0
                step_data[f'subpop_{subpop_name}'] = count
            
            results.append(step_data)
        
        print(f"✅ Simple test completed with {len(results)} data points")
        return results
        
    except Exception as e:
        print(f"❌ Error in simple_reactor_test: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal fallback data
        return [{
            'step': 0,
            'pH': 6.5,
            'volume': volume,
            'dilution': dilution,
            'error': str(e)
        }]


  

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
                 render_mode=None, max_dilution=param_max_dilution,
                 optimization_goals=None):
        #print("KombuchaGym: Initializing with enhanced safety checks")
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
        self.optimization_goals = optimization_goals or {}

        if not ACTUAL_DB_AVAILABLE or DB is None:
            raise RuntimeError("Cannot create KombuchaGym: Database or classes not available")

        try:
            self.metabolome_feed = createMetabolome(DB, mediaName='kombucha_media', pHFunc=getpH)
            species_ids = self._get_species_ids(DB)
            
            microbio_feed_dict = {}
            for sp_id in species_ids:
                bac = self._zero_count_bacteria(sp_id)
                if bac is not None:
                    microbio_feed_dict[sp_id] = bac
            
            self.microbiome_feed = Microbiome(microbio_feed_dict)
            for subpop in self.microbiome_feed.subpopD.values():
                subpop.count = 0

            self.reactor = getReactor(self.simulTime, self.SimulSteps, self.dilution, self.volume)
            #print("KombuchaGym: Reactor created")
            
            # Initial simulation with error handling
            try:
                self.reactor.simulate()
                #print("KombuchaGym: Initial simulation completed")
            except Exception as e:
                print(f"Warning: Initial simulation failed: {e}")
                # Continue with initialization anyway

            self.action_space = spaces.Discrete(2)
            obs = self._get_observation()
            self.observation_space = spaces.Box(
                low=np.zeros_like(obs, dtype=np.float32),
                high=np.full_like(obs, np.inf, dtype=np.float32),
                dtype=np.float32
            )
            self.episode_logs = []
            
        except Exception as e:
            print(f"Error in KombuchaGym initialization: {e}")
            raise

    def get_available_states(self):
        # Get metabolite names
        metabolite_states = list(self.reactor.metabolome.concentration.keys())
        
        # Get bacterial subpopulation names
        bacterial_states = list(self.reactor.microbiome.subpopD.keys())
        
        # Combine and return
        states = metabolite_states + bacterial_states
        print("🧪 get_available_states returns:", states, type(states))
        return states

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.dilution = param_dilution
        
        try:
            self.reactor = getReactor(simulTime=self.simulTime, SimulSteps=self.SimulSteps, 
                                   dilution=0.0, volume=self.volume)
            self.reactor.simulate()
        except Exception as e:
            print(f"Warning: Reset simulation failed: {e}")
        
        self.episode_logs = []
        return self._get_observation(), {}

    def _get_observation(self):
        try:
            states = self.reactor.get_states()
        except AttributeError:
            metabolite_states = list(self.reactor.metabolome.concentration)
            bacterial_states = [subpop.count for subpop in self.reactor.microbiome.subpopD.values()]
            states = metabolite_states + bacterial_states
        
        # Ensure all states are finite and non-negative
        states = np.array(states, dtype=np.float32)
        states = np.nan_to_num(states, nan=0.0, posinf=1e6, neginf=0.0)
        states = np.maximum(states, 0.0)  # Ensure non-negative
        
        return states

    def step(self, action):
        self.current_step += 1
        action_enum = KombuchaGymAction(action)

        if action_enum == KombuchaGymAction.INCREASE:
            self.dilution = min(self.max_dilution, self.dilution + self.unit_change)
        elif action_enum == KombuchaGymAction.DECREASE:
            self.dilution = max(0, self.dilution - self.unit_change)

        try:
            pulse = [Pulse(self.metabolome_feed, self.microbiome_feed, 0, self.simulTime,
                          self.SimulSteps, 0, 0, self.dilution, self.dilution)]
            self.reactor = Reactor(self.reactor.microbiome, self.reactor.metabolome, pulse, self.volume)
            self.reactor.simulate()
            
            subpop_counts = {k: max(0, v.count) for k, v in self.reactor.microbiome.subpopD.items()}
            current_pH = getpH(self.reactor.metabolome.concentration)
            metabolite_conc = dict(zip(self.reactor.metabolome.metabolites, self.reactor.metabolome.concentration))
            
            # Ensure all values are finite
            for k, v in metabolite_conc.items():
                if np.isnan(v) or np.isinf(v):
                    metabolite_conc[k] = 0.0
            
        except Exception as e:
            print(f"Warning: Step simulation failed: {e}")
            # Use previous state values with minimal changes
            subpop_counts = {k: 0.0 for k in self.reactor.microbiome.subpopD.keys()}
            current_pH = 6.5
            metabolite_conc = {k: 0.0 for k in self.reactor.metabolome.metabolites}

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
        reward = 0.0
        total_states = 0

        try:
            # Get current state vector from reactor
            metabolite_conc = dict(zip(self.reactor.metabolome.metabolites, self.reactor.metabolome.concentration))
            subpop_counts = {k: v.count for k, v in self.reactor.microbiome.subpopD.items()}
            current_pH = getpH(self.reactor.metabolome.concentration)

        except Exception:
            # Fallback: empty dicts
            metabolite_conc = {}
            subpop_counts = {}
            current_pH = None

        # Combine all states into one dict
        current_states = {}
        current_states.update(metabolite_conc)
        current_states.update(subpop_counts)
        if "pH" in self.optimization_goals:
            current_states["pH"] = current_pH

        for state, goal in (self.optimization_goals or {}).items():
            if state not in current_states:
                continue
            val = current_states[state]
            
            # Ensure finite value
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            
            total_states += 1

            # Calculate reward contribution for this state
            if "target" in goal:
                target = goal["target"]
                diff = abs(val - target)
                state_reward = max(0.0, 1 - diff / max(abs(target), 1e-3))
            elif "min" in goal and "max" in goal:
                min_val, max_val = goal["min"], goal["max"]
                if min_val <= val <= max_val:
                    state_reward = 1.0
                else:
                    diff = min(abs(val - min_val), abs(val - max_val))
                    range_size = max_val - min_val
                    state_reward = max(0.0, 1 - diff / max(range_size, 1e-3))
            else:
                state_reward = 0.0

            reward += state_reward

        # Average over number of states considered
        if total_states > 0:
            reward /= total_states
        else:
            reward = 0.0

        return reward

    def render(self):
        pass


def main(param_max_steps=5, param_dilution=0.5, param_volume=100,
         param_reward_scale=100, param_max_dilution=60, param_unit_change=0.01, 
         optimization_goals=None, simulTime=1, SimulSteps=5, **kwargs):
    """
    Main function with enhanced error handling
    """
    
    # Check if we should run simple test (faster for Streamlit)
    run_simple = kwargs.get('run_simple_test', True)
    
    if run_simple:
        print("🔍 Running simple reactor test...")
        return simple_reactor_test(
            simulTime=simulTime, 
            SimulSteps=SimulSteps, 
            dilution=param_dilution, 
            volume=param_volume,
            max_steps=param_max_steps
        )
    
    # Original full gym environment code
    print("🔍 Running full KombuchaGym environment...")
    try:
        env = KombuchaGym(
            max_steps=param_max_steps,
            max_dilution=param_max_dilution,
            dilution=param_dilution,
            volume=param_volume,
            reward_scale=param_reward_scale,
            param_unit_change=param_unit_change,
            optimization_goals=optimization_goals,
            simulTime=simulTime,
            SimulSteps=SimulSteps,
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
        
    except Exception as e:
        print(f"❌ Error in main function: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal fallback data
        return [{
            'step': 1,
            'reward': 0.0,
            'done': True,
            'dilution': param_dilution,
            'pH': 6.5,
            'error': str(e)
        }]

"""
def get_dynamic_states(**kwargs):
    print("🔍 Entered get_dynamic_states()")

    mapped_kwargs = {
        'max_steps': kwargs.get('param_max_steps', param_max_steps),
        'simulTime': kwargs.get('simulTime', param_simul_time),
        'SimulSteps': kwargs.get('SimulSteps', param_simul_steps),
        'dilution': kwargs.get('param_dilution', param_dilution),
        'volume': kwargs.get('param_volume', param_volume),
        'reward_scale': kwargs.get('param_reward_scale', param_reward_scale),
        'max_dilution': kwargs.get('param_max_dilution', param_max_dilution),
        'param_unit_change': kwargs.get('param_unit_change', param_unit_change),
        'param_target_ethanol': kwargs.get('param_target_ethanol', param_target_ethanol),
        'render_mode': kwargs.get('render_mode', None),
        'optimization_goals': kwargs.get('optimization_goals', None)
    }

    print("Mapped kwargs for KombuchaGym:", mapped_kwargs)

    try:
        env = KombuchaGym(**mapped_kwargs)
        print("KombuchaGym environment initialized.")

        states = env.get_available_states()
        print("🔍 Raw available_states type:", type(states))
        print("🔍 Raw available_states value:", states)

        # Handle numpy array or list
        if isinstance(states, (list, np.ndarray)):
            state_dict = {str(s): {} for s in list(states)}
            print("✅ Created state_dict:", state_dict)
            return state_dict
        else:
            raise TypeError(f"Unexpected type from get_available_states: {type(states)}")

    except Exception as e:
        print("❌ Failed to load dynamic states:", e)
        print("⚠️ Using fallback default states.")
        return {'ethanol': {}, 'pH': {}, 'subpop_B': {}}

"""
# Enhanced version with better debugging and validation

def get_dynamic_states(**kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Get dynamic states from the kombucha simulation environment.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of state names and their properties
    """
    print("🔍 Entered get_dynamic_states()")
    print(f"📥 Received kwargs: {kwargs}")

    # Define default parameters (you should define these at module level)
    default_params = {
        'param_max_steps': kwargs.get('param_max_steps', 100),
        'param_simul_time': kwargs.get('simulTime', 10.0),
        'param_simul_steps': kwargs.get('SimulSteps', 100),
        'param_dilution': kwargs.get('param_dilution', 0.1),
        'param_volume': kwargs.get('param_volume', 100),
        'param_reward_scale': kwargs.get('param_reward_scale', 1.0),
        'param_max_dilution': kwargs.get('param_max_dilution', 0.5),
        'param_unit_change': kwargs.get('param_unit_change', 0.01),
        'param_target_ethanol': kwargs.get('param_target_ethanol', 5.0),
    }

    mapped_kwargs = {
        'max_steps': kwargs.get('param_max_steps', default_params['param_max_steps']),
        'simulTime': kwargs.get('simulTime', default_params['param_simul_time']),
        'SimulSteps': kwargs.get('SimulSteps', default_params['param_simul_steps']),
        'dilution': kwargs.get('param_dilution', default_params['param_dilution']),
        'volume': kwargs.get('param_volume', default_params['param_volume']),
        'reward_scale': kwargs.get('param_reward_scale', default_params['param_reward_scale']),
        'max_dilution': kwargs.get('param_max_dilution', default_params['param_max_dilution']),
        'param_unit_change': kwargs.get('param_unit_change', default_params['param_unit_change']),
        'param_target_ethanol': kwargs.get('param_target_ethanol', default_params['param_target_ethanol']),
        'render_mode': kwargs.get('render_mode', None),
        'optimization_goals': kwargs.get('optimization_goals', None)
    }

    print(f"🔧 Mapped kwargs for KombuchaGym: {mapped_kwargs}")

    try:
        # Import KombuchaGym here to avoid import issues
        #from your_module import KombuchaGym  # Replace 'your_module' with actual module name
        
        env = KombuchaGym(**mapped_kwargs)
        print("✅ KombuchaGym environment initialized.")

        # Get the raw states from the environment
        raw_states: Union[List[str], Dict[str, Any], np.ndarray] = env.get_available_states()
        print(f"🔍 Raw states from get_available_states(): {raw_states}")
        print(f"🔍 Raw states type: {type(raw_states)}")
        
        # Validate and process the states with proper type checking
        if isinstance(raw_states, list):
            print("📝 Processing list of states...")
            state_dict: Dict[str, Dict[str, Any]] = {}
            for state in raw_states:
                state_name = str(state).strip()
                if state_name:  # Only add non-empty state names
                    state_dict[state_name] = {
                        'min': 0.0,
                        'max': 100.0,
                        'default': 50.0,
                        'unit': 'units'
                    }
            print(f"✅ Created state_dict from list: {state_dict}")
            return state_dict
            
        elif isinstance(raw_states, dict):
            print("📝 Processing dictionary of states...")
            # If it's already a dict, ensure it has the right structure
            processed_states: Dict[str, Dict[str, Any]] = {}
            for key, value in raw_states.items():
                state_name = str(key).strip()
                if state_name:
                    if isinstance(value, dict) and value:
                        # Keep existing structure if it has data
                        processed_states[state_name] = value
                    else:
                        # Add default structure if empty
                        processed_states[state_name] = {
                            'min': 0.0,
                            'max': 100.0,
                            'default': 50.0,
                            'unit': 'units'
                        }
            print(f"✅ Processed state_dict from dict: {processed_states}")
            return processed_states
            
        elif isinstance(raw_states, np.ndarray):
            print("📝 Processing numpy array of states...")
            state_list = raw_states.tolist()
            state_dict = {}
            for state in state_list:
                state_name = str(state).strip()
                if state_name:
                    state_dict[state_name] = {
                        'min': 0.0,
                        'max': 100.0,
                        'default': 50.0,
                        'unit': 'units'
                    }
            print(f"✅ Created state_dict from numpy array: {state_dict}")
            return state_dict
            
        else:
            print(f"⚠️ Unexpected type from get_available_states: {type(raw_states)}")
            print(f"⚠️ Raw value: {raw_states}")
            # This should not be reached if your function is working correctly
            raise TypeError(f"Unexpected type from get_available_states: {type(raw_states)}")

    except Exception as e:
        print(f"❌ Exception in get_dynamic_states: {e}")
        print(f"📍 Full traceback: {traceback.format_exc()}")
        # Re-raise the exception instead of returning fallback
        raise e



if __name__ == "__main__":
    # Test simple version by default
    results = main(param_max_steps=3, SimulSteps=5, run_simple_test=True)
    print(f"Simulation finished with {len(results)} steps.")
