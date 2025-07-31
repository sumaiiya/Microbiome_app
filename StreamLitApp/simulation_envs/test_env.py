import os
import sys
import numpy as np
import cProfile
import pstats

# Ensure base paths for simulation_envs structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts', 'db'))
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts', 'core'))

import kombucha_simulation as sim
from kombucha_simulation import KombuchaGym


def run_test():
    print("KombuchaGym: Initializing")
    env = KombuchaGym()
    obs, _ = env.reset()
    print("KombuchaGym: Reactor created")

    for i in range(10):
        print(f"Step {i+1}")
        action = env.action_space.sample()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        except Exception as e:
            print(f"Error during step {i+1}: {e}")
            break

        if np.isnan(obs).any():
            print(f"NaN detected in observation at step {i+1}")
            break

        print(f"  Reward: {reward:.4f}, Done: {done}")
        if done:
            print("Episode finished.")
            break

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_test()

    profiler.disable()
    print("\n--- Profiling Results (Top 20 by cumulative time) ---")
    stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
    stats.print_stats(20)
