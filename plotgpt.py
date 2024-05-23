import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
from src.env import EvacuationEnv


model_paths = {
    'model1': 'test_n-100_lr-0.0003_gamma-0.99_s-gra_a-2_ss-0.01_vr-0.001_20-May-12-15-16.zip',
    'model2': 'test_n-100_lr-0.0003_gamma-0.99_s-gra_a-2_ss-0.01_vr-0.001_20-May-11-49-56.zip',
    #'model3': 'path/to/model3.zip'
}

models = {name: PPO.load(path) for name, path in model_paths.items()}


def run_simulation(model, env, n_episodes):
    stats = {'escaped': [], 'exiting': [], 'following': []}  # Adjust based on needed stats
    for _ in tqdm(range(n_episodes)):
        obs = env.reset()
        terminated, truncated = False, False
        episode_stats = []

        while not (terminated or truncated):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_stats.append(env.pedestrians.status_stats)  # Assuming a similar structure

        for key in stats.keys():
            # Aggregate stats here (mean, count, etc.), example for 'escaped'
            stats[key].append(sum(data[key] for data in episode_stats))

    return stats



env_settings = {'number_of_pedestrians': 60, 'width': 10, 'height': 10}
env = EvacuationEnv(**env_settings)
n_episodes = 100  # Number of simulations per model

all_stats = {}
for name, model in models.items():
    print(f"Running simulations for {name}")
    all_stats[name] = run_simulation(model, env, n_episodes)


plt.figure(figsize=(10, 8))
for name, stats in all_stats.items():
    plt.plot(np.mean(stats['escaped'], axis=0), label=f'{name} Escape Rate')

plt.title("Model Comparison of Escape Rates")
plt.xlabel("Time (timesteps)")
plt.ylabel("Average Number Escaped")
plt.legend()
plt.show()


plt.savefig('path/to/save/escape_rates_comparison.png')
