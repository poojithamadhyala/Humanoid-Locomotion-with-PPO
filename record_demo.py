import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np
import os

os.makedirs("./videos", exist_ok=True)

# Load model + normalization stats
model = PPO.load("humanoid_ppo_final")
vec_env = DummyVecEnv([lambda: gym.make("Humanoid-v5", render_mode="rgb_array")])
vec_env = VecNormalize.load("vec_normalize_final.pkl", vec_env)
vec_env.training = False       # don't update stats during demo
vec_env.norm_reward = False    # don't normalize rewards

# Collect frames manually
frames = []
obs = vec_env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    frame = vec_env.render()
    frames.append(frame)
    if done[0]:
        obs = vec_env.reset()

vec_env.close()

# Save as MP4
import imageio
imageio.mimwrite("./videos/humanoid_demo.mp4", frames, fps=30)
print(f"✅ Saved {len(frames)} frames to ./videos/humanoid_demo.mp4")
