import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os

# ── Config ──────────────────────────────────────────────────────────────
ENV_ID        = "Humanoid-v5"
TOTAL_STEPS   = 1_000_000   # ~1.5 hrs on Mac; reduce to 500k for faster results
N_ENVS        = 4           # parallel environments
LOG_DIR       = "./logs/"
SAVE_DIR      = "./checkpoints/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Environment ──────────────────────────────────────────────────────────
env  = make_vec_env(ENV_ID, n_envs=N_ENVS)
env  = VecNormalize(env, norm_obs=True, norm_reward=True)  # critical for humanoid

eval_env = make_vec_env(ENV_ID, n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

# ── Callbacks ────────────────────────────────────────────────────────────
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,
    log_path=LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=5,
    verbose=1
)

ckpt_cb = CheckpointCallback(
    save_freq=50_000,
    save_path=SAVE_DIR,
    name_prefix="humanoid_ppo"
)

# ── Model (PPO — same algorithm family Tesla uses for Optimus) ────────────
model = PPO(
    "MlpPolicy",
    env,
    # --- Core PPO hyperparameters ---
    n_steps=2048,          # steps collected per env before update
    batch_size=256,        # minibatch size
    n_epochs=10,           # gradient update passes per rollout
    learning_rate=3e-4,
    gamma=0.99,            # discount factor
    gae_lambda=0.95,       # GAE smoothing
    clip_range=0.2,        # PPO clipping epsilon
    ent_coef=0.01,         # entropy bonus — encourages exploration
    # --- Network ---
    policy_kwargs=dict(net_arch=[256, 256]),
    # --- Logging ---
    tensorboard_log=LOG_DIR,
    verbose=1,
)

print("🚀 Training started. Open TensorBoard to watch live:")
print("   tensorboard --logdir ./logs/\n")

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_cb, ckpt_cb],
    progress_bar=True
)

model.save("humanoid_ppo_final")
env.save("vec_normalize_final.pkl")
print("✅ Training complete!")
