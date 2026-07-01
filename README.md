# 🤖 I Taught a Robot to Walk!

![Robot Walking](results/demo.gif)

## What is this?

Imagine training a dog 🐶
- Sit correctly → give a treat ✅
- Jump on people → say no ❌
- Dog slowly learns what gets treats

I did the exact same thing — but with a robot!
- Robot walks forward → gets points ✅
- Robot falls over → loses points ❌
- Robot tries **1,000,000 times** until it figures it out

This teaching method is called **Reinforcement Learning (RL)**.
The robot doing the learning is called the **Agent**.

## Where does it practice?

In a **simulation** — like a video game world where the robot 
can fall, crash, and try again with zero real-world consequences.

I used **MuJoCo** as the simulation playground.

Tesla uses the same concept with **Isaac Lab** to train their 
**Optimus humanoid robot** — the only difference is they run 
1000s of robots practicing simultaneously on big GPUs.
Same idea. Bigger computer.

## How smart did it get?

| Metric | Start | End |
|--------|-------|-----|
| Score | ~50 | **605** |
| Steps before falling | 40 | **122** |
| Training time | - | **5 minutes** |
| Total attempts | - | **1,000,000** |

## The Algorithm — PPO

The "teaching strategy" used here is called 
**Proximal Policy Optimization (PPO)**.

Think of it as the robot asking itself after every fall:
> *"What small change to my movements would have gotten 
> me more points?"*

It adjusts a tiny bit each time. After a million tries — it walks.

PPO is the same algorithm family used in real humanoid 
robotics research today.

## Reward Function
This is the "points system" that tells the robot what good 
behaviour looks like:

| Behaviour | Points |
|-----------|--------|
| Moving forward fast | +1.25 per step |
| Staying upright | +5.0 per step |
| Using too much joint force | -0.1 (wasteful) |
| Explained variance | **0.936** (value network converged) |
| Slamming into the ground | -0.0005 |

Designing this reward function is the most important 
and creative part of the whole process.

## Run It Yourself
```bash
conda create -n rl_humanoid python=3.10
conda activate rl_humanoid
pip install gymnasium mujoco stable-baselines3 tensorboard imageio
python train.py
```

## What's Next?
- [x] Humanoid learns to walk 🚶
- [ ] 4-legged Ant learns to walk 🐜
- [ ] Compare two teaching methods (PPO vs SAC)
- [ ] Custom reward — teach it to walk efficiently 🔋
- [ ] Domain randomisation — prepare for real world 🌍

## Why Does This Matter?
Tesla, Boston Dynamics, Figure AI — they all train their 
robots this way. Just with more robots, more GPUs, and 
more complex tasks.

This project is the foundation of exactly that. 🚀
