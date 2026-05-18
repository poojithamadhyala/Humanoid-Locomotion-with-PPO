#  I Taught a Robot to Walk!

![Robot Walking](results/demo.gif)

## What is this?
You know how a baby learns to walk by trying, falling, and trying again?

That's exactly what I did here — except with a computer robot! 🍼

The robot starts knowing **absolutely nothing**.
It just falls flat on its face.

Then it tries millions of times...
Gets points when it walks forward ✅
Loses points when it falls ❌

After **1,000,000 tries** — it learned to walk! 🎉

## How smart did it get?
| Thing we measured | Result |
|---|---|
| Score at the start | ~50 |
| Score at the end | **605** |
| How long it stayed upright | **122 steps** |
| Time it took to learn | **5 minutes** |

## What's inside?
- `train.py` — the "teacher" that trains the robot
- `record_demo.py` — records a video of the robot walking
- `results/demo.gif` — the robot actually walking!

## Want to try it yourself?
```bash
conda create -n rl_humanoid python=3.10
conda activate rl_humanoid
pip install gymnasium mujoco stable-baselines3 tensorboard imageio
python train.py
```

## What's next?
- [x] Teach humanoid robot to walk 🚶
- [ ] Teach a 4-legged ant robot to walk 🐜
- [ ] Compare two different teaching methods
- [ ] Make the robot walk more efficiently 🔋
- [ ] Make it work in the real world 🌍

## Why does this matter?
Companies like **Tesla** are teaching their **Optimus robot** 
to do tasks the same way — just with bigger computers and 
more complex movements!

This is the foundation of exactly that. 🚀
