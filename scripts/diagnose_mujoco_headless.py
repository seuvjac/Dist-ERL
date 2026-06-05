#!/usr/bin/env python3
"""Quick headless MuJoCo smoke test (run in tmux before long benchmarks)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.environment import apply_headless_mujoco_runtime, make_env, resolve_gym_env_id


def main():
    apply_headless_mujoco_runtime()
    env_id = os.environ.get('TEST_ENV', 'HalfCheetah-v2')
    gym_id = resolve_gym_env_id(env_id)
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} gym_id={gym_id}", flush=True)
    t0 = time.time()
    env = make_env(env_id, max_episode_steps=100)
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        obs, _, done, trunc, _ = env.step(env.action_space.sample())
        if done or trunc:
            break
    env.close()
    print(f"OK reset+10 steps in {time.time() - t0:.2f}s", flush=True)


if __name__ == '__main__':
    main()
