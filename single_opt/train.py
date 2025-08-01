import os
import sys
sys.path.append(os.getcwd())

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, vec_check_nan
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from khrylib.utils import *
from PIL import Image
import shutil
import numpy as np
from argparse import ArgumentParser


# Import the custom environment
from single_opt.envs.walker import WalkerEnv

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i <= 1 or i == 4 or i >= 7 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

def train(log_path, model_path):
    # Create log and model directories
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Create the environment
    env = DummyVecEnv([lambda: WalkerEnv()])
    env = vec_check_nan.VecCheckNan(env, raise_exception=True)

    # Setup PPO model
    # For a complex task like this, a larger n_steps and a suitable learning rate are beneficial.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_path,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        n_epochs=10,
        gae_lambda=0.95,
        vf_coef=0.5,
    )
    
    # Setup a callback to save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=model_path,
        name_prefix='ppo_walker'
    )
    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    # Train the model
    total_timesteps = 2_000_000
    print("--- Starting Training ---")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    
    # Save the final model
    final_model_path = os.path.join(model_path, "ppo_walker_final")
    model.save(final_model_path)
    print(f"--- Training Finished. Final model saved to {final_model_path} ---")
    env.close()

def visualize(env, model_path, save_video=False, max_num_frames=1000):
    fr = 0
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_path,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        n_epochs=10,
        gae_lambda=0.95,
        vf_coef=0.5,
    )
    state = env.reset()
    for t in range(10000):
        action, _ = model.predict(state, deterministic=True)
        next_state, env_reward, terminated, truncated, info = env.step(action)
        if save_video:
            frame = env.render(mode='rgb_array')
            frame_dir = f'out/videos/single_frames'
            os.makedirs(frame_dir, exist_ok=True)
            img = Image.fromarray(frame)
            img.save(f'{frame_dir}/%04d.png' % fr)
            fr += 1
            if fr >= max_num_frames:
                break
        if terminated or truncated:
            break
        state = next_state

    if save_video:
        frame_dir = f'out/videos/single_frames'
        save_video_ffmpeg(f'{frame_dir}/%04d.png', f'out/videos/single.mp4', fps=30)
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)


if __name__ == "__main__":
    log_path = 'single/logs'
    model_path = 'single/models'
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    args = parser.parse_args()
    if args.train:
        train(log_path, model_path)
    elif args.test:
        visualize(WalkerEnv(), model_path, save_video=True)
    else:
        print("Please specify --train or --test")
        exit(1)