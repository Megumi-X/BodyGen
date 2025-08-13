import os
import sys
sys.path.append(os.getcwd())

from typing import Any, Callable, Dict, List, Optional, Union
import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, vec_check_nan, vec_normalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from khrylib.utils import *
from PIL import Image
import shutil
import numpy as np
from argparse import ArgumentParser


# Import the custom environment
from single_opt.envs.walker import WalkerEnv
from single_opt.envs.ant import AntSingleReconfigEnv, AntSingleEnv

current_env = AntSingleEnv

class SaveVecNormalizeCallback(EvalCallback):
    def __init__(self, 
                 eval_env: Union[gym.Env, VecEnv],
                 train_env: VecEnv,
                 **kwargs):
        super().__init__(eval_env=eval_env, **kwargs)
        self.train_env = train_env

    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        if continue_training is False:
            return False

        if self.last_mean_reward is not None and self.best_mean_reward == self.last_mean_reward:
            if self.n_calls % self.eval_freq == 0:
                print("New best model found!")
                
                stats_path = os.path.join(self.best_model_save_path, "vec_normalize.pkl")
                
                self.train_env.save(stats_path)
                print(f"Successfully synced and saved VecNormalize stats to {stats_path}")
                
        return True

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i <= 1 or i == 4 or i >= 7 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

def train(log_path, model_path, env_name):
    # Create log and model directories
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Create the environment
    env = DummyVecEnv([lambda: current_env(env_name=env_name)])
    env = vec_check_nan.VecCheckNan(env, raise_exception=True)
    env = vec_normalize.VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = DummyVecEnv([lambda: current_env(env_name=env_name)])
    eval_env = vec_check_nan.VecCheckNan(eval_env, raise_exception=True)
    eval_env = vec_normalize.VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.save(os.path.join(model_path, "vec_normalize.pkl"))

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
        learning_rate=5e-5,
        ent_coef=0.01,
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
    eval_callback = SaveVecNormalizeCallback(
        eval_env=eval_env,
        train_env=env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )

    # Train the model
    total_timesteps = 20000000
    print("--- Starting Training ---")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    
    # Save the final model
    final_model_path = os.path.join(model_path, "ppo_walker_final")
    model.save(final_model_path)
    env.save(os.path.join(model_path, "vec_normalize.pkl"))
    print(f"--- Training Finished. Final model saved to {final_model_path} ---")
    env.close()

def visualize(model_path, save_video=False, max_num_frames=1000, env_name='ant_single_tunnel_1'):
    fr = 0  
    
    model_file = os.path.join(model_path, "best_model.zip") 
    stats_path = os.path.join(model_path, "vec_normalize.pkl")

    # --- 2. 创建一个新的普通环境 ---
    env = DummyVecEnv([lambda: current_env(env_name=env_name)])
    env = vec_normalize.VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    
    model = PPO.load(os.path.join(model_path, "best_model.zip"), env=env)
    state = env.reset()
    
    for t in range(10000):
        action, _ = model.predict(state, deterministic=True)
        next_state, env_reward, terminated, info = env.step(action)
        if save_video:
            frame = env.render(mode='rgb_array')
            frame_dir = f'out/single_videos/single_frames'
            os.makedirs(frame_dir, exist_ok=True)
            img = Image.fromarray(frame)
            img.save(f'{frame_dir}/%04d.png' % fr)
            fr += 1
            if fr >= max_num_frames:
                break
        if terminated:
            break
        state = next_state

    env.close()
    
    if save_video:
        frame_dir = f'out/single_videos/single_frames'
        save_video_ffmpeg(f'{frame_dir}/%04d.png', f'out/single_videos/{env_name}.mp4', fps=30)
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--env', type=str, default='ant_single_tunnel_1', help='Environment name')
    args = parser.parse_args()
    log_path = f'single/logs/{args.env}-heavy'
    model_path = f'single/models/{args.env}-heavy'
    if args.train:
        train(log_path, model_path, env_name=args.env)
    elif args.test:
        visualize(model_path, save_video=True, env_name=args.env)
    else:
        print("Please specify --train or --test")
        exit(1)