import os
import sys
sys.path.append(os.getcwd())

from typing import Any, Callable, Dict, List, Optional, Union
import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, vec_check_nan, vec_normalize, sync_envs_normalization
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from khrylib.utils import *
from PIL import Image
import shutil
import numpy as np
from argparse import ArgumentParser


# Import the custom environment
from single_opt.envs.walker import WalkerEnv
from single_opt.envs.ant import AntSingleReconfigEnv, AntSingleEnv, AntSingleNeoReconfigEnv

current_env = AntSingleNeoReconfigEnv

class SaveVecNormalizeCallback(EvalCallback):
    def __init__(self, 
                 eval_env: Union[gym.Env, VecEnv],
                 train_env: VecEnv,
                 **kwargs):
        super().__init__(eval_env=eval_env, **kwargs)
        self.train_env = train_env

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_envs_normalization(self.training_env, self.eval_env)
        
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
    
def create_cosine_annealing_schedule(lr_initial: float, lr_final: float) -> Callable[[float], float]:
    def _scheduler(progress_remaining: float) -> float:
        progress_so_far = 1.0 - progress_remaining
        cosine_factor = 0.5 * (1 + np.cos(np.pi * progress_so_far))
        current_lr = lr_final + (lr_initial - lr_final) * cosine_factor
        return current_lr
    return _scheduler

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

    lr_schedule = create_cosine_annealing_schedule(
        lr_initial=1e-4,
        lr_final=5e-6
    )

    # Setup PPO model
    # For a complex task like this, a larger n_steps and a suitable learning rate are beneficial.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_path,
        n_steps=2048,
        batch_size=256,
        max_grad_norm=5,
        gamma=0.99,
        learning_rate=lr_schedule,
        ent_coef=0.012,
        clip_range=0.2,
        n_epochs=5,
        gae_lambda=0.97,
        vf_coef=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=torch.nn.ReLU
        ),
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
    total_timesteps = 30000000
    print("--- Starting Training ---")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    
    # Save the final model
    final_model_path = os.path.join(model_path, "ppo_walker_final")
    model.save(final_model_path)
    env.save(os.path.join(model_path, "vec_normalize.pkl"))
    print(f"--- Training Finished. Final model saved to {final_model_path} ---")
    env.close()

def visualize(model_path, max_num_frames=1000, env_name='ant_single_tunnel_1'):
    fr = 0  
    
    model_file = os.path.join(model_path, "best_model.zip") 
    stats_path = os.path.join(model_path, "vec_normalize.pkl")

    frame_dir = f'out/single_videos/single_frames'
    os.makedirs(frame_dir, exist_ok=True)
    env = DummyVecEnv([lambda: current_env(env_name=env_name, render_folder=frame_dir)])
    env = vec_normalize.VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    
    model = PPO.load(model_file, env=env)
    state = env.reset()
    
    for _ in range(10000):
        action, _ = model.predict(state, deterministic=True)
        next_state, env_reward, terminated, info = env.step(action)
        if fr >= max_num_frames:
            break
        if terminated:
            break
        state = next_state

    env.close()
    save_video_ffmpeg(f'{frame_dir}/%04d.png', f'out/single_videos/{env_name}.mp4', fps=30)
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--env', type=str, default='ant_single_tunnel_1', help='Environment name')
    args = parser.parse_args()
    log_path = f'single/logs/{args.env}-1'
    model_path = f'single/models/{args.env}-1'
    if args.train:
        train(log_path, model_path, env_name=args.env)
    elif args.test:
        visualize(model_path, env_name=args.env)
    else:
        print("Please specify --train or --test")
        exit(1)

    # fr = 0  
    
    # model_file = os.path.join(model_path, "best_model.zip") 
    # stats_path = os.path.join(model_path, "vec_normalize.pkl")

    # env_name = args.env
    # env = current_env(env_name=env_name)
    
    # for t in range(10):
    #     frame_dir = f'out/single_videos/single_frames'
    #     os.makedirs(frame_dir, exist_ok=True)
    #     succ, _ = env.reconfig(frame_dir)
    #     print(succ)

    # env.close()
    
    # frame_dir = f'out/single_videos/single_frames'
    # save_video_ffmpeg(f'{frame_dir}/%04d.png', f'out/single_videos/{env_name}.mp4', fps=30)
    # if os.path.exists(frame_dir):
    #     shutil.rmtree(frame_dir)    