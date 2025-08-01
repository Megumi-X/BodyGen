import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from gym import utils
from gym.spaces import Box
from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
import mujoco_py
from gym.utils import seeding

class WalkerEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        
        self.frame_skip = 4
        xml_path = "assets/single_mujoco_envs/walker.xml"
        
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        # Observation space
        obs_size = (self.model.nq) + self.model.nv
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        
        # Action space
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64)
        
        self.step_count = 0
        self.max_episode_steps = 1000

        MujocoEnv.__init__(self, xml_path, self.frame_skip)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        # Observation: joint positions (excluding root x-coord), and all velocities
        position = self.sim.data.qpos.flat[:]
        velocity = self.sim.data.qvel.flat[:]
        return np.concatenate((position, velocity)).astype(np.float64)

    def step(self, action):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]

        # Reward calculation
        reward = (posafter - posbefore) / self.dt
        terminated = not (np.isfinite(self.state_vector()).all() and (height > 0.0) and (height < 5.0))

        # Truncation condition
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, add_noise=True):
        if add_noise:
            qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        else:
            qpos = self.init_qpos
            qvel = self.init_qvel

        self.set_state(qpos, qvel)
        return self._get_obs()

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def close(self):
        if self.viewer:
            self.viewer = None
    
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
    
    def viewer_setup(self):
        self.viewer.cam.distance = 5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.lookat[0] = self.data.qpos[0] 
        self.viewer.cam.elevation = -10
        self.viewer.cam.azimuth = 110