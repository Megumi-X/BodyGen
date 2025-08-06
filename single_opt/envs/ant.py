from matplotlib.pylab import f
import numpy as np
from gym import utils
from gym import spaces
from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
from khrylib.robot.xml_robot import Robot
from khrylib.utils import get_single_body_qposaddr, get_graph_fc_edges
from khrylib.utils.transformation import quaternion_matrix
import mujoco_py
from gym.spaces import Box, MultiBinary


class AntSingleReconfigEnv(MujocoEnv, utils.EzPickle):
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
        xml_path = "assets/mujoco_envs/ant_single_reconfig.xml"
        
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.stage = 'reconfig'

        # Observation space
        obs_size = self._get_obs().shape[-1]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        
        # Action space
        lows = np.concatenate([
            np.full(self.model.nu, -1.0), # control 部分的下限
            np.full(4, 0.0)               # reconfig 部分的下限
        ])
        highs = np.concatenate([
            np.full(self.model.nu, 1.0), # control 部分的上限
            np.full(4, 1.0)              # reconfig 部分的上限
        ])
            
        self.step_count = 0
        self.max_episode_steps = 1000

        self.actuator_masks = None
        self.reconfig_action_ratio = 50
        self.init_joint_ranges = []

        MujocoEnv.__init__(self, xml_path, self.frame_skip)
        utils.EzPickle.__init__(self)
        self.action_space = Box(low=lows, high=highs, dtype=np.float64) 
         

    def reconfig_fix(self):
        for i in range(4):
            joint_id = self.model.actuator_trnid[i, 0]
            assert self.model.joint_names[joint_id] == f"{i + 1}_joint"
            if self.actuator_masks[i] == 1:
                qpos_address = self.model.jnt_qposadr[joint_id]
                current_joint_pos = self.sim.data.qpos[qpos_address]
                self.model.jnt_range[joint_id] = [current_joint_pos - 5e-2, current_joint_pos + 5e-2]

    def reconfig_release(self):
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            if joint_id < 0:
                continue
            self.model.jnt_range[joint_id] = [-np.pi, np.pi]

    def set_reconfig(self, reconfig_params):
        self.reconfig_release()
        self.actuator_masks = np.zeros(self.model.nu)
        for i in range(4):
            if reconfig_params[i] > 0.5:
                self.actuator_masks[i] = 1
            else:
                self.actuator_masks[i] = 0
        self.reconfig_fix()
        return True        

    def step(self, a, train=True):
        if not self.is_inited:
            return self._get_obs(), 0, False, False, {'use_transform_action': False, 'stage': 'execution'}

        self.cur_t += 1
        # reconfig stage
        if self.stage == 'reconfig':
            reconfig_a = a[self.model.nu:]
            self.set_reconfig(reconfig_a)
            if self.control_nsteps == 0:
                succ = self.transit_execution()
            else:
                succ = self.transit_execution_running()
            if not succ:
                if train:
                    return self._get_obs(), 0.0, True, {'use_transform_action': False, 'stage': 'reconfig'}
                else:
                    return self._get_obs(), 0.0, True, False, {'use_transform_action': False, 'stage': 'reconfig'}

            ob = self._get_obs()
            reward = 0
            termination = truncation = False
            if train:
                return ob, reward, termination or truncation, {'use_transform_action': False, 'stage': 'reconfig'}
            else:
                return ob, reward, termination, truncation, {'use_transform_action': False, 'stage': 'reconfig'}
        # execution stage
        else:
            self.control_nsteps += 1
            ctrl = a[:self.model.nu] * (np.ones_like(self.actuator_masks) - self.actuator_masks)
            ctrl_cost_coeff = 1e-4
            xposbefore = self.get_body_com("0")[0]

            try:
                self.do_simulation(ctrl, self.frame_skip)
            except:
                print(self.cur_xml_str)
                if train:
                    return self._get_obs(), 0, True, {'use_transform_action': False, 'stage': 'execution'}
                else:
                    return self._get_obs(), 0, True, False, {'use_transform_action': False, 'stage': 'execution'}

            xposafter = self.get_body_com("0")[0]
            reward_fwd = (xposafter - xposbefore) / self.dt
            reward_ctrl = - ctrl_cost_coeff * np.square(ctrl).mean()
            reward = reward_fwd + reward_ctrl

            s = self.state_vector()
            height = s[2]
            zdir = quaternion_matrix(s[3:7])[:3, 2]
            ang = np.arccos(zdir[2])
            min_height = 0
            max_height = 3
            max_ang = 180
            max_nsteps = 1000
            termination = not (np.isfinite(s).all() and (height > min_height) and (height < max_height) and (abs(ang) < np.deg2rad(max_ang)))
            truncation = not (self.control_nsteps < max_nsteps)
            ob = self._get_obs()
            if self.control_nsteps % self.reconfig_action_ratio == 0:
                self.transit_reconfig()
            if train:
                return ob, reward, termination or truncation, {'use_transform_action': False, 'stage': 'execution'}
            else:
                return ob, reward, termination, truncation, {'use_transform_action': False, 'stage': 'execution'}

    def transit_reconfig(self):
        self.stage = 'reconfig'

    def transit_execution(self):
        self.stage = 'execution'
        self.control_nsteps = 0
        try:
            self.reset_state(True)
        except:
            print(self.cur_xml_str)
            return False
        return True
    
    def transit_execution_running(self):
        self.stage = 'execution'
        return True

    def _get_obs(self):
        position = self.sim.data.qpos.flat[:]
        velocity = self.sim.data.qvel.flat[:]
        stage = 0 if self.stage == 'reconfig' else 1
        obs = np.concatenate((position, velocity, [stage])).astype(np.float64)
        return obs

    def reset_state(self, add_noise):
        if add_noise:
            qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        else:
            qpos = self.init_qpos
            qvel = self.init_qvel
        qpos[2] = 0.4
        self.set_state(qpos, qvel)

    def reset_model(self):
        self.control_nsteps = 0
        self.stage = 'reconfig'
        self.cur_t = 0
        self.reset_state(False)
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = 10
        # self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.lookat[:2] = self.data.qpos[:2] 
        self.viewer.cam.elevation = -10
        self.viewer.cam.azimuth = 110