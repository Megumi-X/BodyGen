from collections import defaultdict
from khrylib.utils.torch import LongTensor
import torch.nn as nn
from khrylib.rl.core.distributions import Categorical, DiagGaussian
from khrylib.rl.core.policy import Policy
from khrylib.rl.core.running_norm import RunningNorm
from khrylib.models.mlp import MLP
from khrylib.utils.math import *
from design_opt.utils.tools import *
from design_opt.models.transformer import TransformerSimple
from torch.nn.utils.rnn import pad_sequence

import logging


class BodyGenPolicyNeoReconfig(Policy):
    def __init__(self, cfg, agent):
        super().__init__()
        self.type = 'gaussian'
        self.cfg = cfg
        self.agent = agent
        self.attr_fixed_dim = agent.attr_fixed_dim
        self.sim_obs_dim = agent.sim_obs_dim
        self.attr_design_dim = agent.attr_design_dim
        self.control_state_dim = self.attr_fixed_dim + self.sim_obs_dim + self.attr_design_dim
        self.skel_state_dim = self.attr_fixed_dim + self.attr_design_dim
        self.attr_state_dim = self.attr_fixed_dim + self.attr_design_dim
        self.skel_action_dim = agent.skel_num_action
        self.base_action_dim = agent.control_action_dim
        self.control_action_dim = agent.control_action_dim * 2
        self.reconfig_action_dim = agent.control_action_dim * 3
        self.attr_action_dim = agent.attr_design_dim
        self.action_dim = self.reconfig_action_dim + self.control_action_dim + self.attr_action_dim + 1
        self.skel_uniform_prob = cfg.get('skel_uniform_prob', 0.0)

        # skeleton transform
        if not agent.cfg.uni_obs_norm:
            self.skel_norm = RunningNorm(self.attr_state_dim)
        else:
            self.skel_norm = None
        cur_dim = self.attr_state_dim
        
        self.skel_transformer = TransformerSimple(cur_dim, cfg['skel_transformer_specs'])
        cur_dim = self.skel_transformer.out_dim
        
        if 'skel_mlp' in cfg:
            self.skel_mlp = MLP(cur_dim, cfg['skel_mlp'], cfg['htype'])
            cur_dim = self.skel_mlp.out_dim
        else:
            self.skel_mlp = None
        
        self.skel_action_logits = nn.Linear(cur_dim, self.skel_action_dim)

        # attribute transform
        if not agent.cfg.uni_obs_norm:
            self.attr_norm = RunningNorm(self.skel_state_dim) if cfg.get('attr_norm', True) else None
        else:
            self.attr_norm = None
        cur_dim = self.skel_state_dim
        
        self.attr_transformer = TransformerSimple(cur_dim, cfg['attr_transformer_specs'])
        cur_dim = self.attr_transformer.out_dim
        
        if 'attr_mlp' in cfg:
            self.attr_mlp = MLP(cur_dim, cfg['attr_mlp'], cfg['htype'])
            cur_dim = self.attr_mlp.out_dim
        else:
            self.attr_mlp = None
        
        self.attr_action_mean = nn.Linear(cur_dim, self.attr_action_dim)
        init_fc_weights(self.attr_action_mean)
        self.attr_action_log_std = nn.Parameter(torch.ones(1, self.attr_action_dim) * cfg['attr_log_std'], requires_grad=not cfg['fix_attr_std'])

        # reconfig design
        if not agent.cfg.uni_obs_norm:
            self.reconfig_norm = RunningNorm(self.control_state_dim)
        else:
            self.reconfig_norm = None
        cur_dim = self.control_state_dim

        self.reconfig_transformer = TransformerSimple(cur_dim, cfg['reconfig_transformer_specs'])
        cur_dim = self.reconfig_transformer.out_dim
        if 'reconfig_mlp' in cfg:
            self.reconfig_mlp = MLP(cur_dim, cfg['reconfig_mlp'], cfg['htype'])
            cur_dim = self.reconfig_mlp.out_dim
        else:
            self.reconfig_mlp = None
        
        self.reconfig_action_mean = nn.Linear(cur_dim, self.reconfig_action_dim)
        init_fc_weights(self.reconfig_action_mean)
        self.reconfig_action_log_std = nn.Parameter(torch.ones(1, self.reconfig_action_dim) * cfg['reconfig_log_std'], requires_grad=not cfg['fix_reconfig_std'])

        # execution
        if not agent.cfg.uni_obs_norm:
            self.control_norm = RunningNorm(self.control_state_dim)
        else:
            self.control_norm = None
        cur_dim = self.control_state_dim
        
        self.control_transformer = TransformerSimple(cur_dim, cfg['control_transformer_specs'])
        cur_dim = self.control_transformer.out_dim
        
        if 'control_mlp' in cfg:
            self.control_mlp = MLP(cur_dim, cfg['control_mlp'], cfg['htype'])
            cur_dim = self.control_mlp.out_dim
        else:
            self.control_mlp = None
            
        self.control_action_mean = nn.Linear(cur_dim, self.control_action_dim)
        init_fc_weights(self.control_action_mean)
        self.control_action_log_std = nn.Parameter(torch.ones(1, self.control_action_dim) * cfg['control_log_std'], requires_grad=not cfg['fix_control_std'])

    def get_padded_obs(self, obs, num_nodes, distances, lapPE, body_ind):
        # obs [B, L_i, D]
        # padded_obs [B, L_m, D]
        # num_nodes [B,]
        padded_obs = pad_sequence(obs, batch_first=True, padding_value=0)
        
        B, L_max, D = padded_obs.shape
        
        range_tensor = torch.arange(L_max, device=padded_obs.device).expand(B, L_max) # [B, L_max]
        num_nodes_expanded = torch.tensor(num_nodes, device=padded_obs.device).view(B, 1) # [B, 1]
        
        padding_mask = range_tensor < num_nodes_expanded
        
        padded_distances = torch.zeros((B, L_max, L_max), device=padded_obs.device)
        for i, dis in enumerate(distances):
            L_i = dis.shape[0]
            padded_distances[i, :L_i, :L_i] = dis
           
        k = lapPE[0].shape[1]
        padded_lapPE = torch.zeros((B, L_max, k), device=padded_obs.device)
        for i, lap in enumerate(lapPE):
            L_i = lap.shape[0]
            padded_lapPE[i, :L_i, :] = lap
            
        padded_body_ind = torch.zeros((B, L_max), dtype=int, device=padded_obs.device)
        for i, ind in enumerate(body_ind):
            L_i = ind.shape[0]
            padded_body_ind[i, :L_i] = ind
        
        return padded_obs, padding_mask, padded_distances, padded_lapPE, padded_body_ind
    
    def batch_data(self, x):
        obs, edges, use_transform_action, num_nodes, body_ind, body_depths, body_heights, distances, lapPE = zip(*x)
        
        use_transform_action = np.concatenate(use_transform_action)
        num_nodes = np.concatenate(num_nodes)
        edges_new = torch.cat(edges, dim=1)
        num_nodes_cum = np.cumsum(num_nodes)

        padded_obs, padding_mask, padded_distances, padded_lapPE, padded_body_ind = self.get_padded_obs(obs, num_nodes, distances, lapPE, body_ind)        
        obs = torch.cat(obs)
        
        body_ind = torch.cat(body_ind)
        body_depths = torch.from_numpy(np.concatenate(body_depths))
        body_heights = torch.from_numpy(np.concatenate(body_heights))
        
        if len(x) > 1:
            repeat_num = [x.shape[1] for x in edges[1:]]
            e_offset = np.repeat(num_nodes_cum[:-1], repeat_num)
            e_offset = torch.tensor(e_offset, device=obs.device)
            edges_new[:, -e_offset.shape[0]:] += e_offset
            
        transformer_obs = {
            "padding_mask": padding_mask,
            "padded_obs": padded_obs,
            "padded_distances": padded_distances,
            "padded_lapPE": padded_lapPE,
            "padded_body_ind": padded_body_ind
        }
        
        return obs, edges_new, use_transform_action, num_nodes, num_nodes_cum, body_ind, body_depths, body_heights, transformer_obs

    def forward(self, x):
        stages = ['skel_trans', 'attr_trans', 'reconfig_design', 'execution']
        x_dict = defaultdict(list)
        node_design_mask = defaultdict(list)
        design_mask = defaultdict(list)
        total_num_nodes = 0
        for i, x_i in enumerate(x):
            num = x_i[3].item()
            cur_stage = stages[int(x_i[2].item())]
            x_dict[cur_stage].append(x_i)
            for stage in stages:
                node_design_mask[stage] += [cur_stage == stage] * num
                design_mask[stage].append(cur_stage == stage)
            total_num_nodes += num
        for stage in stages:
            node_design_mask[stage] = torch.BoolTensor(node_design_mask[stage])
            design_mask[stage] = torch.BoolTensor(design_mask[stage])

        # execution
        if len(x_dict['execution']) > 0:
            obs, edges, _, num_nodes, num_nodes_cum_control, body_ind, body_depths, body_heights, transformer_obs = self.batch_data(x_dict['execution'])
            
            if torch.isnan(obs).any():
                print("NaNs appeared in the observation!")
                import pdb; pdb.set_trace()

            if torch.isnan(transformer_obs['padded_obs']).any():
                print("NaNs appeared in the transformer_obs padded observation!")
                import pdb; pdb.set_trace()
            elif torch.isnan(transformer_obs['padded_distances']).any():
                print("NaNs appeared in the transformer_obs padded distances!")
                import pdb; pdb.set_trace()
            elif torch.isnan(transformer_obs['padded_lapPE']).any():
                print("NaNs appeared in the transformer_obs padded lapPE!")
                import pdb; pdb.set_trace()
            elif torch.isnan(transformer_obs['padded_body_ind']).any():
                print("NaNs appeared in the transformer_obs padded body_ind!")
                import pdb; pdb.set_trace()
            elif torch.isnan(transformer_obs['padding_mask']).any():
                print("NaNs appeared in the transformer_obs padding_mask!")
                import pdb; pdb.set_trace()

            if self.control_norm is not None:
                x = self.control_norm(obs)
            else:
                x = obs
            
            x = self.control_transformer(transformer_obs)
            if torch.isnan(x).any():
                print("NaNs appeared after the transformer layer!")
                import pdb; pdb.set_trace()

            if self.control_mlp is not None:
                x = self.control_mlp(x)
                if torch.isnan(x).any():
                    print("NaNs appeared after the MLP layer!")
                    import pdb; pdb.set_trace()

            control_action_mean = self.control_action_mean(x)
            if torch.isnan(control_action_mean).any():
                print("NaNs appeared after the final action_mean layer!")
                import pdb; pdb.set_trace()

            control_action_std = self.control_action_log_std.expand_as(control_action_mean).clamp(-20, 5).exp()
            control_dist = DiagGaussian(control_action_mean, control_action_std)

        else:
            num_nodes_cum_control = None
            control_dist = None

        # reconfig design
        if len(x_dict['reconfig_design']) > 0:
            obs, edges, _, num_nodes, num_nodes_cum_reconfig, body_ind, body_depths, body_heights, transformer_obs = self.batch_data(x_dict['reconfig_design'])

            if self.reconfig_norm is not None:
                x = self.reconfig_norm(obs)
            else:
                x = obs
            
            x = self.reconfig_transformer(transformer_obs)

            if self.reconfig_mlp is not None:
                x = self.reconfig_mlp(x)
            
            reconfig_action_mean = self.reconfig_action_mean(x)
            reconfig_action_std = self.reconfig_action_log_std.expand_as(reconfig_action_mean).exp()
            reconfig_dist = DiagGaussian(reconfig_action_mean, reconfig_action_std)
        else:
            num_nodes_cum_reconfig = None
            reconfig_dist = None    

        # attribute transform
        if len(x_dict['attr_trans']) > 0:
            obs, edges, _, num_nodes, num_nodes_cum_design, body_ind, body_depths, body_heights, transformer_obs = self.batch_data(x_dict['attr_trans'])
            obs = torch.cat((obs[:, :self.attr_fixed_dim], obs[:, -self.attr_design_dim:]), dim=-1)
            transformer_obs['padded_obs'] = torch.cat((transformer_obs['padded_obs'][:, :, :self.attr_fixed_dim], transformer_obs['padded_obs'][:, :, -self.attr_design_dim:]), dim=-1)
            
            if self.attr_norm is not None:
                x = self.attr_norm(obs)
            else:
                x = obs
            
            x = self.attr_transformer(transformer_obs)
            
            if self.attr_mlp is not None:
                x = self.attr_mlp(x)
            
            attr_action_mean = self.attr_action_mean(x)
                
            attr_action_std = self.attr_action_log_std.expand_as(attr_action_mean).exp()
            attr_dist = DiagGaussian(attr_action_mean, attr_action_std)
        else:
            num_nodes_cum_design = None
            attr_dist = None

        # skeleleton transform
        if len(x_dict['skel_trans']) > 0:
            obs, edges, _, num_nodes, num_nodes_cum_skel, body_ind, body_depths, body_heights, transformer_obs = self.batch_data(x_dict['skel_trans'])
            obs = torch.cat((obs[:, :self.attr_fixed_dim], obs[:, -self.attr_design_dim:]), dim=-1)
            transformer_obs['padded_obs'] = torch.cat((transformer_obs['padded_obs'][:, :, :self.attr_fixed_dim], transformer_obs['padded_obs'][:, :, -self.attr_design_dim:]), dim=-1)
            
            if self.skel_norm is not None:
                x = self.skel_norm(obs)
            else:
                x = obs
            
            x = self.skel_transformer(transformer_obs)
            
            if self.skel_mlp is not None:
                x = self.skel_mlp(x)
            
            skel_logits = self.skel_action_logits(x)
                
            skel_dist = Categorical(logits=skel_logits, uniform_prob=self.skel_uniform_prob)
        else:
            num_nodes_cum_skel = None
            skel_dist = None

        return control_dist, reconfig_dist, attr_dist, skel_dist, node_design_mask, design_mask, total_num_nodes, num_nodes_cum_control, num_nodes_cum_reconfig, num_nodes_cum_design, num_nodes_cum_skel, x[0][0].device

    def select_action(self, x, mean_action=False):
        
        control_dist, reconfig_dist, attr_dist, skel_dist, node_design_mask, _, total_num_nodes, _, _, _, _, device = self.forward(x)
        if control_dist is not None:
            control_action = control_dist.mean_sample() if mean_action else control_dist.sample()
        else:
            control_action = None
        
        if reconfig_dist is not None:
            reconfig_action = reconfig_dist.mean_sample() if mean_action else reconfig_dist.sample()
        else:
            reconfig_action = None

        if attr_dist is not None:
            attr_action = attr_dist.mean_sample() if mean_action else attr_dist.sample()
        else:
            attr_action = None

        if skel_dist is not None:
            skel_action = skel_dist.mean_sample() if mean_action else skel_dist.sample()
        else:
            skel_action = None

        action = torch.zeros(total_num_nodes, self.action_dim).to(device)
        if control_action is not None:
            action[node_design_mask['execution'], :2 * self.base_action_dim] = control_action
        if reconfig_action is not None:
            action[node_design_mask['reconfig_design'], 2 * self.base_action_dim: 5 * self.base_action_dim] = reconfig_action
        if attr_action is not None:
            action[node_design_mask['attr_trans'], 5 * self.base_action_dim: -1] = attr_action
        if skel_action is not None:
            action[node_design_mask['skel_trans'], [-1]] = skel_action.double()
        return action

    def get_log_prob(self, x, action):
        '''
        We perform joint-level entropy, rather than body-level entropy
        '''
        action = torch.cat(action)
        control_dist, reconfig_dist, attr_dist, skel_dist, node_design_mask, design_mask, total_num_nodes, num_nodes_cum_control, num_nodes_cum_reconfig, num_nodes_cum_design, num_nodes_cum_skel, device = self.forward(x)
        action_log_prob = torch.zeros(design_mask['execution'].shape[0], 1).to(device)
        # execution log prob
        if control_dist is not None:
            control_action = action[node_design_mask['execution'], :2 * self.base_action_dim]
            control_action_log_prob_nodes = control_dist.log_prob(control_action)
            control_action_log_prob_cum = torch.cumsum(control_action_log_prob_nodes, dim=0)
            control_action_log_prob_cum = control_action_log_prob_cum[torch.LongTensor(num_nodes_cum_control) - 1]
            control_action_log_prob = torch.cat([control_action_log_prob_cum[[0]], control_action_log_prob_cum[1:] - control_action_log_prob_cum[:-1]])
            action_log_prob[design_mask['execution']] = control_action_log_prob
        # reconfig log prob
        if reconfig_dist is not None:
            reconfig_action = action[node_design_mask['reconfig_design'], 2 * self.base_action_dim: 5 * self.base_action_dim]
            reconfig_action_log_prob_nodes = reconfig_dist.log_prob(reconfig_action)
            reconfig_action_log_prob_cum = torch.cumsum(reconfig_action_log_prob_nodes, dim=0)
            reconfig_action_log_prob_cum = reconfig_action_log_prob_cum[torch.LongTensor(num_nodes_cum_reconfig) - 1]
            reconfig_action_log_prob = torch.cat([reconfig_action_log_prob_cum[[0]], reconfig_action_log_prob_cum[1:] - reconfig_action_log_prob_cum[:-1]])
            action_log_prob[design_mask['reconfig_design']] = reconfig_action_log_prob
        # attribute transform log prob
        if attr_dist is not None:
            attr_action = action[node_design_mask['attr_trans'], 5 * self.base_action_dim:-1]
            attr_action_log_prob_nodes = attr_dist.log_prob(attr_action)
            attr_action_log_prob_cum = torch.cumsum(attr_action_log_prob_nodes, dim=0)
            attr_action_log_prob_cum = attr_action_log_prob_cum[torch.LongTensor(num_nodes_cum_design) - 1]
            attr_action_log_prob = torch.cat([attr_action_log_prob_cum[[0]], attr_action_log_prob_cum[1:] - attr_action_log_prob_cum[:-1]])
            action_log_prob[design_mask['attr_trans']] = attr_action_log_prob
        # skeleton transform log prob
        if skel_dist is not None:
            skel_action = action[node_design_mask['skel_trans'], [-1]]
            skel_action_log_prob_nodes = skel_dist.log_prob(skel_action)
            skel_action_log_prob_cum = torch.cumsum(skel_action_log_prob_nodes, dim=0)
            skel_action_log_prob_cum = skel_action_log_prob_cum[torch.LongTensor(num_nodes_cum_skel) - 1]
            skel_action_log_prob = torch.cat([skel_action_log_prob_cum[[0]], skel_action_log_prob_cum[1:] - skel_action_log_prob_cum[:-1]])
            action_log_prob[design_mask['skel_trans']] = skel_action_log_prob
        return action_log_prob


