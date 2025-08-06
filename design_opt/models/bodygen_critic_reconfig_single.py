import torch.nn as nn
import torch
import numpy as np
from khrylib.models.mlp import MLP
from collections import defaultdict
from khrylib.rl.core.running_norm import RunningNorm
from design_opt.utils.tools import *
from design_opt.models.transformer import TransformerSimple
from torch.nn.utils.rnn import pad_sequence

import logging


class BodyGenValueReconfigSingle(nn.Module):
    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.state_dim = agent.state_dim
        if not agent.cfg.uni_obs_norm:
            # self.norm = RunningNorm(self.state_dim)
            self.reconfig_norm = RunningNorm(self.state_dim)
            self.control_norm = RunningNorm(self.state_dim)
        else:
            # self.norm = None
            self.reconfig_norm = self.control_norm = None
        cur_dim = self.state_dim
        
        self.reconfig_transformer = TransformerSimple(cur_dim, cfg['transformer_specs'])
        self.control_transformer = TransformerSimple(cur_dim, cfg['transformer_specs'])
        cur_dim = self.control_transformer.out_dim
            
        if 'mlp' in cfg:
            # self.mlp = MLP(cur_dim, cfg['mlp'], cfg['htype'])
            self.reconfig_mlp = MLP(cur_dim, cfg['mlp'], cfg['htype'])
            self.control_mlp = MLP(cur_dim, cfg['mlp'], cfg['htype'])
            cur_dim = self.control_mlp.out_dim
        else:
            self.reconfig_mlp = self.control_mlp = None

        # self.value_head = nn.Linear(cur_dim, 1)
        self.reconfig_value_head = nn.Linear(cur_dim, 1)
        self.control_value_head = nn.Linear(cur_dim, 1)
        
        # init_fc_weights(self.value_head)
        init_fc_weights(self.reconfig_value_head)
        init_fc_weights(self.control_value_head)

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
        
        return obs, edges_new, use_transform_action, num_nodes, num_nodes_cum, transformer_obs
    
    def get_root_index(self, x):
        obs, edges, use_transform_action, num_nodes, body_ind, body_depths, body_heights, distances, lapPE = zip(*x)
        num_nodes = np.concatenate(num_nodes)
        if len(x) > 1:
            num_nodes_cum = np.cumsum(num_nodes)
        else:
            num_nodes_cum = None
        return num_nodes_cum
    
    def process_input(self, batches):
        obs, edges, use_transform_action, num_nodes, local_num_nodes_cum, transformer_obs = self.batch_data(batches)
        return obs, edges, local_num_nodes_cum, transformer_obs

    def forward(self, x):
        num_nodes_cum = self.get_root_index(x)
        ## get current stage, generate stage masks (x3)
        stages = ['reconfig', 'execution']
        x_dict = defaultdict(list)
        node_design_mask = defaultdict(list)
        total_num_nodes = 0
        for i, x_i in enumerate(x):
            num = x_i[3].item()
            cur_stage = stages[int(x_i[2].item())]
            x_dict[cur_stage].append(x_i)
            for stage in stages:
                node_design_mask[stage] += [cur_stage == stage] * num
            total_num_nodes += num
        for stage in stages:
            node_design_mask[stage] = torch.BoolTensor(node_design_mask[stage])
        
        value_nodes = torch.zeros(total_num_nodes, 1).to(x[0][0].device)
        
        # reconfig
        if len(x_dict['reconfig']) > 0:
            x, edges, _ , transformer_obs = self.process_input(x_dict['reconfig'])
                
            if self.reconfig_norm is not None:
                x = self.reconfig_norm(x)
            
            ## message passing for reconfig
            x = self.reconfig_transformer(transformer_obs)
            
            if self.reconfig_mlp is not None:
                x = self.reconfig_mlp(x)
            value_nodes[node_design_mask['reconfig']] = self.reconfig_value_head(x)
            
        # execution
        if len(x_dict['execution']) > 0:
            x, edges, _ , transformer_obs = self.process_input(x_dict['execution'])
                
            if self.control_norm is not None:
                x = self.control_norm(x)
            
            ## message passing for control
            x = self.control_transformer(transformer_obs)
            
            if self.control_mlp is not None:
                x = self.control_mlp(x)
            value_nodes[node_design_mask['execution']] = self.control_value_head(x)
        
        if num_nodes_cum is None:
            value = value_nodes[[0]]
        else:
            value = value_nodes[torch.LongTensor(np.concatenate([np.zeros(1), num_nodes_cum[:-1]]))]
        return value
