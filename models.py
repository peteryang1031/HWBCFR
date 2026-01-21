from networkx import dijkstra_predecessor_and_distance
from numpy import argmin
import torch
import torch.nn as nn
from copy import deepcopy
import causalml
from econml.dml import CausalForestDML
import torch.nn.functional as F
  
    
class HWBCFR(nn.Module):
    def __init__(self, input_dim, hparams):

        super(HWBCFR, self).__init__()

        out_backbone = hparams.get('dim_backbone', '32,16').split(',')
        out_task = hparams.get('dim_task', '16').split(',')
        self.treat_embed = hparams.get('treat_embed', True)
        in_backbone = [input_dim] + list(map(int, out_backbone))
        print('in_backbone is ' + str(input_dim))
        self.backbone = torch.nn.Sequential()
        for i in range(1, len(in_backbone)):
            self.backbone.add_module(f"backbone_dense{i}", torch.nn.Linear(in_backbone[i-1], in_backbone[i]))
            self.backbone.add_module(f"backbone_relu{i}", torch.nn.ELU())
            self.backbone.add_module(f"backbone_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))
            
        self.inverse_backbone = torch.nn.Sequential()
        for i in range(1, len(in_backbone), -1):
            self.backbone.add_module(f"backbone_dense{i}", torch.nn.Linear(in_backbone[i], in_backbone[i-1]))
            self.backbone.add_module(f"backbone_relu{i}", torch.nn.ELU())
            self.backbone.add_module(f"backbone_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))
        

        in_task = [in_backbone[-1]] + list(map(int, out_task))
        if self.treat_embed is True: 
            in_task[0] += 2

        self.tower_1 = torch.nn.Sequential()
        for i in range(1, len(in_task)):
            self.tower_1.add_module(f"tower_dense{i}", torch.nn.Linear(in_task[i-1], in_task[i]))
            self.tower_1.add_module(f"tower_relu{i}", torch.nn.ELU())
            self.tower_1.add_module(f"tower_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.output_1 = torch.nn.Sequential()
        self.output_1.add_module("output_dense", torch.nn.Linear(in_task[-1], 1))

        self.tower_0 = deepcopy(self.tower_1)
        self.output_0 = deepcopy(self.output_1)

        self.rep_1, self.rep_0 = None, None
        self.out_1, self.out_0 = None, None
        self.list_rep_0, self.list_rep_1 = None, None
        self.embedding = nn.Embedding(2, 2)
        self.recon_criterion = nn.MSELoss()

    def get_reconstruction_loss(self, x):
        covariates = x[:, :-1]
        rep = self.backbone(covariates)
        recon_x = self.inverse_backbone(rep)
        recon_loss = self.recon_criterion(recon_x, rep)
        return recon_loss
        
    def divide_rep_by_env(self, x, env, num_env):
        covariates = x[:, :-1]
        t = x[:, -1]
        rep = self.backbone(covariates)
        if self.treat_embed is True:
            t_embed = self.embedding(t.int())
            rep_t = torch.cat([rep, t_embed], dim=-1)
        else:
            rep_t = rep
        self.list_rep_0 = [rep[(t == 0) & (env == i)] for i in range(1, num_env + 1)]
        self.list_rep_1 = [rep[(t == 1) & (env == i)] for i in range(1, num_env + 1)]
        
    def forward(self, x):

        covariates = x[:, :-1]
        t = x[:, -1]
        rep = self.backbone(covariates)
        if self.treat_embed is True:
            t_embed = self.embedding(t.int())
            rep_t = torch.cat([rep, t_embed], dim=-1)
        else:
            rep_t = rep

        self.rep_1 = rep[t == 1]
        self.rep_0 = rep[t == 0]

        self.out_1 = self.output_1(self.tower_1(rep_t))
        self.out_0 = self.output_0(self.tower_0(rep_t))

        t = t.reshape(-1, 1)
        output_f = t * self.out_1 + (1 - t) * self.out_0

        return output_f
