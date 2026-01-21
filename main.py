# coding=utf-8
import argparse
from re import X

import MyOT as ot
import numpy as np
from responses import target
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from data_processor import MyDataset
from models import *
from utils import *
import wandb
import yaml
import datetime
import time
import causalml
import econml
import random

def hwb_wass(x_0, x_1, rep_0, rep_1, out_0, out_1, t, yf, list_rep_0, list_rep_1, device, hparams):

    dist = hparams['rep_scale'] * ot.dist(rep_0, rep_1) 
    c_x_0 = hparams['x_scale'] * ot.dist(x_0, x_0)
    c_rep_0 = hparams['rep_scale'] * ot.dist(rep_0, rep_0)
    c_x_1 = hparams['x_scale'] * ot.dist(x_1, x_1)
    c_rep_1 = hparams['rep_scale'] * ot.dist(rep_1, rep_1)
    num_env = 5
    
    # Step1: Compute Inner Wasserstein barycenters
    k = 64  # number of Diracs of the barycenter
    list_rho_s = []
    for idx in range(num_env):
        if len(list_rep_0[idx]) == 0 or len(list_rep_1[idx]) == 0:
            continue
        measures_locations = [list_rep_0[idx].to(torch.float32).detach(), list_rep_1[idx].to(torch.float32).detach()]
        measures_weights = [torch.tensor(ot.unif(list_rep_0[idx].shape[0]), device=device).to(torch.float32), torch.tensor(ot.unif(list_rep_1[idx].shape[0]), device=device).to(torch.float32)]
        X_init =  torch.randn(k, list_rep_0[idx].shape[1], device=device).to(torch.float32)  

        b = torch.ones((k,), device=device).to(torch.float32) / k  
        rho_s = ot.bregman.free_support_sinkhorn_barycenter(measures_locations, measures_weights, X_init, reg=0.1, scale=0.01, numItermax=50)

        
        list_rho_s.append(rho_s)
    # Step2: Compute Outer Wasserstein barycenter
    k_out = 64
    measures_locations_out = list_rho_s
    num_rho = len(list_rho_s)        
    measures_weights_out = [torch.tensor(ot.unif(list_rho_s[idx].shape[0]), device=device).to(torch.float32) for idx in range(num_rho)]
    X_init_out = torch.randn(k_out, list_rho_s[0].shape[1], device=device).to(torch.float32)
    b_out = torch.ones((k,), device=device).to(torch.float32) / k
    rho = ot.bregman.free_support_sinkhorn_barycenter_unbalanced(measures_locations_out, measures_weights_out, X_init_out, reg=0.1, reg_m=0.01, scale=0.01, numItermax=50)
    
    # Step3: Loss of Outer Wasserstein barycenter
    list_rho_s_para = [torch.nn.Parameter(rho_s) for rho_s in list_rho_s]
    list_cost_matrix = [ot.dist(rho, rho_s) for rho_s in list_rho_s_para]
    loss_out = 0
    for idx in range(num_rho):
        cost = list_cost_matrix[idx]
        gamma = ot.sinkhorn(
            torch.ones(len(rho), device=device) / len(rho),
            torch.ones(len(list_rho_s_para[0]), device=device) / len(list_rho_s_para[0]),
            cost.detach(),
            reg=hparams.get('epsilon'),
            stopThr=1e-4)
        loss_out += torch.sum(gamma * cost)
    
    loss_out /= num_env
    
    s_optimizer = torch.optim.Adam(list_rho_s_para, lr=hparams.get('lr', 1e-3), weight_decay=hparams.get('l2_reg', 1e-4))
    loss_out.backward()
    s_optimizer.step()
    
    # Step4: Losses of Inner Wasserstein barycenters
    list_rho_s = [rho_s.clone().detach() for rho_s in list_rho_s_para]
    list_cost_matrix_control, list_cost_matrix_treat = [], []
    
    num = 0
    for idx in range(num_env):
        if len(list_rep_0[idx]) == 0 or len(list_rep_1[idx]) == 0:
            continue
        rho_s = list_rho_s[num]
        rep_0 = list_rep_0[idx]
        rep_1 = list_rep_1[idx]
        list_cost_matrix_control.append(ot.dist(rho_s, rep_0))
        list_cost_matrix_treat.append(ot.dist(rho_s, rep_1))
        num += 1

    loss_in = 0
    num = 0
    for idx in range(num_env):
        if len(list_rep_0[idx]) == 0 or len(list_rep_1[idx]) == 0:
            continue
        rho_s = list_rho_s[num]
        cost_control = list_cost_matrix_control[num]


        gamma = ot.sinkhorn(
            torch.ones(len(rho_s), device=device) / len(rho_s),
            torch.ones(len(list_rep_0[idx]), device=device) / len(list_rep_0[idx]),
            cost_control.detach(),
            reg=hparams.get('epsilon'),
            stopThr=1e-4)
        loss_in += torch.sum(gamma * cost_control)
        
        cost_treat = list_cost_matrix_treat[num]
        gamma = ot.sinkhorn(
            torch.ones(len(rho_s), device=device) / len(rho_s),
            torch.ones(len(list_rep_1[idx]), device=device) / len(list_rep_1[idx]),
            cost_treat.detach(),
            reg=hparams.get('epsilon'),
            stopThr=1e-4)
        loss_in += torch.sum(gamma * cost_treat)   
        
        num += 1

    loss_in /= (2 * num_env)            
        
    loss = loss_in
        
    return loss


class BaseEstimator:

    def __init__(self, hparams={}):
        data_name = hparams.get('data')
        print("Current data:", data_name)

        self.train_set = MyDataset(f"Datasets/{data_name}/train.csv", data_name)
        self.traineval_set = MyDataset(f"Datasets/{data_name}/traineval.csv", data_name)
        self.eval_set = MyDataset(f"Datasets/{data_name}/eval.csv", data_name)
        self.test_set = MyDataset(f"Datasets/{data_name}/test.csv", data_name)

        self.device = torch.device(hparams.get('device'))
        if hparams['treat_weight'] == 0:
            self.train_loader = DataLoader(self.train_set, batch_size=hparams.get('batchSize'), drop_last=True)
        else:
            self.train_loader = DataLoader(self.train_set, batch_size=hparams.get('batchSize'), sampler=self.train_set.get_sampler(hparams['treat_weight']), drop_last=True)
        self.traineval_data = DataLoader(self.traineval_set, batch_size=256)  # for test in-sample metric
        self.eval_data = DataLoader(self.eval_set, batch_size=256)
        self.test_data = DataLoader(self.test_set, batch_size=256)

        self.init_model(hparams)

        
    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        random.seed(seed)
        np.random.seed(seed)
        
    def init_model(self, hparams):
        self.set_seed(hparams['seed'])
        self.train_metric = {
             "mae_ate": np.array([]),
             "mae_att": np.array([]),
             "pehe": np.array([]),
             "r2_f": np.array([]),
             "rmse_f": np.array([]),
             "r2_cf": np.array([]),
             "rmse_cf": np.array([]),
             "auuc": np.array([]),
             "rauuc": np.array([])}
        self.eval_metric = deepcopy(self.train_metric)
        self.test_metric = deepcopy(self.train_metric)

        self.train_best_metric = {
             "mae_ate": None,
             "mae_att": None,
             "pehe": None,
             "r2_f": None,
             "rmse_f": None,
             "r2_cf": None,
             "rmse_cf": None,
             "auuc": None,
             "rauuc": None,}
        self.eval_best_metric = deepcopy(self.train_best_metric)
        self.eval_best_metric['r2_f'] = -10  
        self.eval_best_metric["pehe"] = 100
        self.eval_best_metric['auuc'] = 0
        self.loss_metric = {'loss': np.array([]), 'loss_f': np.array([]), 'loss_c': np.array([])}

        self.epochs = hparams.get('epoch', 200)

        self.model = HWBCFR(self.train_set.x_dim, hparams).to(self.device)
        
        self.criterion = torch.nn.MSELoss()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.get('lr', 1e-3), weight_decay=hparams.get('l2_reg', 1e-4))
        self.hparams = hparams
        self.temp_result = None
        self.epoch = 0    

    def fit(self, config=None):
        self.init_model(self.hparams)
        
        iter_num = 0
        for epoch in tqdm(range(1, self.epochs)):
            self.epoch = epoch
            self.model.train()
            total_batch = self.train_set.sample_num // self.hparams['batchSize'] + 1
            _aver_loss, _aver_loss_wass, _aver_loss_fit = 0, 0, 0
            for batch_idx, data in enumerate(self.train_loader):  # train_loader
                self.model.zero_grad()
                data = data.to(self.device)
                if self.hparams['data'] == 'ACIC':
                    _x, _xt, _t, _yf, _, _env = data[:, :-6], data[:, :-5], data[:, -6], data[:, -5], data[:, -4], data[:, -1]  #
                else:
                    _x, _xt, _t, _yf, _, _env = data[:, :-6], data[:, :-5], data[:, -6], data[:, -5], data[:, -4], data[:, -1]  #

                _x_0 = _x[_t == 0]
                _x_1 = _x[_t == 1]

                _pred_f = self.model(_xt)
                self.model.divide_rep_by_env(_xt, _env, num_env=5)
                _loss_fit = self.criterion(_pred_f.view(-1), _yf.view(-1))
                _loss_fit += hparams['alpha'] * self.model.get_reconstruction_loss(_xt)
                
                _loss_wass = 0
                wass_indicator = 0
                if epoch > self.hparams['pretrain_epoch']:
                    wass_indicator = 1
                    _loss_wass = hwb_wass(x_0=_x_0,
                                        x_1=_x_1,
                                        rep_0=self.model.rep_0,
                                        rep_1=self.model.rep_1,
                                        out_0=self.model.out_0,
                                        out_1=self.model.out_1,
                                        t=_t,
                                        yf=_yf,
                                        list_rep_0=self.model.list_rep_0,
                                        list_rep_1=self.model.list_rep_1,
                                        device=self.device,
                                        hparams=self.hparams)

                _loss = _loss_fit + self.hparams['lambda'] * _loss_wass
                _loss.backward()
                self.optimizer.step()
                
                _loss_wass = _loss_wass.item() if wass_indicator else 0
                
                _aver_loss += _loss.item()
                _aver_loss_fit += _loss_fit.item()
                _aver_loss_wass += _loss_wass

                iter_num += 1
            
            _aver_loss = _aver_loss / total_batch
            _aver_loss_fit = _aver_loss_fit / total_batch
            _aver_loss_wass = _aver_loss_wass / total_batch

            # Section: evaluation and model selection
            # in-sample evaluation
            eval_num = 1
            if self.epoch % eval_num == 0:
                _train_metric = self.evaluation(data='train')
                self.train_metric = metric_update(self.train_metric, _train_metric, self.epoch)

            if self.epoch % eval_num == 0:
                _eval_metric = self.evaluation(data='test')
                self.eval_metric = metric_update(self.eval_metric, _eval_metric, self.epoch)

                if abs(_eval_metric['auuc']) > abs(self.eval_best_metric['auuc']):
                    self.eval_best_metric = _eval_metric
                    self.train_best_metric = self.evaluation(data='train')
                    self.test_best_metric = self.evaluation(data='test')
                    stop_epoch = 0
                    print(self.eval_best_metric)
                else:
                    stop_epoch += 1
            if stop_epoch >= self.hparams['stop_epoch'] and self.epoch > 100:
                print(f'Early stop at epoch {self.epoch}')
                break

            self.epoch += 1
        

    def predict(self, dataloader):
        """
        :param dataloader
        :return: np.array, shape: (#sample)
        """
        self.model.eval()
        pred_0 = torch.tensor([], device=self.device)
        pred_1, yf, ycf, t, mu0, mu1 = deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0),

        for data in dataloader:
            data = data.to(self.device)
            if self.hparams['data'] == 'ACIC':
                _x, _xt, _t, _yf, _ycf, _mu_0, _mu_1 = data[:, :-6], data[:, :-5], data[:, [-6]], data[:, -5], data[:, -4], data[:, -3], data[:, -2]#
            else:
                _x, _xt, _t, _yf, _ycf, _mu_0, _mu_1 = data[:, :-6], data[:, :-5], data[:, [-6]], data[:, -5], data[:, -4], data[:, -3], data[:, -2]#
            
            rep = self.model.backbone(_x)

            _x_0 = torch.cat([_x, torch.zeros_like((_t), device=self.device)], dim=-1)
            _x_1 = torch.cat([_x, torch.ones_like((_t), device=self.device)], dim=-1)
        
            _pred_0 = self.model(_x_0).reshape([-1])
            _pred_1 = self.model(_x_1).reshape([-1])

            pred_0 = torch.cat([pred_0, _pred_0], axis=-1)
            pred_1 = torch.cat([pred_1, _pred_1], axis=-1)
            yf = torch.cat([yf, _yf], axis=-1)
            ycf = torch.cat([ycf, _ycf], axis=-1)
            
            mu0 = torch.cat([mu0, _mu_0], axis=-1)
            mu1 = torch.cat([mu1, _mu_1], axis=-1)
        
            
            t = torch.cat([t, _t.reshape([-1])], axis=-1)

        pred_0 = pred_0.detach().cpu().numpy()
        pred_1 = pred_1.detach().cpu().numpy()
        yf = yf.cpu().numpy()
        ycf = ycf.cpu().numpy()
        mu0 = mu0.cpu().numpy()
        mu1 = mu1.cpu().numpy()
        t = t.detach().cpu().numpy()
        
        return pred_0, pred_1, yf, ycf, mu0, mu1, t

    def evaluation(self, data: str) -> dict():

        dataloader = {
            'train': self.traineval_data,
            'eval': self.eval_data,
            'test': self.test_data}[data]


        pred_0, pred_1, yf, ycf, mu0, mu1, t = self.predict(dataloader)
        mode = 'in-sample' if data == 'train' else 'out-sample'
        metric = metrics(pred_0, pred_1, yf, ycf, mu0, mu1, t, mode, self.hparams)

        return metric


if __name__ == "__main__":

    hparams = argparse.ArgumentParser(description='hparams')
    hparams.add_argument('--model', type=str, default='ylearner')
    hparams.add_argument('--data', type=str, default='ACIC')
    hparams.add_argument('--epoch', type=int, default=200)
    hparams.add_argument('--seed', type=int, default=2)
    hparams.add_argument('--stop_epoch', type=int, default=30, help='tolerance epoch of early stopping')
    hparams.add_argument('--treat_weight', type=float, default=0.0, help='whether or not to balance sample')

    hparams.add_argument('--dim_backbone', type=str, default='60,60')
    hparams.add_argument('--dim_task', type=str, default='60,60')
    hparams.add_argument('--batchSize', type=int, default=256)
    hparams.add_argument('--lr', type=float, default=1e-3)
    hparams.add_argument('--l2_reg', type=float, default=1e-4)
    hparams.add_argument('--dropout', type=float, default=0)
    hparams.add_argument('--treat_embed', type=bool, default=True)
    hparams.add_argument('--lambda', type=float, default=0.01, help='weight of wass_loss in loss function')
    hparams.add_argument('--rep_scale', type=float, default=0.00001, help='rescale the representation distance.')
    hparams.add_argument('--x_scale', type=float, default=0.00001, help='rescale the covariate distance.')

    hparams.add_argument('--epsilon', type=float, default=1.0, help='Entropic Regularization in sinkhorn. In IHDP, it should be set to 0.5-5.0 according to simulation conditions')
    hparams.add_argument('--kappa', type=float, default=1.0, help='weight of marginal constraint in UOT. In IHDP, it should be set to 0.1-5.0 according to simulation conditions')
    hparams.add_argument('--gamma', type=float, default=0.000005, help='weight of joint distribution alignment. In IHDP, it should be set to 0.0001-0.005 according to simulation conditions')
    hparams.add_argument('--ot_joint_bp', type=bool, default=True, help='weight of joint distribution alignment')
    hparams.add_argument('--alpha', type=float, default=0.001, help='hyperparameter of reconstruction loss')
    
    hparams.add_argument('--pretrain_epoch', type=int, default=100, help='pretrain the prediction head')
    hparams.add_argument('--device', type=str, default='cuda:2')

    hparams = vars(hparams.parse_args())

    os.nice(0)
    estimator = BaseEstimator(hparams=hparams)
    estimator.fit(config=hparams)

