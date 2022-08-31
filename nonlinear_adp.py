from cProfile import run
from cmath import inf
from curses import curs_set
from turtle import goto
from sympy import re
from notears.locally_connected import LocallyConnected
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
from notears.lbfgsb_scipy import LBFGSBScipy
import numpy as np
import tqdm as tqdm
from notears.loss_func import *
import random
import time
import igraph as ig
import notears.utils as ut
from notears.loss_func import *
import torch.utils.data as data
from adaptive_model.adapModel import adaptiveMLP
from adaptive_model.adapModel import adap_reweight_step

from runhelps.runhelper import config_parser
from torch.utils.tensorboard import SummaryWriter
from sachs_data.load_sachs import *
import json

COUNT = 0

IF_figure = 0

parser = config_parser()
args = parser.parse_args()
IF_baseline = args.run_mode
print(args)
class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2    
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)  
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers) 
        

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1] 
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d] 
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i] 
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        M = torch.eye(d).to(A.device) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg 

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg
    
    def predict(self,x):
        return self.forward(x)

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W

class GOLEM(nn.Module):
    """Set up the objective function of GOLEM.
    Hyperparameters:
        (1) GOLEM-NV: lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: lambda_1=2e-2, lambda_2=5.0.(not used)
    """

    def __init__(self, args):
        super(GOLEM, self).__init__()
        self.n = args.n
        self.d = args.d
        self.lambda_1 = args.lambda1
        self.lambda_2 = args.lambda2
        self.W=nn.Linear(args.d, args.d, bias=False)
        self.lr=args.golem_lr
        nn.init.zeros_(self.W.weight)

        #nn.init.xavier_normal_(self.W.weight)

        # with torch.no_grad():
        #     #self.W.weight=torch.triu(self.W.weight)
        #     idx=torch.triu_indices(*self.W.weight.shape)
        #     self.W.weight[idx[0],idx[1]]=0
            
    
    def predict(self,X):
        return self.W(X)
    

    def forward(self, X, weight):

        likelihood = self._compute_likelihood(X,weight)
        L1_penalty = self._compute_L1_penalty()
        h = self._compute_h()
        loss= likelihood + self.lambda_1 * L1_penalty + self.lambda_2 * h
        return loss, likelihood, self.lambda_1 * L1_penalty, self.lambda_2 * h

    def _compute_likelihood(self,X,weight):
        """Compute (negative log) likelihood in the linear Gaussian case.
        Returns:
            tf.Tensor: Likelihood term (scalar-valued).
        """
        return 0.5 * self.d * torch.log(
                torch.sum(torch.mul(weight,torch.sum(torch.square(X-self.W(X)),dim=1)))
                # torch.square(
                #     torch.linalg.norm(X - self.W(X))
                # )
            ) - torch.linalg.slogdet(torch.eye(self.d) - self.W.weight.T)[1]
        # return 0.5 * torch.sum(
        #     torch.log(
        #         torch.sum(
        #             torch.square(X - self.W(X)), axis=0
        #         )
        #     )
        # ) - torch.linalg.slogdet(torch.eye(self.d) - self.W.weight.T)[1]

    def _compute_L1_penalty(self):
        """Compute L1 penalty.
        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return torch.norm(self.W.weight, 1)

    def _compute_h(self):
        """Compute DAG penalty.
        Returns:
            tf.Tensor: DAG penalty term (scalar-valued).
        """
        return torch.trace(torch.matrix_exp(self.W.weight.T * self.W.weight.T)) - self.d
    
    @torch.no_grad()
    def W_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        w = self.W.weight.T.cpu().detach().numpy()  # [i, j]
        return w


# TODO: create the adaptive_loss function
def adaptive_loss(output, target, reweight_list):
    R = output-target
    # reweight_matrix = torch.diag(reweight_idx).to(args.device)
    # loss = 0.5 * torch.sum(torch.matmul(reweight_matrix, R))
    loss = 0.5 * torch.sum(torch.mul(reweight_list, R**2))
    return loss

def record_weight(reweight_list, cnt, hard_list=[26,558,550,326,915], easy_list=[859,132,82,80,189]):
    writer = SummaryWriter('logs/weight_record_real')
    reweight_idx = reweight_list.squeeze()
    reweight_idx = reweight_idx.tolist()
    for idx in hard_list:
        writer.add_scalar(f'hard_real/hard_reweight_list[{idx}]', reweight_idx[idx], cnt)
    for idx in easy_list:
        writer.add_scalar(f'easy_real/easy_reweight_list[{idx}]', reweight_idx[idx], cnt) 

def record_distribution(reweight_list,j):
    writer = SummaryWriter('logs/distribution_record')
    reweight_idx = reweight_list.squeeze()
    reweight_idx = reweight_idx.tolist()
    for i in range(len(reweight_idx)):
        writer.add_scalar(f'weight_distribution_step{j}', reweight_list[i], i)
    # 画出reweight_idx的分布图
    import matplotlib.pyplot as plt
    # 画出reweight_idx的箱型图
    plt.boxplot(reweight_idx)
    # 保存图片
    plt.savefig(f'logs/box_plot{j}.png')


def dual_ascent_step_golem(model, X, train_loader, adp_flag, adaptive_model, max_epochs, patience=20):
    X = X - X.mean(axis=0, keepdims=True)
    X = X.to(args.device)
    #print(X)
    cur_patience=0
    last_loss=inf
    epoch=0
    while cur_patience<patience:
        global COUNT
        COUNT += 1
        
        optimizer = torch.optim.Adam([ param for param in model.parameters() if param.requires_grad == True], lr=model.lr)

        primal_obj = torch.tensor(0.).to(args.device)
        tot_loss = torch.tensor(0.).to(args.device)
        tot_likelihood = torch.tensor(0.).to(args.device)
        tot_L1 = torch.tensor(0.).to(args.device)
        tot_h = torch.tensor(0.).to(args.device)

        for _ , tmp_x in enumerate(train_loader):
            batch_x = tmp_x[0].to(args.device)
            batch_x = batch_x - torch.mean(batch_x)
            
            X_hat = model.predict(batch_x)
                
                # TODO: the adaptive loss should add here
            if adp_flag == False or IF_baseline == False:
                reweight_list = torch.ones(batch_x.shape[0],1)/batch_x.shape[0]
                reweight_list = reweight_list.to(args.device)
            else:
                with torch.no_grad():
                    model.eval()
                    reweight_list = adaptive_model((batch_x-X_hat)**2)
                # TODO: record the reweight
                if IF_figure:
                    record_weight(reweight_list=reweight_list, cnt=COUNT, hard_list=[748,181,276,151,355,137,846,671], easy_list=[802,673,317,192,167])
            
                model.train()
                # print(reweight_list.squeeze(1))
                # print(reweight_list)
                # print(model.W.weight)
                # input()
            loss, likelihood, L1_penalty, h = model(batch_x,reweight_list)#adaptive_loss(X_hat, batch_x, reweight_list)
            #print(loss)
           
                

            tot_loss+=loss
            tot_likelihood+=likelihood
            tot_L1+=L1_penalty
            tot_h+=h
        
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

        if tot_loss.detach().item() < last_loss:
            last_loss= tot_loss.detach().item()
            cur_patience=0
        else:
            cur_patience+=1
        #print(model.W.weight)
        h_cur = model._compute_h().detach().item()
        perf_str='Epoch %d : training loss ==[%.5f = %.5f + %.5f +  %.5f], curr H: %.5f, curr patience: %d' % (
                epoch, tot_loss.detach().item(),tot_likelihood.detach().item(), 
                tot_L1.detach().item(), tot_h.detach().item(), h_cur,cur_patience)
        epoch+=1
        #print(perf_str)
       
    
    return h

    

    
def dual_ascent_step(model, X, train_loader, lambda1, lambda2, rho, alpha, h, rho_max, adp_flag, adaptive_model):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    # X_torch = torch.from_numpy(X)
    while rho < rho_max:
        
        def closure():
            global COUNT
            COUNT += 1
            optimizer.zero_grad()
            X_hat = model(X)
            loss = squared_loss(X_hat, X)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            # if COUNT % 100 == 0:
            #     print(f"{primal_obj}: {primal_obj.item():.4f}; count: {COUNT}")
            return primal_obj

        def r_closure():
            global COUNT
            COUNT += 1
            optimizer.zero_grad()

            primal_obj = torch.tensor(0.).to(args.device)
            loss = torch.tensor(0.).to(args.device)

            for _ , tmp_x in enumerate(train_loader):
                batch_x = tmp_x[0].to(args.device)
                X_hat = model(batch_x)
                
                # TODO: the adaptive loss should add here
                if adp_flag == False:
                    reweight_list = torch.ones(batch_x.shape[0],1)/batch_x.shape[0]
                    reweight_list = reweight_list.to(args.device)
                else:
                    with torch.no_grad():
                        model.eval()
                        reweight_list = adaptive_model((batch_x-X_hat)**2)
                    # TODO: record the reweight
                    if IF_figure:
                        record_weight(reweight_list=reweight_list, cnt=COUNT, hard_list=[748,181,276,151,355,137,846,671], easy_list=[802,673,317,192,167])
                   
                    model.train()
                # print(reweight_list.squeeze(1))
                primal_obj += adaptive_loss(X_hat, batch_x, reweight_list)
            
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj += penalty + l2_reg + l1_reg
            primal_obj.backward()
            # if COUNT % 100 == 0:
            #     print(f"{primal_obj}: {primal_obj.item():.4f}; count: {COUNT}")
            return primal_obj

        if IF_baseline:
            optimizer.step(closure)  # NOTE: updates model in-place
        else:                        # NOTE: the adaptive reweight operation
            optimizer.step(r_closure)

        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new

def golem_linear(model: nn.Module,
                      adaptive_model: nn.Module,
                      X: np.ndarray,
                      train_loader: data.DataLoader,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 5,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3,
                      duel_epoch: int = 10
                      ):
    adp_flag = False
    for j in tqdm.tqdm(range(max_iter)):
        if j > args.reweight_epoch:
            # TODO: reweight operation here
            adp_flag = True
            if not IF_baseline:
                print("Re-weighting")
                reweight_idx_tmp = adap_reweight_step(adaptive_model, train_loader, args.adaptive_lambda , model, args.adaptive_epoch, args.adaptive_lr)
                # TODO: record the distribution
                
                if IF_figure:
                    record_distribution(reweight_idx_tmp,j)    
            h=dual_ascent_step_golem(model, X, train_loader, adp_flag, adaptive_model, duel_epoch)
        else:
            h=dual_ascent_step_golem(model, X, train_loader, adp_flag, adaptive_model, duel_epoch)
        
        if h <= h_tol:
            break
    

    W_est = model.W_to_adj()
 
    W_est[np.abs(W_est) < w_threshold] = 0

    while not ut.is_dag(W_est):
        w_threshold+=0.01
        W_est[np.abs(W_est) < w_threshold] = 0
    # TODO: 打印fit不好的结果和相关信息
    hard_index, easy_index = hard_mining(X, model, single_loss, ratio=0.01)
    # 分别将hard和easy的索引保存到txt文件中
    # np.savetxt(f'hard{args.seed}.txt', hard_index, fmt='%d')
    # np.savetxt(f'easy{args.seed}.txt', easy_index, fmt='%d')
    return W_est, hard_index, easy_index


def notears_nonlinear(model: nn.Module,
                      adaptive_model: nn.Module,
                      X: np.ndarray,
                      train_loader: data.DataLoader,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 5,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3
                      ):
    rho, alpha, h = 1.0, 0.0, np.inf
    adp_flag = False
    for j in tqdm.tqdm(range(max_iter)):
        if j > args.reweight_epoch:
            # TODO: reweight operation here
            adp_flag = True
            if not IF_baseline:
                print("Re-weighting")
                reweight_idx_tmp = adap_reweight_step(adaptive_model, train_loader, args.adaptive_lambda , model, args.adaptive_epoch, args.adaptive_lr)
                # TODO: record the distribution
                if IF_figure:
                    record_distribution(reweight_idx_tmp,j)
                
            rho, alpha, h = dual_ascent_step(model, X, train_loader, lambda1, lambda2,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model)

        else:
            rho, alpha, h = dual_ascent_step(model, X, train_loader, lambda1, lambda2,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model)
        
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0

    
    # TODO: 打印fit不好的结果和相关信息
    hard_index, easy_index = hard_mining(X, model, single_loss, ratio=0.01)
    # 分别将hard和easy的索引保存到txt文件中
    # np.savetxt(f'hard{args.seed}.txt', hard_index, fmt='%d')
    # np.savetxt(f'easy{args.seed}.txt', easy_index, fmt='%d')
    return W_est, hard_index, easy_index


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def hard_mining(data, model, loss_func, ratio = 0.01):
    """
    data: (N_observations, nodes)
    """
    N_sample = data.shape[0]
    model.eval()
    data_hat = model.predict(data)
    loss_col = loss_func(data_hat, data)
    loss_col = torch.sum(loss_col, dim=1)
    loss_col = loss_col.cpu().detach().numpy()
    # 找出最大ratio的loss_col的index
    hard_index_list = np.argsort(loss_col)[::-1][:int(N_sample * ratio)]
    easy_index_list = np.argsort(loss_col)[:int(N_sample * ratio)]
    return hard_index_list, easy_index_list

def main(trials,seed):
    # fangfu

    tot_perf={}
    for trial in range(trials):
        print('==' * 20)

        import notears.utils as ut
        cur_seed=trial+seed
        set_random_seed(cur_seed)

        linearity = "non-linear"

        if args.use_golem:
            B_true = ut.simulate_dag(args.d, args.s0, args.graph_type)
            X = ut.simulate_linear_sem(B_true, args.n, args.linear_sem_type)
            model = GOLEM(args) # FIXME: the layer of the Notears MLP
            adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)
            linearity = "linear"
        else:
            if args.data_type == 'real':
                # X = np.loadtxt('/opt/data2/git_fangfu/JTT_CD/data/sachs.csv', delimiter=',')
                X = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs.csv', delimiter=',')
                B_true = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs_B_true.csv', delimiter=',')
                model = NotearsMLP(dims=[11, 1], bias=True) # for the real data (sachs)   the nodes of sachs are 11
                adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)
                linearity = "linear"

            elif args.data_type == 'synthetic':
                

                B_true = ut.simulate_dag(args.d, args.s0, args.graph_type)
                if args.linear:
                    linearity='linear'
                    X = ut.simulate_linear_sem(B_true, args.n, args.linear_sem_type)
                else:
                    X = ut.simulate_nonlinear_sem(B_true, args.n, args.sem_type)
                model = NotearsMLP(dims=[args.d, 10, 1], bias=True) # FIXME: the layer of the Notears MLP
                adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)
            
            elif args.data_type == 'testing':
                B_true = np.loadtxt('testing_B_true.csv', delimiter=',')
                X = np.loadtxt('testing_X.csv', delimiter=',')
                model = NotearsMLP(dims=[args.d ,10, 1], bias=True) # FIXME: the layer of the Notears MLP
                adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)

            elif args.data_type == 'sachs_full':
                X = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs7466.csv', delimiter=',')
                B_true = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs_B_true.csv', delimiter=',')
                model = NotearsMLP(dims=[11, 1], bias=True) # for the real data (sachs)   the nodes of sachs are 11
                adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)
        

        datatype = args.data_type if not args.use_golem else 'synthetic'

        sem_type = args.sem_type if linearity=="non-linear" else args.linear_sem_type

        #print(B_true)
        f_dir=f'reweight_experiment/{linearity}/{args.d}_{args.s0}_{args.n}/{args.graph_type}_{sem_type}/'
        import os
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
        
        
        X = torch.from_numpy(X).float().to(args.device)
        model.to(args.device)
        
        # TODO: 将X装入DataLoader
        X_data = data.TensorDataset(X)
        train_loader = data.DataLoader(X_data, batch_size=args.batch_size, shuffle=True)
        if args.use_golem:
            W_est , _, _= golem_linear(model, adaptive_model, X, train_loader, args.lambda1, args.lambda2, duel_epoch=args.duel_epoch)
        else:
            W_est , _, _= notears_nonlinear(model, adaptive_model, X, train_loader, args.lambda1, args.lambda2)
        #assert ut.is_dag(W_est)
        # np.savetxt('W_est.csv', W_est, delimiter=',')
        #print(B_true)
        #print(W_est)
        acc = ut.count_accuracy(B_true, W_est != 0)
        print(acc)
        # 根据args.d和args.s0生成文件夹
        
        f_path=f'reweight_experiment/{linearity}/{args.d}_{args.s0}_{args.n}/{args.graph_type}_{sem_type}/seed_{cur_seed}.txt'
        
        # 创建该'my_experiment/{args.d}_{args.s0}/{args.graph_type}_{args.sem_type}/{args.seed}.txt'该文件

        if args.data_type == 'synthetic':
            with open(f_path, 'a') as f:
                f.write(f'args:{args}\n')
                f.write(f'run_mode: {IF_baseline}\n')
                f.write(f'observation_num: {args.n}\n')
                if not IF_baseline:
                    f.write(f'temperature: {args.temperature}\n')
                    f.write(f'batch_size:{args.batch_size}\n')
                f.write(f'dataset_type:{args.data_type}\n')
                f.write(f'is_golem:{args.use_golem}\n')
                f.write(f'acc:{acc}\n')
                f.write('-----------------------------------------------------\n')
        

        for key,value in acc.items():
            if key not in tot_perf:
                tot_perf[key]={"value":[],"mean":[],"std":[]}
            tot_perf[key]["value"].append(value)
    for key,value in tot_perf.items():
        perf=np.array(value["value"])
        tot_perf[key]['mean']=float(np.mean(perf))
        tot_perf[key]['std']=float(np.std(perf))
    
    reweight_str="_reweight" if not IF_baseline else ""
    with open(f_dir+"stats_"+str(args.lambda1)+"_"+str(args.lambda2)+reweight_str+".json",'w') as f:
        json.dump(tot_perf,f)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=10)
    main(args.trial,args.seed)
