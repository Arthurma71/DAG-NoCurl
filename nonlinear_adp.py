from cProfile import run
from cmath import inf
from curses import curs_set
from os import rename
from turtle import goto
from sympy import re
import os

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
import torch.utils.data as data
from adaptive_model.adapModel import *
from adaptive_model.baseModel import *
from runhelps.runhelper import config_parser
from torch.utils.tensorboard import SummaryWriter
from sachs_data.load_sachs import *
import json

COUNT = 0



parser = config_parser()
args = parser.parse_args()
IF_baseline = args.run_mode
IF_figure = args.figure
print(args)


class TensorDatasetIndexed(data.Dataset):
    def __init__(self,tensor):
        self.tensor=tensor
    
    def __getitem__(self,index):
        return (self.tensor[index],index)
    
    def __len__(self):
        return self.tensor.size(0)



def record_weight(reweight_list, cnt, hard_list=[26,558,550,326,915], easy_list=[859,132,82,80,189]):
    writer = SummaryWriter('logs/weight_record_real')
    reweight_idx = reweight_list.squeeze()
    reweight_idx = reweight_idx.tolist()
    for idx in hard_list:
        writer.add_scalar(f'hard_real/hard_reweight_list[{idx}]', reweight_idx[idx], cnt)
    for idx in easy_list:
        writer.add_scalar(f'easy_real/easy_reweight_list[{idx}]', reweight_idx[idx], cnt) 

def record_distribution(reweight_list,R,j,idx):
    writer = SummaryWriter('logs/distribution_record')
    reweight_idx = reweight_list.squeeze()
    reweight_idx = reweight_idx.tolist()
    R=torch.sum(R,dim=1).squeeze()
    R=R.tolist()
    for i in range(len(reweight_idx)):
        writer.add_scalar(f'weight_distribution_step{j}', reweight_list[i], i)
    # 画出reweight_idx的分布图
    import matplotlib.pyplot as plt
    # 画出reweight_idx的箱型图
    # plt.boxplot(reweight_idx)
    # # 保存图片
    # plt.savefig(f'logs/box_plot{j}.png')
    # plt.clf()
    if args.scaled_noise:
        color=['b' if idx[i].item() < int(args.n*(1-args.p_n)) else 'r' for i in range(len(idx)) ]
    else:
        color='b'
    #print(color)

    plt.cla()
    plt.scatter(R,reweight_idx,c=color)

    plt.savefig(f'logs/R_vs_weight_{j}_seed_{args.p_n}_{args.p_d}_{COUNT}.png')

def is_acyclic(adjacency):
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0: return False
    return True


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
                      ):
    adp_flag = False
    for j in tqdm.tqdm(range(max_iter)):
        if j > args.reweight_epoch:
            # TODO: reweight operation here
            adp_flag = True
            if not IF_baseline:
                print("Re-weighting")
                reweight_idx_tmp,R_tmp,idx= adap_reweight_step(args,adaptive_model, train_loader, args.adaptive_lambda , model, args.adaptive_epoch, args.adaptive_lr)
                # TODO: record the distribution
                
                #if IF_figure:
                #record_distribution(reweight_idx_tmp,R_tmp,j,idx)    
            h=dual_ascent_step_golem(args, model, X, train_loader, adp_flag, adaptive_model)
        else:
            h=dual_ascent_step_golem(args, model, X, train_loader, adp_flag, adaptive_model)
        
        if h <= h_tol:
            break
    

    W_est = model.W_to_adj()
 
    W_est[np.abs(W_est) < w_threshold] = 0

    while not ut.is_dag(W_est):
        w_threshold+=0.01
        W_est[np.abs(W_est) < w_threshold] = 0
    # TODO: 打印fit不好的结果和相关信息
    hard_index, easy_index = hard_mining(args, X, model, single_loss, ratio=0.01)
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
                      max_iter: int = 50,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3
                      ):
    #print([param.device for param in model.parameters()])
    rho, alpha, h = 1.0, 0.0, np.inf
    adp_flag = False
    for j in tqdm.tqdm(range(max_iter)):
        if j > args.reweight_epoch:
            # TODO: reweight operation here
            adp_flag = True
            if not IF_baseline:
                print("Re-weighting")
                reweight_idx_tmp = adap_reweight_step(args,adaptive_model, train_loader, args.adaptive_lambda , model, args.adaptive_epoch, args.adaptive_lr)
                # TODO: record the distribution
                if IF_figure:
                    record_distribution(reweight_idx_tmp,j)
                
            rho, alpha, h = dual_ascent_step(args, model, X, train_loader, lambda1, lambda2,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model)

        else:
            rho, alpha, h = dual_ascent_step(args, model, X, train_loader, lambda1, lambda2,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model)
        
        #print(h, rho, alpha)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0


    # if not is_acyclic(W_est):
    #     W_est_pre = W_est
    #     thresholds = np.unique(W_est_pre)
    #     for step, t in enumerate(thresholds):
    #         #print("Edges/thresh", model.adjacency.sum(), t)
    #         to_keep = np.array(W_est_pre > t + 1e-8)
    #         new_adj =  W_est_pre * to_keep

    #         if is_acyclic(new_adj):
    #             W_est=new_adj
    #             #model.adjacency.copy_(new_adj)
    #             break

    
    # TODO: 打印fit不好的结果和相关信息
    hard_index, easy_index = hard_mining(args, X, model, single_loss, ratio=0.01)
    # 分别将hard和easy的索引保存到txt文件中
    # np.savetxt(f'hard{args.seed}.txt', hard_index, fmt='%d')
    # np.savetxt(f'easy{args.seed}.txt', easy_index, fmt='%d')
    return W_est, hard_index, easy_index

def daggnn_nonlinear(model: nn.Module,
                      adaptive_model: nn.Module,
                      X: np.ndarray,
                      train_loader: data.DataLoader,
                      max_iter: int = 20,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3,
                      true_graph=None
                      ):
    rho, alpha, h = 1.0, 0.0, np.inf
    adp_flag = False
    for j in tqdm.tqdm(range(max_iter)):
        if j > args.reweight_epoch:
            # TODO: reweight operation here
            adp_flag = True
            if not IF_baseline:
                print("Re-weighting")
                reweight_idx_tmp = adap_reweight_step(args,adaptive_model, train_loader, args.adaptive_lambda , model, args.adaptive_epoch, args.adaptive_lr)
                # TODO: record the distribution
                if IF_figure:
                    record_distribution(reweight_idx_tmp,j)
                
            rho, alpha, h = dual_ascent_step_daggnn(args, model, X, train_loader,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model,true_graph)

        else:
            rho, alpha, h = dual_ascent_step_daggnn(args, model, X, train_loader,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model,true_graph)
        
        print(rho," ",alpha," ",h)

        W_est = model.get_adj()
        W_est[np.abs(W_est) < w_threshold] = 0

        acc = ut.count_accuracy(true_graph, W_est != 0)
        print(acc)
        
        if h <= h_tol or rho >= rho_max:
            break
    # W_est = model.get_adj()
    # W_est[np.abs(W_est) < w_threshold] = 0

    # acc = ut.count_accuracy(true_graph, W_est != 0)
    # print(acc)

    
    # TODO: 打印fit不好的结果和相关信息
    hard_index, easy_index = hard_mining(args, X, model, single_loss, ratio=0.01)
    return W_est, hard_index, easy_index

def grandag_nonlinear(model: nn.Module,
                      adaptive_model: nn.Module,
                      X: np.ndarray,
                      train_loader: data.DataLoader,
                      max_iter: int = 1000,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3,
                      true_graph=None
                      ):
    rho, alpha, h = 1e-3, 0.0, np.inf
    mus, lambdas, w_adjs= [],[],[]
    iter_cnt=0
    adp_flag = False
    for j in tqdm.tqdm(range(max_iter)):
        if j > args.reweight_epoch:
            # TODO: reweight operation here
            adp_flag = True
            if not IF_baseline:
                print("Re-weighting")
                reweight_idx_tmp = adap_reweight_step(args,adaptive_model, train_loader, args.adaptive_lambda , model, args.adaptive_epoch, args.adaptive_lr)
                # TODO: record the distribution
                if IF_figure:
                    record_distribution(reweight_idx_tmp,j)
                
            rho, alpha, h, mus, lambdas,w_adjs,iter_cnt= dual_ascent_step_grandag(args, model, X, train_loader,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model,true_graph,mus,lambdas,w_adjs,iter_cnt)

        else:
            rho, alpha, h, mus, lambdas,w_adjs,iter_cnt = dual_ascent_step_grandag(args, model, X, train_loader,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model,true_graph,mus,lambdas,w_adjs,iter_cnt)
        
        #print(rho," ",alpha," ",h)

        # W_est = model.adjacency.detach().cpu().numpy().astype(np.float32)
        # print(W_est)
        # W_est[np.abs(W_est) < w_threshold] = 0

        #acc = ut.count_accuracy(true_graph, W_est != 0)
        #print(acc)
        
        if h <= h_tol or rho >= rho_max:
            break
    # W_est = model.get_adj()
    # W_est[np.abs(W_est) < w_threshold] = 0

    # acc = ut.count_accuracy(true_graph, W_est != 0)
    # print(acc)

    W_est_pre = model.get_w_adj().detach().cpu().numpy().astype(np.float32)


    # Find the smallest threshold that removes all cycle-inducing edges
    thresholds = np.unique(W_est_pre)
    for step, t in enumerate(thresholds):
        #print("Edges/thresh", model.adjacency.sum(), t)
        to_keep = torch.Tensor(W_est_pre > t + 1e-8).to(model.device)
        new_adj = model.adjacency * to_keep
        if is_acyclic(new_adj.cpu().detach()):
            model.adjacency.copy_(new_adj)
            break

    W_est = model.adjacency.cpu().detach().numpy().astype(np.float32)

    print(W_est)
    acc = ut.count_accuracy(true_graph, W_est != 0)
    print(acc)

    
    # TODO: 打印fit不好的结果和相关信息
    hard_index, easy_index = hard_mining(args, X, model, single_loss, ratio=0.01)
    return W_est, hard_index, easy_index



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def hard_mining(args, data, model, loss_func, ratio = 0.01):
    """
    data: (N_observations, nodes)
    """
    data.to(args.device)
    model.to(args.device)
    N_sample = data.shape[0]
    model.eval()
    if args.modeltype!="grandag":
        data_hat = model.predict(data)
        loss_col = loss_func(data_hat, data)
        loss_col = torch.sum(loss_col, dim=1)
    else:
        weights, biases, extra_params = model.get_parameters(mode="wbx")
        log_likelihood = model.compute_log_likelihood(data, weights, biases, extra_params)
        loss_col=-torch.mean(log_likelihood)
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
        global COUNT
        COUNT=cur_seed
        set_random_seed(cur_seed)

        if args.modeltype=="notears":
            if args.data_type == 'real' or args.data_type == 'sachs_full':
                model = NotearsMLP(dims=[11, 1], bias=True) # for the real data (sachs)   the nodes of sachs are 11
            else:
                if args.linear:
                    model = NotearsMLP(dims=[args.d, 1], bias=True)
                else:
                    model = NotearsMLP(dims=[args.d, 10, 1], bias=True)
        elif args.modeltype=="golem":
            model = GOLEM(args)
        elif args.modeltype=="daggnn":
            encoder=DAGGNN_MLPEncoder(args.d,1,10,1,args.batch_size)
            decoder=DAGGNN_MLPDecoder(1,1,10,args.batch_size,10)
            model = DAGGNN(encoder,decoder)
        elif args.modeltype=="grandag":
            if args.gran_model == "NonLinGauss":
                model = LearnableModel_NonLinGauss(args.d, args.gran_layers, args.gran_hid_dim, nonlin=args.nonlin,
                                                norm_prod=args.norm_prod, square_prod=args.square_prod,device=args.device)
            elif args.gran_model == "NonLinGaussANM":
                model = LearnableModel_NonLinGaussANM(args.d, args.gran_layers, args.granhid_dim, nonlin=args.nonlin,
                                                    norm_prod=args.norm_prod,
                                                    square_prod=args.square_prod,device=args.device)        

        noise_scale=None
        if args.scaled_noise:
            #noise_scale:[d, n]
            p_n=int((args.p_n)*args.n)
            p_d=int((args.p_d)*args.d)
            info_scale=np.concatenate([args.noise_1*np.ones(args.d-p_d),args.noise_0*np.ones(p_d)])
            non_info_scale=np.concatenate([args.noise_0*np.ones(p_d),args.noise_1*np.ones(args.d-p_d)])
            a1=np.tile(info_scale, (args.n-p_n,1))
            a2=np.tile(non_info_scale,(p_n,1))
            print(np.shape(a1))
            print(np.shape(a2))
            noise_scale=np.concatenate([a1,a2],axis=0)
            noise_scale=noise_scale.T

            print(noise_scale)
        

        
        
        
        if args.data_type=="synthetic" and args.linear:
            linearity = "linear"
        else:
            linearity = "non-linear"
        datatype = args.data_type
        sem_type = args.sem_type if linearity=="non-linear" else args.linear_sem_type
        if args.scaled_noise:
            sem_type+="_scaled"


        
        
        data_dir=f'data/{linearity}/{args.graph_type}_{sem_type}/'
        ensureDir(data_dir)

        
        
        if not args.scaled_noise:
            data_name=f'{args.d}_{args.s0}_{args.n}_{cur_seed}'
        else:
            data_name=f'{args.d}_{args.s0}_{args.n}_{args.p_d}_{args.p_n}_{args.noise_1}_{args.noise_0}_{cur_seed}'

        
        try:
            # if args.data_type=='synthetic_gran':
            #     print('gran!')
            #     X=np.load(f'/storage/wcma/DAG-NoCurl/data/data_p{args.d}_e{args.s0}_n{args.n}_GP/'+f'data{cur_seed}.npy')
            #     B_true=np.load(f'/storage/wcma/DAG-NoCurl/data/data_p{args.d}_e{args.s0}_n{args.n}_GP/'+f'DAG{cur_seed}.npy')
            # else:
            X=np.load(data_dir+data_name+"_X.npy")
            B_true=np.load(data_dir+data_name+"_B.npy")
            print("data loaded...")
        except:
            print("generating data from scratch...")
            if args.data_type == 'real':
                # X = np.loadtxt('/opt/data2/git_fangfu/JTT_CD/data/sachs.csv', delimiter=',')
                X = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs.csv', delimiter=',')
                B_true = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs_B_true.csv', delimiter=',')
                #model = NotearsMLP(dims=[11, 1], bias=True) # for the real data (sachs)   the nodes of sachs are 11
                #adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)

            elif args.data_type == 'synthetic':
                B_true = ut.simulate_dag(args.d, args.s0, args.graph_type)
                if args.linear:
                    X = ut.simulate_linear_sem(B_true, args.n, args.linear_sem_type)
                else:
                    X = ut.simulate_nonlinear_sem(B_true, args.n, args.sem_type)
                #model = NotearsMLP(dims=[args.d, 10, 1], bias=True) # FIXME: the layer of the Notears MLP
                #adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)
            
            elif args.data_type == 'testing':
                B_true = np.loadtxt('testing_B_true.csv', delimiter=',')
                X = np.loadtxt('testing_X.csv', delimiter=',')
                #model = NotearsMLP(dims=[args.d ,10, 1], bias=True) # FIXME: the layer of the Notears MLP
                #adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)

            elif args.data_type == 'sachs_full':
                X = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs7466.csv', delimiter=',')
                B_true = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs_B_true.csv', delimiter=',')
                #model = NotearsMLP(dims=[11, 1], bias=True) # for the real data (sachs)   the nodes of sachs are 11
                #adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)
            np.save(data_dir+data_name+"_X",X)
            np.save(data_dir+data_name+"_B",B_true)

        



        adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature,device=args.device).to(args.device)

       
        #print(B_true)
        f_dir=f'reweight_experiment/{linearity}/{args.d}_{args.s0}_{args.n}/{args.graph_type}_{sem_type}/'
        import os
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
        
        X = torch.from_numpy(X).float().to(args.device)
        model.to(args.device)
        #print([(param.name,param.device) for param in model.parameters()])


    
        # TODO: 将X装入DataLoader
        X_data = TensorDatasetIndexed(X)
        train_loader = data.DataLoader(X_data, batch_size=args.batch_size, shuffle=True)
        if args.modeltype=="golem":
            W_est , _, _= golem_linear(model, adaptive_model, X, train_loader, args.lambda1, args.lambda2)
        elif args.modeltype=='notears':
            W_est , _, _= notears_nonlinear(model, adaptive_model, X, train_loader, args.lambda1, args.lambda2)
        elif args.modeltype=='daggnn':
            W_est , _, _= daggnn_nonlinear(model, adaptive_model, X, train_loader,true_graph=B_true)
        elif args.modeltype=='grandag':
            W_est , _, _= grandag_nonlinear(model, adaptive_model, X, train_loader,true_graph=B_true)


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
                f.write(f'modeltype:{args.modeltype}\n')
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
    with open(f_dir+"stats_"+str(args.lambda1)+"_"+str(args.lambda2)+"_"+args.modeltype+reweight_str+".json",'w') as f:
        json.dump(tot_perf,f)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=10)
    main(args.trial,args.seed)
