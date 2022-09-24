# coding=utf-8
"""
GraN-DAG

Copyright © 2019 Authors of Gradient-Based Neural DAG Learning

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os

import matplotlib

# To avoid displaying the figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_weighted_adjacency(weighted_adjacency, gt_adjacency, exp_path, name="abs-weight-product", mus=None,
                            lambdas=None, iter=None, plotting_callback=None):
    """iter is useful to deal with jacobian, it will interpolate."""
    weighted_adjacency=np.stack(weighted_adjacency)
    num_vars = weighted_adjacency.shape[1]
    max_value = 0
    fig, ax1 = plt.subplots()

    # Plot weight of incorrect edges
    for i in range(num_vars):
        for j in range(num_vars):
            if gt_adjacency[i, j]:
                continue
            else:
                color = 'r'
            y = weighted_adjacency[:, i, j]
            num_iter = len(y)
            if iter is not None and len(y) > 1:
                num_iter = iter + 1
                y = np.interp(np.arange(iter + 1), np.linspace(0, iter, num=len(y), dtype=int), y)
            ax1.plot(range(num_iter), y, color, linewidth=1)
            if len(y) > 0: max_value = max(max_value, np.max(y))

    # Plot weight of correct edges
    for i in range(num_vars):
        for j in range(num_vars):
            if gt_adjacency[i, j]:
                color = 'g'
            else:
                continue
            y = weighted_adjacency[:, i, j]
            num_iter = len(y)
            # plt.plot(range(len(weighted_adjacency[:, 0, 0])), y, color, alpha=0.1, linewidth=1)
            if iter is not None and len(y) > 1:
                num_iter = iter + 1
                y = np.interp(np.arange(iter + 1), np.linspace(0, iter, num=len(y), dtype=int), y)
            ax1.plot(range(num_iter), y, color, linewidth=1)
            if len(y) > 0: max_value = max(max_value, np.max(y))

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(name)
    ax1.set_yscale("log")

    if mus is not None or lambdas is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel(r'$\frac{\mu}{2}$ and $\lambda$', color='blue')
        if mus is not None:
            ax2.plot(range(len(mus)), 0.5 * np.array(mus), color='blue', linestyle="dashed", linewidth=1,
                     label=r"$\frac{\mu}{2}$")
        if lambdas is not None:
            ax2.plot(range(len(lambdas)), lambdas, color='blue', linestyle="dotted", linewidth=1, label=r"$\lambda$")
        ax2.legend()
        ax2.set_yscale("log")
        ax2.tick_params(axis='y', labelcolor='blue')

    fig.tight_layout()
    if plotting_callback is not None:
        plotting_callback("weighted_adjacency", fig)
    fig.savefig(os.path.join(exp_path, name + '.png'))
    fig.clf()

def plot_adjacency(adjacency, gt_adjacency, exp_path, name=''):
    plt.clf()
    f, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1)
    print(adjacency)
    print(gt_adjacency)
    sns.heatmap(adjacency, ax=ax1, cbar=False, vmin=-1, vmax=1, cmap="Blues_r", xticklabels=False, yticklabels=False)
    sns.heatmap(gt_adjacency, ax=ax2, cbar=False, vmin=-1, vmax=1, cmap="Blues_r", xticklabels=False, yticklabels=False)
    sns.heatmap(adjacency - gt_adjacency, ax=ax3, cbar=False, vmin=-1, vmax=1, cmap="Blues_r", xticklabels=False,
                yticklabels=False)

    ax1.set_title("Learned")
    ax2.set_title("Ground truth")
    ax3.set_title("Learned - GT")

    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(exp_path, 'adjacency' + name + '.png'))

if __name__ == '__main__':
    

    import os
    import json
    xs=[]
    ys={}
    method="golem"
    temperature=20
    reweight=f"_reweight_{temperature}"
    #reweight="_reweight"
    setting="10_20_2000"

    best_shd_base=1e10
    best_perf_base=""
    best_shd_rescore=1e10
    best_perf_rescore=""


    path=f"/storage/wcma/DAG-NoCurl/reweight_experiment/linear/{setting}/ER_gauss"
    for file in os.listdir(path):
        if file.endswith(f"{method}_cut0.01{reweight}.json"):
        #if file.endswith(f"{method}{reweight}.json"):
            # if float(file.split("_")[1])==0.015:
            #     continue
            xs.append(float(file.split("_")[1]))
            abs_path=os.path.join(path, file)
            with open(abs_path) as f:
                data=json.load(f)
            perf_str=file.split("_")[1]+"\n"
            flag=0
            for key,value in data.items():
                if key not in ys:
                    ys[key]=[[],[]]
                ys[key][0].append(value['mean'])
                ys[key][1].append(value['std'])
                if key =="shd":
                    if value['mean']<best_shd_rescore:
                        best_shd_rescore=value['mean']
                        flag=1
                perf_str+=f"{key}:{value['mean']}+-{value['std']}\n"
            
            if flag==1:
                best_perf_rescore=perf_str
    
    xs_ori=[]
    ys_ori={}
    for file in os.listdir(path):
        if file.endswith(f"{method}_cut0.01.json"):
        #if file.endswith(f"{method}.json"):
            # if float(file.split("_")[1])<0.001:
            #     continue
            xs_ori.append(float(file.split("_")[1]))
            abs_path=os.path.join(path, file)
            with open(abs_path) as f:
                data=json.load(f)
            perf_str=file.split("_")[1]+"\n"
            flag=0
            for key,value in data.items():
                if key not in ys_ori:
                    ys_ori[key]=[[],[]]
                ys_ori[key][0].append(value['mean'])
                ys_ori[key][1].append(value['std'])
                if key =="shd":
                    if value['mean']<best_shd_base:
                        best_shd_base=value['mean']
                        flag=1
                perf_str+=f"{key}:{value['mean']}+-{value['std']}\n"
            
            if flag==1:
                best_perf_base=perf_str
    
    idx=np.argsort(np.array(xs))
    idx_ori=np.argsort(np.array(xs_ori))
    xs=np.array(xs)[idx]
    xs_ori=np.array(xs_ori)[idx_ori]

    plt.figure(figsize=(16,8))
    cnt=0
    for key,item in ys.items():
        plt.subplot(f"23{cnt}")
        means=np.array(item[0])[idx]
        stds=np.array(item[1])[idx]
        means_ori=np.array(ys_ori[key][0])[idx_ori]
        stds_ori=np.array(ys_ori[key][1])[idx_ori]
        plt.errorbar(xs, means, yerr=stds,color='r',fmt='-o',label=method+"_reweight")
        plt.errorbar(xs_ori, means_ori, yerr=stds_ori,color='b',fmt='-o',label=method)
        plt.title(key)
        plt.legend()
        cnt+=1
    plt.suptitle(setting)
    plt.savefig(os.path.join(path,f"final_{temperature}.jpg"))
    plt.cla()

    print("baseline:")
    print(best_perf_base)
    print("rescore:")
    print(best_perf_rescore)
        

        