#!/bin/bash
D=10
S=20
temperature=20
GRAPH_TYPE="ER"
python nonlinear_adp.py --run_mode 1 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.01 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.01 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.008 --lambda2 500  --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.008 --lambda2 500  --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.005 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.005 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.002 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.002 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.015 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.015 --lambda2 500 --linear  --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.02 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.02 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.025 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.025 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.03 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype golem --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.03 --lambda2 500 --linear --temperature $temperature --graph_type $GRAPH_TYPE
