#!/bin/bash
temperature=40
GRAPH_TYPE="SF"
S=40
D=20
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.01 --lambda2 0.01 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.01 --lambda2 0.01 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.008 --lambda2 0.008  --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.008 --lambda2 0.008  --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.005 --lambda2 0.005 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.005 --lambda2 0.005 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.002 --lambda2 0.002 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.002 --lambda2 0.002 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.015 --lambda2 0.015 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.015 --lambda2 0.015 --linear  --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.02 --lambda2 0.02 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.02 --lambda2 0.02 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.025 --lambda2 0.025 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.025 --lambda2 0.025 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.03 --lambda2 0.03 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.03 --lambda2 0.03 --linear --temperature $temperature --graph_type $GRAPH_TYPE

GRAPH_TYPE="ER"
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.01 --lambda2 0.01 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.01 --lambda2 0.01 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.008 --lambda2 0.008  --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.008 --lambda2 0.008  --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.005 --lambda2 0.005 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.005 --lambda2 0.005 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.002 --lambda2 0.002 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.002 --lambda2 0.002 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.015 --lambda2 0.015 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.015 --lambda2 0.015 --linear  --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.02 --lambda2 0.02 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.02 --lambda2 0.02 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.025 --lambda2 0.025 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.025 --lambda2 0.025 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 1 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.03 --lambda2 0.03 --linear --temperature $temperature --graph_type $GRAPH_TYPE
python nonlinear_adp.py --run_mode 0 --modeltype notears --s0 $S --d $D --n 2000 --trial 10 --seed 0 --lambda1 0.03 --lambda2 0.03 --linear --temperature $temperature --graph_type $GRAPH_TYPE
