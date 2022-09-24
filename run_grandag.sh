#!/bin/bash

temperature=20
MODEL_TYPE='grandag'

GRAPH_TYPE="SF"
D=10
S=20
python nonlinear_adp.py --run_mode 1 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log
python nonlinear_adp.py --run_mode 0 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log

D=10
S=40
python nonlinear_adp.py --run_mode 1 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log
python nonlinear_adp.py --run_mode 0 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log


GRAPH_TYPE="ER"
D=20
S=40
python nonlinear_adp.py --run_mode 1 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log
python nonlinear_adp.py --run_mode 0 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log

D=20
S=80
python nonlinear_adp.py --run_mode 1 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log
python nonlinear_adp.py --run_mode 0 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log

GRAPH_TYPE="SF"
D=20
S=40
python nonlinear_adp.py --run_mode 1 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log
python nonlinear_adp.py --run_mode 0 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log

D=20
S=80
python nonlinear_adp.py --run_mode 1 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log
python nonlinear_adp.py --run_mode 0 --modeltype $MODEL_TYPE --s0 $S --d $D --n 2000 --trial 10 --seed 0 --temperature $temperature --graph_type $GRAPH_TYPE --pns --cam-pruning --no-w-adjs-log