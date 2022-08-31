#!/bin/bash
python nonlinear_adp.py --lambda1 0.002 --lambda2 500  --run_mode 0 --use_golem --s0 30 --d 10 --n 1000 --trial 10 --seed 0

python nonlinear_adp.py --lambda1 0.002 --lambda2 500  --run_mode 1 --use_golem --s0 30 --d 10 --n 1000 --trial 10 --seed 0