#!/bin/bash

model_name=$1
device=$2

for r in {1..10}
do
	# m = 0
	python3 ./train_and_eval.py --model_name $model_name --norm_type contextnorm --lr 5e-4 --task dist3 --m_holdout 0 --epochs 50 --run $r --device $device
	# m = 50
	python3 ./train_and_eval.py --model_name $model_name --norm_type contextnorm --lr 5e-4 --task dist3 --m_holdout 50 --epochs 50 --run $r --device $device
	# m = 85
	python3 ./train_and_eval.py --model_name $model_name --norm_type contextnorm --lr 5e-4 --task dist3 --m_holdout 85 --epochs 50 --run $r --device $device
	# m = 95 
	python3 ./train_and_eval.py --model_name $model_name --norm_type contextnorm --lr 5e-4 --task dist3 --m_holdout 95 --epochs 150 --run $r --device $device
done