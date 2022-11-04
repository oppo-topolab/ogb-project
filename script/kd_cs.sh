#!/bin/bash

DATASET=ogbn-arxiv

# Fill Runtime here to load a pre-trainned model for CS or KD
# RUNTIME=20221017140842  # exp[2022-10-17] without self-loop, LG-LabelReuse(1) 
# RUNTIME=20221019095838  # exp[2022-10-19] with self-loop for Graph, LG-LabelReuse(1)
RUNTIME=20221028140319  # exp[2022-10-28] self-loop LG+LabelReuse(1)  lr=2e-4 up up

echo ${RUNTIME}

# Work Path
WORK_PATH=

# Code Path
CODE_PATH=${WORK_PATH}/src

# KNN Graph Path (build in advance)
KNNG_PATH=${WORK_PATH}/knng

# Checkpoints Path (save model pt)
LOG_PATH=${WORK_PATH}/logs/${RUNTIME}

# OGB DataSet Path (OGB package data saved, Top level)
DATA_PATH=

# Extra Embedding Path (e.g. GIANT-XRT file)
EMB_PATH=


# run each time, save in a new directory via ${RUNTIME}
if [ ! -d ${LOG_PATH} ]; then
    mkdir -p ${LOG_PATH}
fi


# set dgl backend
export DGLBACKEND=pytorch

# use C&S
python3 ${CODE_PATH}/kd_cs.py \
    --ogbn_dataset ${DATASET} \
    --data_path ${DATA_PATH}/${DATASET} \
    --emb_path ${EMB_PATH}/${DATASET}/X.all.xrt-emb.npy \
    --knn_g_path ${KNNG_PATH}/knn_bg_30.bin \
    --ckpt_path ${LOG_PATH} \
    --graph_transform bg \
    --l_layers 3 \
    --g_layers 1 \
    --l_model sage \
    --g_model sage \
    --n_hidden 256 \
    --n_mlp 2 \
    --dropout 0.2 \
    --self_loop \
    --aggregator_type mean \
    --seed 0 \
    --n_epochs 2000 \
    --lr 2e-4 \
    --weight_decay 5e-4 \
    --gpu 0 \
    --n_runs 10 \
    --use_labels \
    --n_label_iters 1 \
    --mask_rate 0.5 \
    --cs \
    --correction-alpha 0.87 \
    --smoothing-alpha 0.81 \
    --correction-adj AD \
    --autoscale \
    --save_pred


# # use Self-KD (uncomment this and comment CS)
# python3 ${CODE_PATH}/kd_cs.py \
#     --ogbn_dataset ${DATASET} \
#     --data_path ${DATA_PATH}/${DATASET} \
#     --emb_path ${EMB_PATH}/${DATASET}/X.all.xrt-emb.npy \
#     --knn_g_path ${KNNG_PATH}/knn_bg_30.bin \
#     --ckpt_path ${LOG_PATH} \
#     --graph_transform bg \
#     --l_model sage \
#     --g_model sage \
#     --l_layers 3 \
#     --g_layers 1 \
#     --n_hidden 256 \
#     --n_mlp 2 \
#     --dropout 0.2 \
#     --self_loop \
#     --aggregator_type mean \
#     --seed 0 \
#     --n_epochs 2000 \
#     --lr 2e-4 \
#     --weight_decay 5e-4 \
#     --gpu 0 \
#     --n_runs 10 \
#     --use_labels \
#     --n_label_iters 1 \
#     --mask_rate 0.5 \
#     --self_kd \
#     --alpha 0.95 \
#     --temp 0.7
#     # --save_pred
