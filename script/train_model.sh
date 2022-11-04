#!/bin/bash

DATASET=ogbn-arxiv

RUNTIME=$(date -d '+8 hour' +"%Y%m%d%H%M%S")
# RUNTIME=20221012110648

echo "Train LG Model: ${DATASET}"
echo "RunTime: ${RUNTIME}"


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

python3 ${CODE_PATH}/train_model.py \
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
    --aggregator_type mean \
    --use_labels \
    --n_label_iters 1 \
    --seed 0 \
    --n_epochs 2000 \
    --lr 2e-4 \
    --weight_decay 5e-4 \
    --gpu 0 \
    --n_runs 10 \
    --mask_rate 0.5 \
    --plot_curves \
    --save_pred \
    --self_loop
