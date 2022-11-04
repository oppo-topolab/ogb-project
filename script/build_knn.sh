#!/bin/bash


# =============================================================================
#                    Build KNN Graph from Node Embeddings
# =============================================================================


DATASET=ogbn-arxiv
RUNTIME=$(date -d '+8 hour' +"%Y%m%d%H%M%S")
echo "Build KNN Graph: ${DATASET}"
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


# Check KNN Graph Save Path
if [ ! -d ${KNNG_PATH} ]; then
    mkdir -p ${KNNG_PATH}
fi


python3 ${CODE_PATH}/graph_utils.py \
    --data_path ${DATA_PATH}/${DATASET} \
    --emb_path ${EMB_PATH}/${DATASET}/X.all.xrt-emb.npy \
    --graph_transform bg \
    --n_neigh 30 \
    --p -1 \
    --save_path ${KNNG_PATH}
