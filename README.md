# Report Performance for ogbn-arxiv

## Result on 'ogbn-arxiv'

Here, we demonstrate the following performance on the [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) dataset from [OGB](https://ogb.stanford.edu/docs/nodeprop/)

|  Model   |  Test Acc    |  Val Acc   | \#Params | Hardware |
| :---: | :---: | :---: | :---: | :---: |
|  LGGNN+LabelReuse(1)+CS  |   **0.7570 ± 0.0018**  |  0.7687 ± 0.0005   | 1,161,640 | Tesla V100 (32GB) |

The best performance got from Local&Global Model with LabelReuse(just 1 iterations) and post processing with C&S in train dataset.

We run the model 10 times with seed from 0~9 to get the mean and standard deviation of test accuracy with ogb evaluator.


## Reproduction Instructions

### 0. Requirements

the core python3 packages in this experiments

```
numpy==1.21.5
scipy==1.7.3
torch==1.10.2+cu113
dgl-cu113==0.9.0
ogb==1.3.4
pynndescent==0.5.6
```


### 1. Check dataset and environment variables in each scripts

```shell
# Experiment Dataset(fixed here)
DATASET=ogbn-arxiv

# Work Path(where you replace ogb_arxiv)
WORK_PATH=

# Code Path(src)
CODE_PATH=${WORK_PATH}/src

# KNN Graph Path (build in advance)
KNNG_PATH=${WORK_PATH}/knng

# Checkpoints Path (save model pt)
LOG_PATH=${WORK_PATH}/logs/${RUNTIME}

# OGB DataSet Path (OGB package data saved, Top level)
DATA_PATH=

# Extra Embedding Path (e.g. GIANT-XRT file)
EMB_PATH=

```

### 2. Build KNN Graph

The KNN Graph is built with node features, using the shell script `build_knn.sh`, export to DGL graph format and saved in `KNNG_PATH`

```shell
python3 ${CODE_PATH}/graph_utils.py \
    --data_path ${DATA_PATH}/${DATASET} \
    --emb_path ${EMB_PATH}/${DATASET}/X.all.xrt-emb.npy \
    --graph_transform bg \
    --n_neigh 30 \
    --p -1 \
    --save_path ${KNNG_PATH}
```


### 3. Train the LGGNN Model

LGGNN model is pre-trained using the script `train_model.sh`, model(torch, pt format) and `best val`'s predictions will be save in `LOG_PATH`

```shell
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
    --seed 0 \
    --n_epochs 2000 \
    --lr 2e-4 \
    --weight_decay 5e-4 \
    --gpu 0 \
    --n_runs 10 \
    --use_labels \
    --n_label_iters 1 \
    --mask_rate 0.5 \
    --plot_curves \
    --save_pred \
    --self_loop
```

### 4. Using C&S or Self-KD to boost the performance

Run C&S with the pre-trained LGGNN model using script `kd_cs.sh`, it will load the dumped model(`model_dict`) and run C&S.

```shell
# set the pre-trained model first !!! It saved by RUNTIME in LOG_PATH
RUNTIME=YYYYMMDDHHMMSS
```

```shell
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
```

You can use Self-KD techniques also with script `kd_cs.sh`, comment the C&S part and uncomment Self-KD part. But in our experiments Self-KD will not boost the performance on test set.

```shell
# set dgl backend
export DGLBACKEND=pytorch

# use Self-KD (uncomment this and comment CS)
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
    --self_kd \
    --alpha 0.95 \
    --temp 0.7
```
