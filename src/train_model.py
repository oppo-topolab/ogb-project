import os
import time
import argparse
import math

import numpy as np

import dgl

import torch
import torch.nn.functional as F
import torch.optim as optim

# udf
from data import load_ogbn_dataset, preprocess
from model import LocalGlobalGNN
from utils import (
    plot_acc_curve, plot_loss_curve, seed_everything
)


epsilon = 1 - math.log(2)
device = None
dataset = "ogbn-arxiv"
n_node_feats, n_classes = 0, 0


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats
    
    print("=" * 50)
    print(f"Local Tower used: '{args.l_model}'")
    print(f"Global Tower used: '{args.g_model}'")

    model = LocalGlobalGNN(
        in_feat=n_node_feats_,
        n_hidden=args.n_hidden,
        n_classes=n_classes,
        l_layers=args.l_layers,
        g_layers=args.g_layers,
        n_mlp=args.n_mlp,
        dropout=args.dropout,
        activation=F.relu,
        l_model=args.l_model,
        g_model=args.g_model,
        aggregator_type=args.aggregator_type,
        residual=args.use_residual
    )

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"# Model Params: {n_params:d}")

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_labels(feat, labels, idx):
    # feat = [X, Y]
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(args, model, graphs, labels, train_idx, val_idx, test_idx, optimizer, evaluator):
    model.train()

    graph, knn_g = graphs  # unpack

    feat = graph.ndata["feat"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_labels_idx = train_idx[mask] # label used part
        train_pred_idx = train_idx[~mask]  # need to predict part

        feat = add_labels(feat, labels, train_labels_idx)  # Y --> onehot --> mask partially
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, knn_g, feat)

    if args.n_label_iters > 0:  # label propagation ?
        unlabel_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            torch.cuda.empty_cache()
            # pred softmax fill in the feat Y part (feat=[X, Y]) for unlabel samples
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, knn_g, feat)

    # need to train part (part of original train)
    loss = custom_loss_function(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    optimizer.step()

    return evaluator(pred[train_idx], labels[train_idx]), loss.item()


@torch.no_grad()
def evaluate(args, model, graphs, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()

    # prepare data
    graph, knn_g = graphs
    feat = graph.ndata["feat"]

    if args.use_labels:
        feat = add_labels(feat, labels, train_idx)  # full train used as 

    # model forward
    pred = model(graph, knn_g, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, knn_g, feat)

    # compute loss
    train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
    val_loss = custom_loss_function(pred[val_idx], labels[val_idx])
    test_loss = custom_loss_function(pred[test_idx], labels[test_idx])

    # evaluate performance
    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
        train_loss.item(),
        val_loss.item(),
        test_loss.item(),
        pred,
    )


def save_checkpoint(save_path, model, pred, run_num):
    fname = os.path.join(save_path, f'best_pred_run{run_num}.pt')
    mname = os.path.join(save_path, f'model_run{run_num}.pt')
    print('Saving prediction.......')
    torch.save(pred.cpu(), fname)
    torch.save(model.state_dict(), mname)


def run(args, graphs, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    """
    graphs pack two [g, knn_g], g with node feature, knn_g just graph
    """
    work_path = os.path.join(args.ckpt_path, f'{n_running}')
    os.makedirs(work_path, exist_ok=True)

    # define model and optimizer
    model = gen_model(args).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
    final_pred = None

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)

        acc, loss = train(args, model, graphs, labels, train_idx, val_idx, test_idx, optimizer, evaluator)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graphs, labels, train_idx, val_idx, test_idx, evaluator
        )

        toc = time.time()
        total_time += toc - tic

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_val_acc = val_acc
        #     final_test_acc = test_acc
        #     final_pred = pred

        if val_acc > best_val_acc:
            best_val_acc = val_acc    
            final_test_acc = test_acc
            final_pred = pred
            save_checkpoint(work_path, model, final_pred, n_running)

        if epoch == args.n_epochs or epoch % args.log_every == 0:
            print(
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)

    print("=" * 50)
    print(f"Best_Val_Acc: {best_val_acc:.4f}, Final_Test_Acc: {final_test_acc:.4f}")
    print("=" * 50)

    # plot learning curves
    if args.plot_curves:
        # plot acc vs epoch
        plot_acc_curve(
            args.n_epochs, accs, train_accs, val_accs, test_accs, 
            os.path.join(work_path, f"model_acc_{n_running}.png")
        )

        # plot loss vs epoch
        plot_loss_curve(
            args.n_epochs, losses, train_losses, val_losses, test_losses, 
            os.path.join(work_path, f"model_loss_{n_running}.png")
        )

    # save final pred (logits without softmax from model)
    if args.save_pred:
        torch.save(
            F.softmax(final_pred, dim=1), 
            os.path.join(work_path, f"final_pred{n_running}.pt")
        )

    return best_val_acc, final_test_acc


def parse_args(desc):
    parser = argparse.ArgumentParser(description=desc)

    # Data Params
    parser.add_argument(
        "--ogbn_dataset", type=str, default='ogbn-arxiv',
        choices=['ogbn-arxiv', 'ogbn-products',
                 'ogbn-proteins', 'ogbn-papers100M'],
        help="ogb node property prediction dataset name"
    )
    parser.add_argument(
        "--data_path", type=str, 
        default="/home/notebook/data/group/topolab/graphtools/datamarket/public/ogbn-arxiv",
        help="ogb data path for local storage"
    )
    parser.add_argument(
        "--emb_path", type=str,
        default="/home/notebook/code/personal/OGB/CodePackage/ogbn-arxiv/X.all.xrt-emb.npy",
        help="extra embeddings for ogbn dataset, like: GIANT-XRT"
    )
    parser.add_argument(
        "--knn_g_path", type=str,
        default="/home/notebook/code/personal/OGB/ogb_lsc/self_dev/knn_fg20.bin",
        help="knn graph data path"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="",
        help="torch model checkpoints/best pred save path"
    )
    parser.add_argument(
        "--graph_transform", type=str, default="none",
        choices=['none', 'rg', 'bg'],
        help="graph transform for original OGBN dataset, rg: reversed, bg: bi-directional"
    )

    # Model Params
    parser.add_argument("--l_model", type=str, default='sage',
        choices=['sage', 'gat', 'agdn'],
        help="GNN backbone used in local tower"
    )
    parser.add_argument("--g_model", type=str, default='sage',
        choices=['sage'],
        help="GNN backbone used in global tower"
    )
    parser.add_argument("--self_loop", action="store_true", default=False, help="add self loop to graph in preprocess")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--use_residual", action="store_true", default=False, help="whether to use residual connection (Tri-Tower)")
    parser.add_argument("--n_hidden", type=int, default=256, help="number of hidden units for gnn")
    parser.add_argument("--l_layers", type=int, default=3, help="number of hidden layers for local gnn tower")
    parser.add_argument("--g_layers", type=int, default=2, help="number of hidden layers for global gnn tower")
    parser.add_argument("--n_mlp", type=int, default=2, help="number of layers for final MLP predictor")
    parser.add_argument("--aggregator_type", type=str, default="mean",
        choices=['mean', 'gcn', 'pool', 'lstm'],
        help="Aggregator type for GraphSAGE: mean/gcn/pool/lstm"
    )

    # Training Params
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--gpu", type=int, default=-1, help="use gpu for computation")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--early_stopping_rounds", type=int, default=0, help="early stopping rounds, if > 0")
    parser.add_argument("--save_pred", action="store_true", default=False, help="save final predictions logits & labels")
    parser.add_argument("--label_weight", action="store_true", default=False, help="whether to use label weight to adjust loss")

    # Experiment Params
    parser.add_argument("--n_runs", type=int, default=10, help="running times")
    parser.add_argument("--use_labels", action="store_true", default=False, help="Use labels in the training set as input features.")
    parser.add_argument("--n_label_iters", type=int, default=0, help="number of label iterations(propagation)")
    parser.add_argument("--mask_rate", type=float, default=0.5, help="mask rate for label usage")
    parser.add_argument("--log_every", type=int, default=20, help="log every LOG_EVERY epochs")
    parser.add_argument("--plot_curves", action="store_true", help="plot learning curves")

    # C&S params
    parser.add_argument("--cs", action="store_true", default=False, help="use Correct & Smooth after original model")
    parser.add_argument("--num-correction-layers", type=int, default=50)
    parser.add_argument("--correction-alpha", type=float, default=0.979)
    parser.add_argument("--correction-adj", type=str, default="DAD", choices=['DAD', 'AD', 'DA'])
    parser.add_argument("--num-smoothing-layers", type=int, default=50)
    parser.add_argument("--smoothing-alpha", type=float, default=0.756)
    parser.add_argument("--smoothing-adj", type=str, default="DAD", choices=['DAD', 'AD', 'DA'])
    parser.add_argument("--autoscale", action="store_true")
    parser.add_argument("--scale", type=float, default=20.0)

    args = parser.parse_args()

    print('Show all arguments configuration...')
    for k, v in args.__dict__.items():
        print(f"{k:>25s}: {v}")

    return args


def main():
    global device, n_node_feats, n_classes, epsilon

    args = parse_args("LGGNN+LabelUse+self-KD")

    if not args.use_labels and args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    # device
    if torch.cuda.is_available():
        device = torch.device("cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")

    # load data & preprocess
    g, feat, labels, train_idx, val_idx, test_idx, evaluator = load_ogbn_dataset(
        datapath=args.data_path, 
        dataname=args.ogbn_dataset, 
        embpath=args.emb_path
    )
    g = preprocess(g, feat, tf=args.graph_transform, self_loop=args.self_loop)
    knn_g, _ = dgl.load_graphs(args.knn_g_path, [0])
    knn_g = knn_g[0]
    # knn_g = knn_g.remove_self_loop().add_self_loop()
    # should not add self loop to KNN Graph again, after adding to original graph

    # push data to 'device'
    g, knn_g, labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (g, knn_g, labels, train_idx, val_idx, test_idx)
    )

    n_node_feats = feat.shape[1]
    n_classes = int(labels.max() - labels.min()) + 1

    # run
    val_accs, test_accs = [], []
    for i in range(args.n_runs):
        seed_everything(args.seed + i)
        val_acc, test_acc = run(args, (g, knn_g), labels, train_idx, val_idx, test_idx, evaluator, i + 1)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"Average test accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    # print(f"Number of params: {count_parameters(args)}")


if __name__ == "__main__":
    main()
