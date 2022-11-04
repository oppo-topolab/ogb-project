import os
import numpy as np
import torch
import dgl
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


def compute_norm(g):
    degs = g.in_degrees().float().clamp(min=1)
    
    # D^{-1}
    deg_inv = torch.pow(degs, -1)
    
    # D^{-1/2}
    deg_isqrt = torch.pow(degs, -0.5)
    
    # D^{1/2}
    deg_sqrt = torch.pow(degs, 0.5)

    return deg_inv, deg_sqrt, deg_isqrt


def preprocess(g, feat, tf='bg', self_loop=False, norm=False):
    """
    pre-process the original graph 'g'
    [1] graph transformation: to reversed graph, to bi-directional graph (add reversed edges)
    [2] add self-loop to graph 'g'
    [3] optional: calculate some normalization factors to edges
    """
    # graph transformation
    if tf == 'rg':  # use reversed graph
        g = dgl.reverse(g)

    elif tf == 'bg':  # use bi-directional graph (add reversed edges)
        g = dgl.to_bidirected(g)
    
    elif tf == 'none':  # without transformation
        pass

    else:
        raise Warning(f"invalid graph transformation: '{tf}'")

    # add self-loop
    if self_loop:
        print(f"Total edges before adding self-loop {g.number_of_edges()}")
        g = g.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {g.number_of_edges()}")

    # Create all sparse matrices allowed for the graph.
    # https://docs.dgl.ai/generated/dgl.DGLGraph.create_formats_.html?highlight=create_formats_
    g.create_formats_()

    # fill the node features
    g.ndata["feat"] = feat

    if norm:
        # do nomalization only once
        deg_inv, deg_sqrt, deg_isqrt = compute_norm(g)
        
        # edge data: \frac{1}{\sqrt{d_i d_j}}
        g.srcdata.update({"src_norm": deg_isqrt})
        g.dstdata.update({"dst_norm": deg_isqrt})
        g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm"))

        # edge data: \sqrt{\frac{d_j}{d_i}}
        g.srcdata.update({"src_norm": deg_isqrt})
        g.dstdata.update({"dst_norm": deg_sqrt})
        g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm_adjust"))

        # edge data: \frac{1}{d_j}
        g.edata["sage_norm"] = deg_inv[g.edges()[1]]

    return g


def get_ogb_evaluator(dataset='ogbn-arxiv'):
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.argmax(dim=-1, keepdim=True),
        })["acc"]


def load_ogbn_dataset(datapath, dataname='ogbn-arxiv', embpath=None):
    """
    load ogb npp dataset, use extra embedding if possible, e.g. GIANT-XRT
    """
    assert dataname in ['ogbn-arxiv', 'ogbn-products']

    # performance evaluator
    evaluator = get_ogb_evaluator(dataname)

    # original data graph & label
    dataset = DglNodePropPredDataset(name=dataname, root=datapath)
    g, labels = dataset[0]
    # labels = labels.squeeze()  # ?

    # train/val/test division
    split_nid = dataset.get_idx_split()
    train_nid, val_nid, test_nid = split_nid["train"], split_nid["valid"], split_nid["test"]

    # use extra embeddings(e.g. GIANT-XRT) if possible
    if embpath is not None:
        assert embpath.split('.')[-1] == 'npy', "unsupported embedding data type!"
        emb = np.load(embpath, allow_pickle=True)

        assert g.number_of_nodes() == emb.shape[0], "mismatch number of nodes in graph & embedding"
        print("----- Use Extra Embeddings -----")
        feat = torch.from_numpy(emb)

    else:
        print("----- Use Original Embeddings -----")
        feat = g.ndata['feat']
        
    print(f"""
        Dimension: {feat.shape[0]:d}
        N_Samples: {feat.shape[1]:d}
    """)

    # data statistics
    in_feats = feat.shape[1]

    print(f"""----- Graph statistics -----
        DataSet: {dataname:s}
        # Edges: {g.number_of_edges():d}
      # Classes: {dataset.num_classes:d}
    # Train Set: {len(train_nid):d}
      # Val Set: {len(val_nid):d}
     # Test Set: {len(test_nid):d}
    """)

    return g, feat, labels, train_nid, val_nid, test_nid, evaluator
