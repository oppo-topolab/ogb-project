import os
import argparse
import torch
import numpy as np
import dgl
from scipy.sparse import coo_matrix
from ogb.nodeproppred import DglNodePropPredDataset


def get_knn_from_emb(feat, k=30, metric='cosine', save_emb=None):
    """
    build KNN Graph from feature embeddings, used method in 
    <Efficient k-nearest neighbor graph construction for generic similarity measures>
    also used in GAS <Spam Review Detection with Graph Convolutional Networks> Ref.3

    :param feat: numpy.ndarray, N x D, N for node, D for feature

    :param k: int, number of nearest neighbors, default=30, determines by mean degree from original graph

    :param metric: str, metric used for calculating distance

    :param save_emb: str, data path for file
    """
    import pynndescent
    
    assert metric in pynndescent.distances.named_distances, f"unsupported metric: '{metric}'"

    index = pynndescent.NNDescent(feat, metric=metric, n_neighbors=k+1)  # it takes a while to build index!
    adj, dist = index.neighbor_graph

    if save_emb is not None:
        np.save(os.path.join(save_emb, f'knn_adj_{k}.npy'), adj[:, 1])

    return adj[:, 1:], dist[:, 1:]


def filter_graph_v3(knn, g, p=-1, use_reverse=True, save_g=None):
    """
    KNN Graph Adjacency Matrix diffs Original Graph Adjacency Matrix
    just like <Spam Review Detection with Graph Convolutional Networks> do P6

    :param knn: pack (numpy.ndarray, numpy.ndarray), adjacency list of KNN Graph & distance matrix

    :param g: dgl.DGLGraph, original graph object(homogeneous)

    :param p: int, percentile between 0 to 100, if None, do not filter 

    :param use_reverse: bool, default=True, build reversed knn graph
    """
    adj, dist = knn  # unpack data

    # use 'scipy.sparse.coo.coo_matrix' to replace  'torch.sparse_coo_tensor'
    N, D = adj.shape
    
    # build a sparse matrix: sparse.coo_matrix((V,(I,J)),shape=(N,N))
    I = np.arange(N).repeat(D)
    J = adj.flatten()

    # filter by distance threshold
    if p >= 0:
        cutoff = np.percentile(dist.flatten(), p)
        chosen = (dist.flatten() <= cutoff)
        I = I[chosen]
        J = J[chosen]
    
    V = np.ones(I.shape[0])

    index = (J, I) if use_reverse else (I, J)
    A_knn = coo_matrix((V, index), shape=(N, N))

    A_old = g.adjacency_matrix(scipy_fmt='coo')
    # core instance: use knn graph, and drop exists edges in original graph
    A_new = A_knn * (A_knn - A_old)
    
    g_new = dgl.from_scipy(A_new)  # same number of nodes as 'g'

    if save_g is not None:
        dgl.save_graphs(os.path.join(save_g, 'knn_fg.bin'), g_new)
    
    return A_knn, A_new, g_new


def filter_graph_v2(adj, g, use_reverse=True, save_g=None):
    """
    KNN Graph Adjacency Matrix diffs Original Graph Adjacency Matrix
    just like <Spam Review Detection with Graph Convolutional Networks> do P6

    :param adj: numpy.ndarray, adjacency list of KNN Graph, row for src, column for dst

    :param g: dgl.DGLGraph, original graph object(homogeneous)

    :param use_reverse: bool, default=True, build reversed knn graph
    """
    # 改用 scipy.sparse.coo.coo_matrix 替代 torch 的 sparse_coo_tensor
    N, D = adj.shape
    A_old = g.adjacency_matrix(scipy_fmt='coo')  # scipy.sparse.coo.coo_matrix
    
    # sparse.coo_matrix((V,(I,J)),shape=(4,4))
    V = np.ones(N * D)
    I = np.arange(N).repeat(D)
    J = adj.flatten()

    index = (J, I) if use_reverse else (I, J)
    A_knn = coo_matrix((V, index), shape=(N, N))

    A_new = A_knn * (A_knn - A_old)  # core [high efficient], drop exists edges in original graph
    
    g_new = dgl.from_scipy(A_new)  # same number of nodes as 'g'

    if save_g is not None:
        dgl.save_graphs(os.path.join(save_g, 'knn_fg.bin'), g_new)
    
    return A_knn, A_new, g_new


def filter_graph(adj, g):
    N, D = adj.shape
    A_old = g.adj()

    I = torch.arange(N).repeat_interleave(D)
    J = torch.from_numpy(adj).flatten()
    V = torch.ones(N * D)
    A_knn = torch.sparse_coo_tensor(
        torch.cat([I.unsqueeze(0), J.unsqueeze(0)], dim=0), V, (N, N)
    )
    
    A_knn_new = A_knn * (A_knn - A_old)
    
    return A_knn, A_knn_new


def build_knn_graph(g, emb, n_neigh=20, p=None):
    # build knn graph via embeddings
    adj, dist = get_knn_from_emb(emb, k=n_neigh)
    # may filter the dist via absolute value
    
    _, _, g_new = filter_graph_v3((adj, dist), g, p=p) if p is not None else filter_graph_v2(adj, g)

    return g_new


if __name__ == "__main__":
    # build knn graph in advance (run this before train the model)
    
    parser = argparse.ArgumentParser()

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
        "--graph_transform", type=str, default="none",
        choices=['none', 'rg', 'bg'],
        help="graph transform for original OGBN dataset, rg: reversed, bg: bi-directional"
    )
    parser.add_argument(
        "--n_neigh", type=int, default=20,
        help="knn graph number of neighbors"
    )
    parser.add_argument(
        "--p", type=int, default=-1,
        help="percentile for filter absolute distance"
    )
    parser.add_argument(
        "--save_path", type=str, default="./",
        help="knn graph save path"
    )

    args = parser.parse_args()

    dataset = DglNodePropPredDataset(name='ogbn-arxiv', root=args.data_path)
    g, _ = dataset[0]

    if args.graph_transform == 'rg':
        g = dgl.reverse(g)  # for citation network, use reverse graph
    
    elif args.graph_transform == 'bg':
        g = dgl.to_bidirected(g)

    else:  # none
        pass

    emb = np.load(args.emb_path, allow_pickle=True)
    g_str = 'g' if args.graph_transform == 'none' else args.graph_transform

    knn_g = build_knn_graph(g, emb, args.n_neigh, args.p)

    dgl.save_graphs(
        os.path.join(args.save_path, 
        f'knn_{g_str}_{args.n_neigh}_p{args.p}.bin' if args.p >= 0 else f'knn_{g_str}_{args.n_neigh}.bin'
        ), 
        knn_g
    )
