import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import SAGEConv

from agdn import AGDN
from gat import GAT


class LocalGlobalGNN(nn.Module):
    """
    local & global fusion gnn
    """
    def __init__(self, in_feat, n_hidden, n_classes, l_layers, g_layers, n_mlp,
                 dropout, activation, l_model='sage', g_model='sage', 
                 aggregator_type='mean', residual=False):
        super(LocalGlobalGNN, self).__init__()

        self._residual = residual
        if self._residual:
            self.proj = nn.Linear(in_feat, n_hidden)

        # first fixed model for local & global
        if l_model == 'sage':
            self.local_gnn = GraphSAGE(
                in_feats=in_feat,
                n_hidden=n_hidden,
                n_classes=n_hidden,
                n_layers=l_layers,
                activation=activation,
                dropout=dropout,
                aggregator_type=aggregator_type
            )

        elif l_model == 'gat':
            self.local_gnn = GAT(
                in_feats=in_feat,
                n_classes=n_hidden,
                n_hidden=n_hidden // 4,
                n_layers=l_layers,
                n_heads=4,
                activation=activation,
                dropout=dropout,
                input_drop=0.,
                attn_drop=0.,
                edge_drop=0.,
                use_attn_dst=False,
                use_symmetric_norm=True,
            )

        elif l_model == 'agdn':
            self.local_gnn = AGDN(
                in_feats=in_feat,
                n_classes=n_hidden,
                n_hidden=n_hidden,
                n_layers=l_layers,
                K=3,
                n_heads=3,
                activation=activation,
                dropout=0.85,
                input_drop=0.35,
                edge_drop=0.6,
                attn_drop=0.0,
                diffusion_drop=0.3,
                use_attn_dst=True,
                position_emb=True,
                transition_matrix='gat_adj',
                weight_style='HA',
                HA_activation='leakyrelu',
                residual=True,
                bias_last=True,
                no_bias=False,
                zero_inits=False,
            )
        
        else:
            raise ValueError(f"unsupported local model type: {l_model}")

        if g_model == 'sage':
            self.global_gnn = GraphSAGE(
                in_feats=in_feat,
                n_hidden=n_hidden,
                n_classes=n_hidden,
                n_layers=g_layers,
                activation=activation,
                dropout=dropout,
                aggregator_type=aggregator_type  
            )
        
        else:
            raise ValueError(f"unsupported global model type: {g_model}")


        multiplier = 3 if self._residual else 2 
        self.MLP = MLP(
            in_feats=multiplier * n_hidden,
            hidden=n_hidden // multiplier,
            out_feats=n_classes,
            n_layers=n_mlp,
            dropout=0.2,
            normalization='batch'
        )
        # self.MLP = nn.Linear(2*n_hidden, n_classes)

    def forward(self, g, knn_g, feat):
        # local part: original gnn
        local_emb = self.local_gnn(g, feat)
        
        # global part: knn gnn
        global_emb = self.global_gnn(knn_g, feat)

        h = torch.cat([local_emb, global_emb], dim=1)

        # residual part
        if self._residual:
            res_emb = self.proj(feat)
            h = torch.cat([h, res_emb], dim=1)

        # concat all info
        h = self.MLP(h)

        # return h.softmax(dim=1)
        return h


class MLP(nn.Module):
    """
    a reusable Multi-Layer Perceptron module
    with a normalization layer, a nonlinear layer and a dropout layer 
    between two successive linear layers, just like in details
    <Simple and Efficient Heterogeneous Graph Neural Network> P6 5.2
    """
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout=0.1, 
                 input_drop=0., residual=False, normalization="batch",
                 activation='relu'):
        super(MLP, self).__init__()
        self._residual = residual
        self.activation = activation
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.n_layers = n_layers
        
        self.input_drop = nn.Dropout(input_drop)

        if n_layers == 1:
            # just one layer MLP
            self.layers.append(nn.Linear(in_feats, out_feats))
        
        else:
            # multi-layer 

            # 1st layer (input, hidden)
            self.layers.append(nn.Linear(in_feats, hidden))
            if normalization == "batch":
                self.norms.append(nn.BatchNorm1d(hidden))
            if normalization == "layer":
                self.norms.append(nn.LayerNorm(hidden))
            if normalization == "none":
                self.norms.append(nn.Identity())
            
            # 2nd and later layers (hidden, hidden)
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
                
                if normalization == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden))
                if normalization == "layer":
                    self.norms.append(nn.LayerNorm(hidden))
                if normalization == "none":
                    self.norms.append(nn.Identity())
            
            # last layer (hidden, output), without batch norm
            self.layers.append(nn.Linear(hidden, out_feats))
        
        if self.n_layers > 1:
            # shares for each layer: activation & dropout
            self.dropout = nn.Dropout(dropout)
            
            if activation == 'relu':
                self.act = nn.ReLU()
            elif activation == 'tanh':
                self.act = nn.Tanh()
            elif activation == 'sigmoid':
                self.act= nn.Sigmoid()
            elif activation == 'identity':
                self.act = nn.Identity()
            else:
                self.act = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.activation)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

        for norm in self.norms:
            norm.reset_parameters()
        # print(self.layers[0].weight)

    def forward(self, x):
        x = self.input_drop(x)
        if self._residual:
            prev_x = x
        
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)  # linear part
            
            if layer_id < self.n_layers - 1:
                # without last layer, need this 3 operations
                x = self.dropout(self.act(self.norms[layer_id](x)))
            
            if self._residual:
                if x.shape[1] == prev_x.shape[1]:
                    x += prev_x
                prev_x = x

        return x


class GraphSAGE(nn.Module):
    """
    GraphSAGE Model for Graph Node Property Prediction
    """
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, 
                 activation,  dropout=0.2, aggregator_type='mean'):
        super(GraphSAGE, self).__init__()

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # support one layer GraphSAGE 20220919
        if n_layers == 1:  # just one layer GraphSAGE
            self.layers.append(SAGEConv(in_feats, n_classes, aggregator_type))
        
        else:  # multi-layer
            # input layer (in, hid)
            self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))

            # hidden layers (hid, hid)
            for _ in range(n_layers - 2):
                self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
            
            # output layer (hid, out)
            self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))

    def forward(self, graph, feat):
        # full batch training for GraphSAGE
        h = self.dropout(feat)
        
        for l, layer in enumerate(self.layers):
            # last/output layer, without activation & dropout
            h = layer(graph, h)
            
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)

            # h = self.activation(h)
            # h = self.dropout(h)

        return h


class CAREConv(nn.Module):
    """
    CAREConv for homogeneous graph, e.g. ogbn-arxiv
    """
    def __init__(self, in_dim, out_dim, num_classes, activation=None, step_size=0.02):
        super(CAREConv, self).__init__()

        self.activation = activation
        self.step_size = step_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes

        self.dist = 0
        
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.MLP = nn.Linear(self.in_dim, self.num_classes)

        self.p = 0.5
        self.last_avg_dist = 0
        self.f = []
        self.cvg = False
    
    # def _calc_distance(self, edges):
    #     """udf msg function"""
    #     # formula 2
    #     d = torch.norm(torch.tanh(self.MLP(edges.src['h'])) - torch.tanh(self.MLP(edges.dst['h'])), 1, 1)
    #     return {'d': d}
    
    def _top_p_sampling(self, g, p):
        """choose most similar neighbors as ratio p """
        dist = g.edata['d']
        neigh_list = []
        for nid in g.in_degrees().nonzero().squeeze():
            eid = g.in_edges(nid, form='eid')  # just edge ids, torch.LongTensor, maybe 

            num_neigh = torch.ceil(p * g.in_degrees(nid)).int().item()
            
            # neighbors dist for each node(all in edges)
            neigh_dist = dist[eid]

            if num_neigh < neigh_dist.shape[0]:  # np.argpartition use requires 
                neigh_index = np.argpartition(neigh_dist.cpu().detach(), num_neigh)[:num_neigh]
            else:
                neigh_index = np.arange(num_neigh)
            
            neigh_list.append(eid[neigh_index])

        return torch.cat(neigh_list)

    def forward(self, g, feat):
        """full batch training, first run out"""
        with g.local_scope():
            # first compute edge dist (high efficient instance than original)
            g.ndata['t'] = torch.tanh(self.MLP(feat))
            g.apply_edges(fn.u_sub_v('t', 't', 'd'))
            g.edata['d'] = torch.norm(g.edata['d'], 1, 1)
            self.dist = g.edata['d']
            g.ndata.pop('t')

            sampled_edges = self._top_p_sampling(g, self.p)  # choose part of edges(by eid)

            # formula 8
            # dgl.DGLGraph.send_and_recv
            # Send messages along the specified edges and reduce them on the destination nodes to update their features.
            
            # MPNN: for homogeneous graph, judst intra-aggregation
            g.ndata['h'] = feat
            g.send_and_recv(sampled_edges, fn.copy_u('h', 'm'), fn.mean('m', 'h'))
            hr = g.ndata.pop('h')
            if self.activation is not None:
                hr = self.activation(hr)
            
            h_homo = feat + hr
            if self.activation is not None:
                h_homo = self.activation(h_homo)
            
            return self.linear(h_homo)


class CAREGNN(nn.Module):
    def __init__(self, in_dim, num_classes, hid_dim=64, n_layers=2, 
                 activation=None, step_size=0.02):
        super(CAREGNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.activation = activation
        self.step_size = step_size
        self.n_layers = n_layers

        self.layers = nn.ModuleList()

        if self.n_layers == 1:   # Single Layer
            self.layers.append(
                CAREConv(
                    in_dim=self.in_dim,
                    out_dim=self.num_classes,
                    num_classes=self.num_classes,
                    activation=self.activation,
                    step_size=self.step_size
                )
            )
        
        else:  # Multiple Layer
            # Input Layer
            self.layers.append(
                CAREConv(
                    in_dim=self.in_dim,
                    out_dim=self.hid_dim,
                    num_classes=self.num_classes,
                    activation=self.activation,
                    step_size=self.step_size
                )
            )

            # Hidden Layer
            for i in range(self.n_layers - 2):
                self.layers.append(
                    CAREConv(
                        in_dim=self.hid_dim,
                        out_dim=self.hid_dim,
                        num_classes=self.num_classes,
                        activation=self.activation,
                        step_size=self.step_size
                    )
                )
            
            # Output Layer
            self.layers.append(
                CAREConv(
                    in_dim=self.hid_dim,
                    out_dim=self.num_classes,
                    num_classes=self.num_classes,
                    activation=self.activation,
                    step_size=self.step_size
                )
            )            
    
    def forward(self, graph, feat):
        # full batch training
        # formula 4
        sim = torch.tanh(self.layers[0].MLP(feat))

        # Forward of n layers of CARE-GNN
        for layer in self.layers:
            feat = layer(graph, feat)
        
        return feat, sim

    def RLModule(self, graph, epoch, nid):
        for layer in self.layers:
            if not layer.cvg:
                # formula 5
                eid = graph.in_edges(nid, form='eid')
                avg_dist = torch.mean(layer.dist[eid])

                # formula 6
                if layer.last_avg_dist < avg_dist:
                    if layer.p - self.step_size > 0:
                        layer.p -= self.step_size
                    layer.f.append(-1)
                else:
                    if layer.p + self.step_size <= 1:
                        layer.p += self.step_size
                    layer.f.append(+1)
                layer.last_avg_dist = avg_dist

                # formula 7
                if epoch >= 9 and abs(sum(layer.f[-10:])) <= 2:
                    layer.cvg = True
