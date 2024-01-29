import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GatedGraphConv

from factory.embed_factory import NodeEmbedFactory
from pymodels.node_emb_layers.average_embed_node import AverageEmbedNode
from pymodels.node_emb_layers.distinct_embed_node import DistinctEmbedNode
from pymodels.node_emb_layers.embed_node import EmbedNode
from pymodels.node_emb_layers.lstm_embed_node import TextualLSTMEmbedNode, BothLSTMEmbedNode
from utils.pymodels_util import to_cuda, graph_readout


class GGNNModel(nn.Module):
    def __init__(self, config):
        super(GGNNModel, self).__init__()
        self.config = config
        self.name = "ggnn"
        self.cur_meanfeats = None
        self.use_cuda = self.config.use_cuda
        self.class_num = self.config.class_num
        self.graph_config = getattr(self.config, 'ggnn')
        self.use_nfeat = self.config.node_emb_layer['use_nfeature']
        self.in_dim = self.graph_config['in_dim']
        self.out_dim = self.graph_config['out_dim']
        self.nsteps = self.graph_config['nsteps']

        # Embedding Configuration
        self.node_emb_layer1 = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)
        self.node_emb_layer2 = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        # Graph Convolution
        self.g_conv1 = GatedGraphConv(self.in_dim, self.out_dim, self.nsteps, len(self.config.edge_type_list))
        self.g_conv2 = GatedGraphConv(self.in_dim, self.out_dim, self.nsteps, len(self.config.edge_type_list))

        # Activation and Batch Norms
        self.activation1 = nn.LeakyReLU()
        self.activation2 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(self.config.dropout)
        self.dropout2 = nn.Dropout(self.config.dropout)

        # Sub Networks
        fforward_dims = self.out_dim
        if self.config.ggnn['initial_representation']:
            fforward_dims = self.out_dim * 2
        self.batch_norm1 = nn.BatchNorm1d(fforward_dims)
        self.batch_norm2 = nn.BatchNorm1d(fforward_dims)
        self.fforward = nn.Linear(fforward_dims*2, self.class_num)

    def forward(self, batch_dict, running_mode, loss_fn, feat_only=False):
        g1 = batch_dict['graphs1']
        g2 = batch_dict['graphs2']
        class_target = batch_dict['tgt']
        h1 = to_cuda(g1.ndata['node_feat'], self.use_cuda)
        h2 = to_cuda(g2.ndata['node_feat'], self.use_cuda)
        node_len1 = g1.ndata['node_len'].cpu().tolist()
        node_len2 = g2.ndata['node_len'].cpu().tolist()
        elist1 = to_cuda(g1.edata['edge_type'], self.use_cuda)
        elist2 = to_cuda(g2.edata['edge_type'], self.use_cuda)
        embed_h1 = self.node_emb_layer1(h1, node_len1)
        embed_h2 = self.node_emb_layer2(h2, node_len2)

        h1 = F.leaky_relu(self.g_conv1(g1, embed_h1, elist1))
        h2 = F.leaky_relu(self.g_conv2(g2, embed_h2, elist2))
        if self.config.ggnn['initial_representation']:
            h1 = torch.cat([h1, embed_h1], -1)
            h2 = torch.cat([h2, embed_h2], -1)

        h1 = self.batch_norm1(h1)
        h2 = self.batch_norm2(h2)
        h1 = self.dropout1(h1)
        h2 = self.dropout2(h2)
        g1.ndata['h'] = h1
        g2.ndata['h'] = h2
        mean_feats1 = graph_readout(g1, self.graph_config['graph_agg'])
        mean_feats2 = graph_readout(g2, self.graph_config['graph_agg'])
        if running_mode == "test":
            self.cur_meanfeats = to_cuda(torch.cat((mean_feats1, mean_feats2), 1), self.use_cuda)
        if feat_only:
            return torch.cat((mean_feats1, mean_feats2), 1)
        dense_output = F.leaky_relu(self.fforward(torch.cat((mean_feats1, mean_feats2), 1)))
        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(torch.tensor(class_target, dtype=torch.long),
                          use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss
