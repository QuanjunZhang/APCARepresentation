import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pymodels_util import to_cuda
from factory.embed_factory import NodeEmbedFactory
from pymodels.submodels.gnn_layers.gcn_layer import GCNLayer


class GCNModel(nn.Module):
    def __init__(self, config):
        super(GCNModel, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.in_dim = self.config.gcn['in_dim']
        self.out_dim = self.config.gcn['out_dim']

        # Embedding Configurations
        self.node_emb_layer1 = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)
        self.node_emb_layer2 = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        # GCN Layers
        self.gcn_layers1 = nn.ModuleList([GCNLayer(config) for _ in range(self.config.gcn['layers'])])
        self.gcn_layers2 = nn.ModuleList([GCNLayer(config) for _ in range(self.config.gcn['layers'])])

        # Sub Networks
        self.fforward = nn.Linear(self.out_dim * 2, self.config.class_num)

    def forward(self, batch_dict, running_mode, loss_fn):
        g1 = batch_dict['graphs1']
        g2 = batch_dict['graphs2']
        class_target = batch_dict['tgt']
        h1 = to_cuda(g1.ndata['node_feat'], self.use_cuda)
        h2 = to_cuda(g2.ndata['node_feat'], self.use_cuda)
        node_len1 = g1.ndata['node_len'].cpu().tolist()
        node_len2 = g2.ndata['node_len'].cpu().tolist()
        h1 = self.node_emb_layer1(h1, node_len1)
        h2 = self.node_emb_layer2(h2, node_len2)
        for gcn in self.gcn_layers1:
            h1 = gcn(g1, h1)
        for gcn in self.gcn_layers2:
            h2 = gcn(g2, h2)

        # Remove final feat = h and batch norm
        g1.ndata['h'] = h1
        g2.ndata['h'] = h2
        mean_feats1 = dgl.max_nodes(g1, 'h')
        mean_feats2 = dgl.max_nodes(g2, 'h')
        dense_output = F.leaky_relu(self.fforward(torch.cat((mean_feats1, mean_feats2), 1)))

        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(torch.tensor(class_target, dtype=torch.long), use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss
