import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

from factory.embed_factory import NodeEmbedFactory
from pymodels.node_emb_layers.embed_node import EmbedNode
from pymodels.node_emb_layers.lstm_embed_node import TextualLSTMEmbedNode, BothLSTMEmbedNode
from pymodels.submodels.gnn_layers.gat_layer import GATLayer
from utils.pymodels_util import to_cuda


class GATModel(nn.Module):
    def __init__(self, config):
        super(GATModel, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.in_dim = self.config.gat['in_dim']
        self.out_dim = self.config.gat['out_dim']

        # Embedding Configurations
        self.node_emb_layer1 = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)
        self.node_emb_layer2 = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        # Conv Layers
        self.gat_layers1 = nn.ModuleList([GATLayer(config) for _ in range(self.config.gat['layers'])])
        self.gat_layers2 = nn.ModuleList([GATLayer(config) for _ in range(self.config.gat['layers'])])

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

        for gat in self.gat_layers1:
            h1 = gat(g1, h1)
            h1 = torch.mean(h1, dim=1)
        for gat in self.gat_layers2:
            h2 = gat(g2, h2)
            h2 = torch.mean(h2, dim=1)
        # Due to the head, the size of h is [node_size, nhead, dims]
        # For now, we mean the head output
        g1.ndata['h'] = h1
        g2.ndata['h'] = h2
        mean_feats1 = dgl.max_nodes(g1, 'h')
        mean_feats2 = dgl.max_nodes(g2, 'h')
        dense_output = F.leaky_relu(self.fforward(torch.cat((mean_feats1, mean_feats2), 1)))

        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(torch.LongTensor(class_target), use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss

    def get_node_emb(self):
        """
        Get the node embedding layer based on the configuration
        :return:
        """
        # Since structure only a scalar value in the data, there is no need
        # process it further
        emb_tech = None
        if self.config.use_nfeature == "structure":
            emb_tech = EmbedNode(self.config)
        elif self.config.use_nfeature == "textual":
            # If it is structure or both type, we have three option
            if self.config.node_emb_layer['mode'] == "LSTMEmbedNode":
                # LSTM-based Representation
                emb_tech = TextualLSTMEmbedNode(self.config)
        elif self.config.use_nfeature == "both":
            # If it is structure or both type, we have three option
            if self.config.node_emb_layer['mode'] == "LSTMEmbedNode":
                # LSTM-based Representation
                emb_tech = BothLSTMEmbedNode(self.config)

        if not emb_tech:
            raise NotImplementedError
        return emb_tech
