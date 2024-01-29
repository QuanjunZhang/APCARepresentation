"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

from factory.embed_factory import NodeEmbedFactory
from pymodels.node_emb_layers.average_embed_node import AverageEmbedNode
from pymodels.node_emb_layers.distinct_embed_node import DistinctEmbedNode
from pymodels.node_emb_layers.embed_node import EmbedNode
from pymodels.node_emb_layers.lstm_embed_node import TextualLSTMEmbedNode
from pymodels.submodels.tree_cell import ChildSumTreeLSTMCell

from utils.pymodels_util import to_cuda, graph_readout
from utils.util import simple_plot_dgl_graph


class TreeLSTMModel(nn.Module):
    def __init__(self, config):
        super(TreeLSTMModel, self).__init__()
        self.config = config
        self.name = "TreeLSTM"
        self.use_cuda = self.config.use_cuda
        # Embedding Configurations
        self.node_emb_layer1 = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)
        self.node_emb_layer2 = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        self.dropout = nn.Dropout(self.config.dropout)
        # cell = TreeLSTMCell if self.config.treelstm['cell_type'] == 'nary' else ChildSumTreeLSTMCell
        cell = ChildSumTreeLSTMCell
        self.cell1 = cell(self.config.word_emb_dims, self.config.treelstm['in_dim'])
        self.cell2 = cell(self.config.word_emb_dims, self.config.treelstm['in_dim'])
        # self.fforward = nn.Linear(self.config.treelstm['in_dim'], self.config.class_num)
        self.fforward = nn.Linear(self.config.treelstm['in_dim'] * 2, self.config.class_num)

        self.cur_meanfeats = None

    def forward(self, batch_dict, running_mode, loss_fn,feat_only=False):
        class_target = batch_dict['tgt']
        g1 = batch_dict['graphs1']
        g2 = batch_dict['graphs2']

        n1 = g1.num_nodes()
        n2 = g2.num_nodes()
        mask1 = to_cuda(g1.ndata['mask'], use_cuda=self.config.use_cuda)
        mask2 = to_cuda(g2.ndata['mask'], use_cuda=self.config.use_cuda)

        h1 = to_cuda(th.zeros((n1, self.config.treelstm['in_dim'])), use_cuda=self.config.use_cuda)
        h2 = to_cuda(th.zeros((n2, self.config.treelstm['in_dim'])), use_cuda=self.config.use_cuda)
        c1 = to_cuda(th.zeros((n1, self.config.treelstm['in_dim'])), use_cuda=self.config.use_cuda)
        c2 = to_cuda(th.zeros((n2, self.config.treelstm['in_dim'])), use_cuda=self.config.use_cuda)

        embeds1 = self.node_emb_layer1(g1.ndata['node_feat'],
                                       g1.ndata['node_len'].cpu().tolist())
        embeds2 = self.node_emb_layer2(g2.ndata['node_feat'],
                                       g2.ndata['node_len'].cpu().tolist())

        g1.ndata['iou'] = self.cell1.W_iou(self.dropout(embeds1)) * mask1.float().unsqueeze(-1)
        g1.ndata['h'] = h1
        g1.ndata['c'] = c1
        g2.ndata['iou'] = self.cell2.W_iou(self.dropout(embeds2)) * mask2.float().unsqueeze(-1)
        g2.ndata['h'] = h2
        g2.ndata['c'] = c2

        # propagate
        dgl.prop_nodes_topo(g1, self.cell1.message_func, self.cell1.reduce_func,
                            apply_node_func=self.cell1.apply_node_func)
        dgl.prop_nodes_topo(g2, self.cell2.message_func, self.cell2.reduce_func,
                            apply_node_func=self.cell2.apply_node_func)
        # compute logits
        h1 = self.dropout(g1.ndata['h'])
        g1.ndata['h'] = h1
        h2 = self.dropout(g2.ndata['h'])
        g2.ndata['h'] = h2
        mean_feats1 = F.relu(graph_readout(g1, self.config.treelstm['graph_agg']))
        mean_feats2 = F.relu(graph_readout(g2, self.config.treelstm['graph_agg']))
        if feat_only:
            return th.cat((mean_feats1, mean_feats2), 1)
        dense_output = F.leaky_relu(self.fforward(th.cat((mean_feats1, mean_feats2), 1)))
        # dense_output = F.leaky_relu(self.fforward(mean_feats2))

        if running_mode == "test":
            self.cur_meanfeats = th.cat((mean_feats1, mean_feats2), 1).cpu()

        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(th.tensor(class_target, dtype=th.long),
                          use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss
