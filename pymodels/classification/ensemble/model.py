import functools

import torch
from torch import nn

from utils.pymodels_util import to_cuda
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    def __init__(self, models: list, configs: list, config):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList([
            to_cuda(model(config), config.use_cuda) for model, config in zip(models, configs)
        ])
        self.config = config
        self.fforward = to_cuda(nn.Linear(config['forward_dim'], config['class_num']), config['use_cuda'])
        self.loss_fn = nn.CrossEntropyLoss()
        if self.config['mid_fusion'] and self.config["fusion_strategy"] == "max_pooling":
            self.max_pooling = nn.MaxPool2d(kernel_size=1)
        if self.config['mid_fusion'] and self.config["fusion_strategy"] == "attention":
            self.attention = nn.MultiheadAttention(num_heads=8, embed_dim=config['forward_dim'], batch_first=True,
                                                   dropout=0.2)

    def forward(self, batch_dicts: list, running_mode, loss_fns: list):

        if self.config['back_fusion']:
            masks = []
            losses = []
            class_target = []
            for batch_dict, loss_fn, sub_model in zip(batch_dicts, loss_fns, self.models):
                sm_mask_output, tgt_tensor, loss = sub_model(batch_dict, running_mode, loss_fn)
                class_target.append(tgt_tensor)
                masks.append(sm_mask_output)
                losses.append(loss)
            weight = self.config['loss_weight']
            weight_losses = [losses[i] * weight[i] for i in range(len(weight))]
            weight_masks = [masks[i] * weight[i] for i in range(len(weight))]
            return sum(weight_masks), class_target[0], sum(weight_losses)
        elif self.config['mid_fusion']:
            feats = []
            class_target = None
            for batch_dict, loss_fn, sub_model in zip(batch_dicts, loss_fns, self.models):
                feat = sub_model(batch_dict, running_mode, loss_fn, feat_only=True)
                feats.append(F.normalize(feat,2,dim=-1))
                if class_target is None:
                    class_target = batch_dict['tgt']

            fusion_feat = self.fusion(feats)
            dense_output = F.leaky_relu(self.fforward(fusion_feat))
            tgt_tensor = to_cuda(torch.tensor(class_target, dtype=torch.long), use_cuda=self.config['use_cuda'])
            loss = self.loss_fn(dense_output, tgt_tensor)
            sm_mask_output = F.softmax(dense_output, dim=-1)
            return sm_mask_output, class_target, loss

    def fusion(self, feats):
        if self.config["fusion_strategy"] == "concatenation":
            # batch-first
            return torch.cat(feats, dim=-1)
        elif self.config["fusion_strategy"] == "weighted_sum":
            weight = self.config['weight']
            return sum([feats[i] * weight[i] for i in range(len(weight))])
        elif self.config["fusion_strategy"] == "attention":
            feats = torch.cat(feats, dim=-1)
            return self.attention(feats, feats, feats)[0]
        elif self.config["fusion_strategy"] == "max_pooling":
            return functools.reduce(torch.max,feats)
        else:
            raise NotImplementedError
