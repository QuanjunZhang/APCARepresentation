import operator
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tokenizer.vocab_dict import TokenizationConstant
from utils.pymodels_util import convert_logit_to_indices, to_cuda, compute_loss


class LSTMClassifyModel(nn.Module):
    def __init__(self, config):
        super(LSTMClassifyModel, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.vocab_len = self.config.get_vocab_size_for_embedding()

        self.word_emb_dim = self.config.word_emb_dims
        self.lstm_dims = self.config.lstm['dims']
        self.lstm_dropout = self.config.lstm['dropout']
        self.embedding = nn.Embedding(self.vocab_len, self.word_emb_dim)

        self.encoder1 = nn.LSTM(self.word_emb_dim,
                                self.lstm_dims,
                                num_layers=self.config.lstm['layer'],
                                bidirectional=self.config.lstm['bidirectional'],
                                dropout=self.lstm_dropout,
                                batch_first=True)
        self.encoder2 = nn.LSTM(self.word_emb_dim,
                                self.lstm_dims,
                                num_layers=self.config.lstm['layer'],
                                bidirectional=self.config.lstm['bidirectional'],
                                dropout=self.lstm_dropout,
                                batch_first=True)
        if self.config.lstm['bidirectional']:
            forward_dims = self.lstm_dims * 2
            self.name = "bilstm"
        else:
            forward_dims = self.lstm_dims
            self.name = "lstm"

        # Classic Classification
        if hasattr(self.config, "class_num"):
            self.fforward = nn.Linear(forward_dims * 2, self.config.class_num)
        else:
            self.fforward = nn.Linear(forward_dims * 2, self.config.target_vocab_dict.vocab_size())
        self.cur_meanfeats = None
        if self.use_cuda:
            self.fforward = self.fforward.cuda()
            self.encoder1 = self.encoder1.cuda()
            self.encoder2 = self.encoder2.cuda()
            self.embedding = self.embedding.cuda()

    def forward(self, batch_dict, running_mode, loss_fn, feat_only=False):
        # src = [bsz, seq_len]
        class_target = batch_dict['tgt']
        fn_tensors1 = to_cuda(batch_dict['fn_tensors1'], use_cuda=self.use_cuda)
        fn_tensors2 = to_cuda(batch_dict['fn_tensors2'], use_cuda=self.use_cuda)
        fn_len1 = batch_dict['funcs_lens1']
        fn_len2 = batch_dict['funcs_lens2']

        embed_fn1 = self.embedding(fn_tensors1)
        embed_fn2 = self.embedding(fn_tensors2)
        # Output = [bsz,  max token num, word emb dim]
        packed_src1 = pack_padded_sequence(embed_fn1, fn_len1,
                                           batch_first=True, enforce_sorted=False)
        packed_src2 = pack_padded_sequence(embed_fn2, fn_len2,
                                           batch_first=True, enforce_sorted=False)
        outputs1, (hidden1, cell1) = self.encoder1(packed_src1)
        outputs2, (hidden2, cell2) = self.encoder2(packed_src2)
        outputs1, input_sizes1 = pad_packed_sequence(outputs1, batch_first=True)
        outputs2, input_sizes2 = pad_packed_sequence(outputs2, batch_first=True)
        last_seq_idxs1 = torch.tensor([x - 1 for x in fn_len1], dtype=torch.long)
        last_seq_idxs2 = torch.tensor([x - 1 for x in fn_len2], dtype=torch.long)
        last_seq_items1 = outputs1[range(outputs1.shape[0]), last_seq_idxs1, :]
        last_seq_items2 = outputs2[range(outputs2.shape[0]), last_seq_idxs2, :]
        if feat_only:
            return torch.cat((last_seq_items1, last_seq_items2), dim=1)
        dense_output = F.leaky_relu(self.fforward(torch.cat((last_seq_items1, last_seq_items2), dim=1)))
        if running_mode == "test":
            self.cur_meanfeats = dense_output
        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(torch.tensor(class_target, dtype=torch.long), use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss
