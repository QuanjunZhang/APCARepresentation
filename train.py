#!/usr/bin/env python

"""
Usage:
    train.py
    train.py (--config_path FILE) [--gpu_id GPU_ID]
    train.py (--config_pathFILE)
Options:
    -h --help               Show this screen.
    --gpu_id GPUID          GPU ID [default: 0]
    --config_path=FILE      Configuration Path of YML, most likely in ./yml [default: "."]
    --quiet                 Less output (not one per line per minibatch). [default: False]
"""
import pickle
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader

from bases.base_dataset import DatapointDataset
from factory.collate_factory import CollateFactory
from factory.model_factory import ModelFactory
from pymodels.classification.ensemble.model import EnsembleModel
from docopt import docopt
from configs.config import Config
from factory.dataset_factory import DatasetFactory
from factory.trainer_factory import TrainerFactory
from utils.pymodels_util import to_cuda


def evaluate_metrics(preds, labels):
    """
            Get binary score e.g., accuracy, f1, precision and recall
            :return:
            """
    metrics = dict()
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    prec = precision_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, preds)

    metrics['accuracy'] = acc
    metrics['recall'] = recall
    metrics['precision'] = prec
    metrics['f1'] = f1
    metrics['auc'] = auc
    TP, FP, TN, FN, i = 0, 0, 0, 0, 0
    for i in range(len(preds)):
        if preds[i] == 1 and labels[i] == 1:
            TP += 1
        elif preds[i] == 1 and labels[i] == 0:
            FP += 1
        elif preds[i] == 0 and labels[i] == 0:
            TN += 1
        elif preds[i] == 0 and labels[i] == 1:
            FN += 1
    metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN'] = TP, FP, TN, FN

    return metrics

def pretty_format(metrics: dict,
                  metric_headers=('accuracy', 'recall', 'precision', 'f1', 'auc', 'TP', 'FP', 'TN', 'FN')):
    return '|'.join(["{}: {}".format(x, metrics[x] if type(metrics[x])==int else round(metrics[x],4)) for x in metric_headers])



class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.ds = datasets

    def __getitem__(self, index):
        return [d[index] for d in self.ds]

    def __len__(self):
        return len(self.ds[0])


collate_fns = []


def multi_collate(samples_list):
    result = []
    samples_list = [[x[i] for x in samples_list] for i in range(len(samples_list[0]))]
    for collate_fn, samples in zip(collate_fns, samples_list):
        result.append(collate_fn(samples))
    return result


def main_ensemble():
    global collate_fns
    """
        Entry Method for Ensemble model only
        :param arguments: Arguments from docopt
        :return: NIL
     """

    ensemble_config = {
        'sub_model_configs': [
            # '/Users/tom/Downloads/learning-program-representation-master/data/custom/sequence/yml/bilstm.yml',
            # '/Users/tom/Downloads/learning-program-representation-master/data/custom/tree/yml/treelstm.yml',
            '/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/yml/ggnn.yml',
            '/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/yml/ggnn2.yml'
        ],
        'mid_fusion': True,
        'back_fusion': False,
        'fusion_strategy': 'attention',
        'class_num': 2,
        'max_epoch': 50,
        'lr': 0.001,
        "use_cuda": True,
        'forward_dim': 1024,
    }
    ensemble_config['weight'] = [1 / len(ensemble_config['sub_model_configs'])] * len(
        ensemble_config['sub_model_configs'])
    ensemble_config['loss_weight'] = [1 / len(ensemble_config['sub_model_configs'])] * len(
        ensemble_config['sub_model_configs'])
    sub_configs = [Config(c) for c in ensemble_config['sub_model_configs']]
    for sub_config in sub_configs:
        sub_config.setup_vocab_dict()

    datapoints = []

    for sub_config in sub_configs:
        datapoints.append(DatasetFactory().get_dataset(sub_config))

    collate_fns = [CollateFactory().get_collate_fn(config) for config in sub_configs]

    train_datasets = [DatapointDataset(dp.train_datapoints) for dp in datapoints]
    val_datasets = [DatapointDataset(dp.val_datapoints) for dp in datapoints]
    train_multi_datasets = MultiDataset(train_datasets)
    val_multi_datasets = MultiDataset(val_datasets)
    train_dl = DataLoader(train_multi_datasets, batch_size=128,
                          shuffle=True, collate_fn=multi_collate, num_workers=0)
    val_dl = DataLoader(val_multi_datasets, batch_size=128,
                        shuffle=False, collate_fn=multi_collate, num_workers=0)

    sub_models = []
    sub_loss_funcs = []
    for sub_config in sub_configs:
        model_class = ModelFactory().get_model(sub_config)
        sub_models.append(model_class)
        assert model_class is not None, "Model Factory fails to get Model Class"

        if sub_config.class_weights:
            assert sub_config.class_weights is not None
            sub_class_weight = to_cuda(torch.tensor(sub_config.class_weights, dtype=torch.float), sub_config.use_cuda)
            sub_loss_funcs.append(nn.CrossEntropyLoss(weight=sub_class_weight))
        else:
            sub_loss_funcs.append(nn.CrossEntropyLoss())

    ensemble_model = to_cuda(EnsembleModel(sub_models, sub_configs, ensemble_config), ensemble_config['use_cuda'])
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=ensemble_config['lr'])

    val_best_acc = 0
    for epoch_num in range(ensemble_config['max_epoch']):

        train_preds = []
        val_preds = []
        train_labels = []
        val_labels = []
        train_loss = []
        val_loss = []

        print("----------------------- epoch{} -----------------------".format(epoch_num + 1))

        for idx, batch_dicts in enumerate(train_dl):
            ensemble_model.train()
            optimizer.zero_grad()
            probs, labels, loss = ensemble_model(batch_dicts, "train", sub_loss_funcs)

            train_best_preds = np.asarray([np.argmax(line) for line in probs.cpu().tolist()])
            train_preds.extend(train_best_preds)
            train_labels.extend(labels)
            if type(loss) == int:
                train_loss.append(loss)
            else:
                train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        print("train loss:{}".format(sum(train_loss) / len(train_loss)))
        print(pretty_format(evaluate_metrics(train_preds, train_labels)))
        for idx, batch_dicts in enumerate(val_dl):
            ensemble_model.eval()
            with torch.no_grad():
                probs, labels, loss = ensemble_model(batch_dicts, "val", sub_loss_funcs)
            if type(loss) == int:
                val_loss.append(loss)
            else:
                val_loss.append(loss.item())
            val_best_preds = np.asarray([np.argmax(line) for line in probs.cpu().tolist()])
            val_preds.extend(val_best_preds)
            val_labels.extend(labels)
        print("val loss:{}".format(sum(val_loss) / len(val_loss)))
        m = evaluate_metrics(val_preds, val_labels)
        print(pretty_format(m))
        if m['accuracy'] >= val_best_acc:
            print("val accuracy from {} to {}".format(val_best_acc, m['accuracy']))
            val_best_acc = m['accuracy']


def main(arguments):
    """
    Entry Method for Code Intelligent Tasks
    :param arguments: Arguments from docopt
    :return: NIL
    """
    # Setup Configuration Object
    config_path = arguments.get('--config_path')
    config = Config(config_path)

    config.print_params()
    config.setup_vocab_dict()

    # Formatting the dataset and start the trainer
    dataset = DatasetFactory().get_dataset(config)
    trainer = TrainerFactory().get_trainer(config)
    config.logger.info("Trainer: %s | Dataset: %s" % (trainer.name, dataset.name))

    # Start the Training
    trainer.setup_model()
    trainer.start_train(dataset)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtype')
    args = parser.parse_args()
    rtype_value = args.rtype

    if rtype_value=='fusion':
        main_ensemble()
    elif rtype_value=='single'
        args = docopt(__doc__)
        main(args)
    
    
