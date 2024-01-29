import time
import datetime

import torch

from configs.experiment_mode import ExperimentMode
from configs.task_type import TaskType
from factory.model_factory import ModelFactory
from utils.util import get_pretty_metric, print_msg, is_tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluation.evaluator.classification_evaluator import ClassificationEvaluator


class FeatureTrainer:
    def __init__(self, config):
        self.name = "FeatureTrainer"
        self.config = config
        self.timestamp = self.config.timestamp
        self.evaluator = ClassificationEvaluator(config)
        self.model_class = ModelFactory().get_model(self.config)
        assert self.model_class is not None, "Model Factory fails to get Model Class"
        self.model = self.model_class(self.config)

    def setup_model(self):
        pass

    def start_train(self, dataset):
        """
        Training Iteration/Epoch
        :param dataset: Dataset Object for Training and Validating
        :return: NIL
        """
        start = time.time()
        train_datapoints = dataset.train_datapoints
        val_datapoints = dataset.val_datapoints
        self.train(train_datapoints)
        self.config.logger.info("=" * 100)
        self.validate(val_datapoints, "val")
        end = time.time()
        total_time = end - start
        self.config.logger.info("Total Time: %s" % str(datetime.timedelta(seconds=total_time)).split(".")[0])

    def train(self, train_datapoints):
        """
        Train self.model using train_datapoints
        :param train_datapoints: Train DataPoints
        :return: Return None as training has no accuracy
        """
        train_x = [dp.features for dp in train_datapoints]
        train_y = [dp.tgt for dp in train_datapoints]
        self.model.train(train_x, train_y)
        rm_str = "[{:^5}] ".format("train")
        self.config.logger.info(rm_str)
        return None

    def validate(self, val_datapoints, running_mode=""):
        """
        Validate self.model using val_datapoints
        :param val_datapoints: Validation Datapoints
        :param running_mode: Running Mode
        :return: Return the scores
        """
        val_x = [dp.features for dp in val_datapoints]
        val_y = [dp.tgt for dp in val_datapoints]
        preds = self.model.val(val_x, val_y)
        self.evaluator.add_metric_data(preds, val_y)
        metric_str = self.evaluator.get_pretty_metric(self.evaluator.evaluate_score())
        rm_str = "[{:^5}] ".format(running_mode)
        self.config.logger.info(rm_str + metric_str)
