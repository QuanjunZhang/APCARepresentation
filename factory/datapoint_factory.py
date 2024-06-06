from configs.experiment_mode import ExperimentMode
from configs.model_type import ModelType
from configs.task_type import TaskType
from tasks.common.feature.feature_datapoint import FeatureDataPoint
from tasks.common.graph.graph_datapoint import GraphDataPoint
from tasks.common.sequence.sequence_datapoint import SequenceDataPoint
from tasks.common.tfidf.tfidf_datapoint import TFIDFDataPoint


class DataPointFactory:
    @staticmethod
    def get_datapoint(config):
        # Classification Task
        if config.task_type == TaskType.Classification:
            if config.model_type in [ModelType.NaiveBayes,
                                     ModelType.XGBoost,
                                     ModelType.SVM]:
                return TFIDFDataPoint
            elif config.model_type in [ModelType.NaiveBayes_Feature,
                                       ModelType.XGBoost_Feature,
                                       ModelType.SVM_Feature]:
                return FeatureDataPoint
            elif config.model_type in [ModelType.LSTM,
                                       ModelType.BiLSTM,
                                       ModelType.TRANSFORMERENCODER]:
                return SequenceDataPoint
            elif config.model_type in [ModelType.TreeLSTM,
                                       ModelType.GCN,
                                       ModelType.GAT,
                                       ModelType.GGNN]:
                return GraphDataPoint
            else:
                raise SystemExit(NotImplementedError(
                    "Unknown Model Type in Datapoint Factory for classification" % config.model_type))

