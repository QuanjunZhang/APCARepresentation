from configs.experiment_mode import ExperimentMode
from configs.model_type import ModelType
from configs.task_type import TaskType, Task
from tasks.common.feature.feature_formatter import FeatureFormatter
from tasks.common.graph.megraph_formatter import MultiEdgeGraphFormatter
from tasks.common.graph.segraph_formatter import SingleEdgeGraphFormatter
from tasks.common.graph.tbcnn_formatter import TBCNNFormatter
from tasks.common.graph.treelstm_formatter import TreeLSTMFormatter
from tasks.common.sequence.sequence_formatter import SequenceFormatter
from tasks.common.tfidf.tfidf_formatter import TFIDFFormatter


class FormatterFactory:
    @staticmethod
    def get_formatter(config):
        # Classification Task for Code Classification and Vulnerability Detection
        # Both Tasks uses code so they can use the same formatter
        if config.task_type == TaskType.Classification and config.task in [Task.CodeClassification]:
            if config.model_type in [ModelType.NaiveBayes,
                                     ModelType.XGBoost,
                                     ModelType.SVM]:
                return TFIDFFormatter(config)
            elif config.model_type in [ModelType.NaiveBayes_Feature,
                                       ModelType.XGBoost_Feature,
                                       ModelType.SVM_Feature]:
                return FeatureFormatter(config)
            elif config.model_type in [ModelType.LSTM,
                                       ModelType.BiLSTM,
                                       ModelType.TRANSFORMERENCODER]:
                return SequenceFormatter(config)
            elif config.model_type == ModelType.TreeLSTM:
                return TreeLSTMFormatter(config)
            elif config.model_type in [ModelType.GCN, ModelType.GAT]:
                return SingleEdgeGraphFormatter(config)
            elif config.model_type == ModelType.GGNN:
                return MultiEdgeGraphFormatter(config)
        return None
