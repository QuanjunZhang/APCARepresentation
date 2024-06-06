from configs.model_type import ModelType
from configs.task_type import TaskType, Task
from tasks.common.graph.graph_collate import collate_graph_for_classification, collate_graph_for_classification2
from tasks.common.sequence.sequence_collate import collate_sequence_for_classification


class CollateFactory:
    @staticmethod
    def get_collate_fn(config):
        # All TF-IDF methods does not have collate function
        if config.model_type in [ModelType.NaiveBayes,
                                 ModelType.XGBoost,
                                 ModelType.SVM]:
            return None
        elif config.model_type in [ModelType.NaiveBayes_Feature,
                                   ModelType.XGBoost_Feature,
                                   ModelType.SVM_Feature]:
            return None
        # Classification Task
        elif config.task_type == TaskType.Classification and config.task in [Task.CodeClassification]:
            if config.model_type in [ModelType.LSTM,
                                     ModelType.BiLSTM,
                                     ModelType.TRANSFORMERENCODER]:
                return collate_sequence_for_classification
            elif config.model_type in [
                ModelType.GCN,
                ModelType.GAT,
                ModelType.GGNN]:
                return collate_graph_for_classification2
            elif config.model_type in [ModelType.TreeLSTM, ]:
                return collate_graph_for_classification2
            else:
                raise SystemExit(NotImplementedError(
                    "Unknown Collate Fn in CollateFactory for classification" % config.model_type))