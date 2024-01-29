from enum import Enum


class ModelType(Enum):
    # TF-IDF Based Methods
    XGBoost = "XGBoost"
    SVM = "SVM"
    NaiveBayes = "NaiveBayes"
    XGBoost_Feature = "XGBoost_Feature"
    SVM_Feature = "SVM_Feature"
    NaiveBayes_Feature = "NaiveBayes_Feature"

    # Sequence-Based Methods
    LSTM = "LSTM"
    BiLSTM = "BiLSTM"
    TRANSFORMERENCODER = "TRANSFORMERENCODER"

    # Tree-Based Method
    TreeLSTM = "TreeLSTM"

    # Graph-Based Methods
    GCN = 'GraphConvNetwork'
    GAT = 'GraphAttentionNetwork'
    GGNN = 'GatedGraphNeuralNetwork'

    # Ensemble Method
    Ensemble = "Ensemble"
