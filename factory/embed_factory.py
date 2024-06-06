from pymodels.node_emb_layers.distinct_embed_node import DistinctEmbedNode
from pymodels.node_emb_layers.embed_node import EmbedNode
from pymodels.node_emb_layers.lstm_embed_node import TextualLSTMEmbedNode, BothLSTMEmbedNode


class NodeEmbedFactory:
    def get_node_embed_technique(self, config):
        use_nfeats = config.node_emb_layer['use_nfeature']
        if use_nfeats == "structure":
            return self.get_structural_tech(config)
        elif use_nfeats == "textual":
            return self.get_textual_tech(config)
        elif use_nfeats == "both":
            # If it is structure or both type, we have three option
            if config.node_emb_layer['mode'] == "LSTMEmbedNode":
                return BothLSTMEmbedNode
        elif use_nfeats in ["distinct-feats"]:
            return DistinctEmbedNode
        else:
            raise SystemExit(NotImplementedError("Technique %s is not implemented in "
                                                 "Collate" % use_nfeats))

    @staticmethod
    def get_structural_tech(config):
        """
        Get the Structual Embedding Technique, based on the config.
        :param config: Configuration Object
        :return:
        """
        return EmbedNode

    @staticmethod
    def get_textual_tech(config):
        """
        Get Textual Embed Technique
        :param config: Configuration Object
        :return:
        """
        # If it is structure or both type, we have three option
        if config.node_emb_layer['mode'] == "LSTMEmbedNode":
            # LSTM-based Representation
            return TextualLSTMEmbedNode

    @staticmethod
    def get_both_tech(config):
        """
        Get Two form of Embed Technique
        :param config: Configuration Object
        :return:
        """
        if config.node_emb_layer['mode'] == "LSTMEmbedNode":
            return BothLSTMEmbedNode