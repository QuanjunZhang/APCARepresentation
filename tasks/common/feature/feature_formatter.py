from bases.base_formatter import BaseFormatter
from tokenizer.code_tokenizer import CodeTokenizer


class FeatureFormatter(BaseFormatter):
    def __init__(self, config, name="FeatureFormatter"):
        """
        TFIDFFormatter will format the input data
        """
        self.name = name
        self.disable_tqdm = config.disable_tqdm
        self.config = config
        self.t3_parser = CodeTokenizer(data=[], lang="C", tlevel='t3')
        BaseFormatter.__init__(self, config, name)

    def format(self, item_json, vocab_dicts):
        """
        Format single item_json using the Vocab Dictionary
        :param item_json: JSON of a single item in the dataset
        :param vocab_dicts: ["Token", "Node", "Target"].
        :return: Return Datapoints
        """
        datapoint = self.datapoint_class()
        feature_vocab = vocab_dicts[1]
        features_dict = dict()
        for k in item_json:
            if k == "pattern-features" and "pattern" in self.config.feature:
                features_dict.update(item_json[k])
            if k == "context_features" and "context" in self.config.feature:
                features_dict.update(item_json[k])
            if k == "code_features" and "code" in self.config.feature:
                features_dict.update(item_json[k])
        features = [0] * len(feature_vocab)
        for i in range(0, len(feature_vocab)):
            word = feature_vocab.get_i2w(i)
            if word in features_dict.keys():
                features[i] = float(features_dict[word])
        datapoint.features = features
        datapoint.tgt = item_json['target']
        return datapoint
