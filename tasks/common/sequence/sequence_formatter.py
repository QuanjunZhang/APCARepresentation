from bases.base_formatter import BaseFormatter
from tokenizer.code_tokenizer import CodeTokenizer


class SequenceFormatter(BaseFormatter):
    def __init__(self, config, name="SequenceFormatter"):
        """
        SequenceFormatter will format the input data
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
        :param vocab_dicts:  ["Token", "Node", "Target"].
        :return: Return Datapoints
        """
        token_vd, node_vd, target_vd, word_vd = vocab_dicts
        datapoint = self.datapoint_class()
        # datapoint.file_index = item_json['function_id']
        fn1 = item_json['function1']  # .splitlines()[0]
        fn2 = item_json['function2']
        datapoint.function1 = fn1
        datapoint.function2 = fn2

        tok_fn_str1 = self.t3_parser.tokenize(fn1)
        tok_fn_str2 = self.t3_parser.tokenize(fn2)
        tok_fn_str = self.t3_parser.tokenize(fn1 + ' ' + fn2)

        tok_fn_tokens1 = tok_fn_str1.split()
        tok_fn_tokens2 = tok_fn_str2.split()
        tok_fn_tokens = tok_fn_str.split()
        length1 = len(tok_fn_tokens1)
        length2 = len(tok_fn_tokens2)
        length = len(tok_fn_tokens)
        start1 = 0 if length1 < self.config.max_function_length else length1 - self.config.max_function_length
        start2 = 0 if length2 < self.config.max_function_length else length2 - self.config.max_function_length
        start = 0 if length < self.config.max_function_length else length - self.config.max_function_length
        tok_function1, blen1 = self.tokenize_sentence(" ".join(tok_fn_tokens1[start1:]), token_vd)
        tok_function2, blen2 = self.tokenize_sentence(" ".join(tok_fn_tokens2[start2:]), token_vd)
        tok_function, blen = self.tokenize_sentence(" ".join(tok_fn_tokens[start:]), token_vd)

        datapoint.function_vec = tok_function
        datapoint.function_vec1 = tok_function1
        datapoint.function_vec2 = tok_function2
        datapoint.tgt = item_json['target']
        return datapoint


class SequenceTranslationSingleVocabFormatter(BaseFormatter):
    def __init__(self, config, name="SequenceTranslationSingleVocabFormatter"):
        """
        SequenceFormatter will format the input data
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
        token_vd, node_vd, target_vd = vocab_dicts
        datapoint = self.datapoint_class()
        # print(item_json['function'])
        # input()
        fn = " ".join(item_json['function'].splitlines()[1:])

        tok_fn_str = self.t3_parser.tokenize(fn)
        tok_fn_tokens = tok_fn_str.split()
        tok_function, blen = self.tokenize_sentence(" ".join(tok_fn_tokens[:self.config.max_function_length]), token_vd)
        datapoint.function_vec = tok_function

        tok_tgt = self.t3_parser.tokenize(item_json['target'])
        datapoint.tgt = tok_tgt
        tok_tgt, blen = self.tokenize_sentence(tok_tgt, token_vd)
        datapoint.tgt_vec = tok_tgt
        return datapoint


class SequenceTranslationDoubleVocabFormatter(BaseFormatter):
    def __init__(self, config, name="SequenceTranslationDoubleVocabFormatter"):
        """
        SequenceFormatter will format the input data
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
        token_vd, node_vd, target_vd = vocab_dicts
        datapoint = self.datapoint_class()
        # print(item_json['function'])
        # input()
        fn = " ".join(item_json['function'].splitlines()[1:])

        tok_fn_str = self.t3_parser.tokenize(fn)
        tok_fn_tokens = tok_fn_str.split()
        tok_function, blen = self.tokenize_sentence(" ".join(tok_fn_tokens[:self.config.max_function_length]), token_vd)
        datapoint.function_vec = tok_function

        tok_tgt = self.t3_parser.tokenize(item_json['target'])
        datapoint.tgt = tok_tgt
        assert target_vd is not None
        tok_tgt, blen = self.tokenize_sentence(tok_tgt, target_vd)
        datapoint.tgt_vec = tok_tgt
        return datapoint


class SequenceNamePredictionFormatter(BaseFormatter):
    def __init__(self, config, name="SequenceNamePredictionFormatter"):
        """
        SequenceFormatter will format the input data
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
        token_vd, node_vd, target_vd = vocab_dicts
        datapoint = self.datapoint_class()
        # print(item_json['function'])
        # input()
        fn = " ".join(item_json['function'].splitlines()[1:])

        tok_fn_str = self.t3_parser.tokenize(fn)
        tok_fn_tokens = tok_fn_str.split()
        tok_function, blen = self.tokenize_sentence(" ".join(tok_fn_tokens[:self.config.max_function_length]), token_vd)
        datapoint.function_vec = tok_function

        tok_tgt = self.t3_parser.tokenize(item_json['target'])
        clean_tgt = "|".join(tok_tgt.split())

        datapoint.tgt = clean_tgt
        assert target_vd is not None
        print(clean_tgt)
        tok_tgt, blen = self.tokenize_sentence(clean_tgt, target_vd, eos=False)
        print(tok_tgt)
        input()
        datapoint.tgt_vec = tok_tgt
        return datapoint
