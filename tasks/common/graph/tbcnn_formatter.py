from bases.base_formatter import BaseFormatter
from bases.base_graph_formatter import BaseGraphFormatter
from tokenizer.code_tokenizer import CodeTokenizer


class TBCNNFormatter(BaseGraphFormatter):
    def __init__(self, config, name="TBCNNFormatter"):
        """
        TBCNNFormatter will format the input data.
        """
        self.name = name
        self.disable_tqdm = config.disable_tqdm
        self.config = config
        self.t3_parser = CodeTokenizer(data=[], lang="C", tlevel='t3')
        BaseFormatter.__init__(self, config, name)

    def format(self, item_json, vocab_dicts):
        """
        Format single item_json using the Vocab Dictionary
        {'IS_AST_PARENT': 0, 'isCFGParent': 1, "POST_DOM": 2,
        "FLOWS_TO": 3, "USE": 4, "DEF": 5, 'REACHES': 6, "CONTROLS": 7}
        :param item_json: JSON of a single item in the dataset
        :param vocab_dicts: ["Token", "Node", "Target"].
        :return: Return Datapoints
        """
        token_vd, node_vd, target_vd, word_vd = vocab_dicts
        datapoint = self.datapoint_class()
        datapoint.function1 = item_json['function1']
        datapoint.function2 = item_json['function2']
        dgl_graph1 = self._convert_to_multi_edge_dglgraph(item_json['jsgraph1'], token_vd, node_vd)
        dgl_graph2 = self._convert_to_multi_edge_dglgraph(item_json['jsgraph2'], token_vd, node_vd)
        # if dgl_graph1.num_nodes()==0 or dgl_graph2.num_nodes()==0:
        #     print(0)
        datapoint.function_graph1 = dgl_graph1
        datapoint.function_graph2 = dgl_graph2
        datapoint.graph_size1 = item_json['graph_size1']
        datapoint.graph_size2 = item_json['graph_size2']
        if type(item_json['target']) == int:
            datapoint.tgt = item_json['target']
        else:
            tok_tgt = self.t3_parser.tokenize(item_json['target'])
            datapoint.tgt = tok_tgt
            tok_tgt, blen = self.tokenize_sentence(tok_tgt, token_vd)
            datapoint.tgt_vec = tok_tgt
        return datapoint
