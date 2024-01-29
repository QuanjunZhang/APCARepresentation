import dgl

from utils.pymodels_util import pad_to_max, stack_seq_to_tensor


def collate_graph_for_classification(samples):
    """
    Collate Function for Sequence
    :param samples: Samples for each batch
    :return:
    """
    datapoint_list = samples
    batch_dict = dict()
    # Graphs
    graphs = [dp.function_graph for dp in datapoint_list]
    tgt = [dp.tgt for dp in datapoint_list]
    batched_graph = dgl.batch(graphs)
    batch_dict['graphs'] = batched_graph

    # Target
    batch_dict['tgt'] = tgt

    # Function
    batch_dict['function'] = [dp.function for dp in datapoint_list]
    return batch_dict


def collate_graph_for_classification2(samples):
    """
    Collate Function for Treelstm only
    :param samples: Samples for each batch
    :return:
    """
    datapoint_list = samples
    batch_dict = dict()
    # Graphs
    graphs1 = [dp.function_graph1 for dp in datapoint_list]
    graphs2 = [dp.function_graph2 for dp in datapoint_list]
    tgt = [dp.tgt for dp in datapoint_list]
    batched_graph1 = dgl.batch(graphs1)
    batch_dict['graphs1'] = batched_graph1
    batched_graph2 = dgl.batch(graphs2)
    batch_dict['graphs2'] = batched_graph2

    # Target
    batch_dict['tgt'] = tgt

    # Function
    batch_dict['function1'] = [dp.function1 for dp in datapoint_list]
    batch_dict['function2'] = [dp.function2 for dp in datapoint_list]
    return batch_dict

def collate_graph_for_summarization(samples):
    """
    Collate the graphs for Summarization Tasks
    :param samples:
    :return:
    """
    datapoint_list = samples
    batch_dict = dict()

    # Graphs
    graphs = [dp.function_graph for dp in datapoint_list]
    batched_graph = dgl.batch(graphs)
    batch_dict['graphs'] = batched_graph

    # Targets
    tgts = [dp.tgt_vec for dp in datapoint_list]
    batch_dict["tgt_lens"] = [len(tgt) for tgt in tgts]
    batch_dict["tgt"] = [dp.tgt for dp in datapoint_list]
    largest_tgt_len = max(batch_dict['tgt_lens'])
    tgt_ts = [pad_to_max(t, largest_tgt_len) for t in tgts]
    batch_dict['tgt_tensors'] = stack_seq_to_tensor(tgt_ts)
    return batch_dict
