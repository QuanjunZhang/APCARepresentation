import dgl
import torch
from bases.base_formatter import BaseFormatter
from utils.pymodels_util import pad_to_max, stack_seq_to_tensor


class BaseGraphFormatter(BaseFormatter):
    def __init__(self, config, name="BaseGraphFormatter"):
        """
        Contain function for Graph Formatting
        """
        BaseFormatter.__init__(self, config, name)

    def _convert_to_single_edge_graph(self, jsgraph, token_vd, node_vd):
        """
        Convert the JSGraph to DGL Graph
        Since it is Isotropic Graph, edge type is not considered.
        We will map all edges into a single type - 0
        :param jsgraph: JSGraph
        :param token_vd: Token Vocab Dictionary
        :param node_vd: Node Vocab Dictionary
        :return: Return a DGL Graph
        """
        edges = jsgraph['graph']
        node_features = jsgraph['node_features']
        src_edges = [x[0] for x in edges]
        dst_edges = [x[1] for x in edges]
        # Each element i is map to each element k in src and dst
        g = dgl.graph((src_edges, dst_edges))
        assert g.num_nodes() == len(node_features)

        node_feat_vecs = [0] * g.num_nodes()
        node_feat_lens = [0] * g.num_nodes()
        for key, node_feat in node_features.items():
            node_feat, node_feat_len = self.get_feats(node_feat, token_vd, node_vd)
            node_feat_vecs[int(key)] = node_feat
            node_feat_lens[int(key)] = node_feat_len
        g.ndata['node_feat'] = stack_seq_to_tensor(node_feat_vecs)
        g.ndata['node_len'] = torch.tensor(node_feat_lens, dtype=torch.long)
        if self.config.use_cuda:
            g = g.to('cuda')
        return g

    def _convert_to_multi_edge_dglgraph(self, jsgraph, token_vd, node_vd):
        """
        Convert the JSGraph to DGL Graph
        :param jsgraph: JSGraph
        :param token_vd: Token Vocab Dictionary
        :param node_vd: Node Vocab Dictionary
        :return: Return a DGL Graph
        """
        edges = jsgraph['graph']
        node_features = jsgraph['node_features']
        src_nodes, dst_nodes, edges_type = self.get_edges_based_on_config(edges)
        subset_node_features = self.get_sub_node_features(node_features, src_nodes, dst_nodes)

        # Each element i is map to each element k in src and dst
        src_nodes, dst_nodes, subset_node_features = self.normalize_node_id(src_nodes, dst_nodes, subset_node_features)
        edges_type = self.get_norm_edge_type(edges_type)
        g = dgl.graph((dst_nodes, src_nodes))
        # if g.num_nodes() != len(subset_node_features):
        #     print(0)
        assert g.num_nodes() == len(subset_node_features)
        node_feat_vecs = [0] * g.num_nodes()
        node_feat_lens = [0] * g.num_nodes()
        mask_list = [0] * g.num_nodes()
        for key, raw_node_feat in subset_node_features.items():
            node_feat, node_feat_len = self.get_feats(raw_node_feat, token_vd, node_vd)
            node_feat_vecs[int(key)] = node_feat
            node_feat_lens[int(key)] = node_feat_len
            if int(key) not in set(src_nodes):
                mask_list[int(key)] = 1

        g.ndata['node_feat'] = stack_seq_to_tensor(node_feat_vecs)
        g.ndata['node_len'] = torch.tensor(node_feat_lens, dtype=torch.long)
        g.ndata['mask'] = torch.tensor(mask_list, dtype=torch.long)
        g.edata['edge_type'] = torch.tensor(edges_type, dtype=torch.long)
        if self.config.self_loop:
            g = dgl.add_self_loop(g)
        if self.config.reverse_edge:
            g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
        if self.config.use_cuda:
            g = g.to('cuda')
        return g

    def get_feats(self, node_feat, token_vd, node_vd):
        """
        Get the respective features based on either structure or textual
        Sample Node Feature: ['Function', '', 0, False]
        :param node_feat: The four elements of Node Item
        :param token_vd: Token Vocab Dictionary
        :param node_vd: Node Vocab Dictionary
        :return: Return a vectorized vec
        """
        use_feats = self.config.node_emb_layer['use_nfeature']
        if use_feats == "structure":
            struct_vec, struct_len = self.get_structure_feat(node_feat, node_vd)
            return struct_vec, struct_len
        elif use_feats == "textual":
            text_vec, text_len = self.get_textual_feat(node_feat, token_vd)
            return text_vec, text_len
        elif use_feats in ["both-full-textual", "distinct-feats", "both"]:
            struct_vec, struct_len = self.get_structure_feat(node_feat, node_vd)
            text_vec, text_len = self.get_textual_feat(node_feat, token_vd)
            assert len(struct_vec) == 1
            text_vec.insert(0, struct_vec[0])
            return text_vec, text_len + 1
        elif use_feats in ["leaf-textual"]:
            node_type = node_feat[0]
            code_str = node_feat[1]
            tk_code_str = self.t3_parser.tokenize(code_str)
            nidxes, nlen = self.tokenize_sentence(tk_code_str, token_vd, eos=False)
            if nlen == 1:
                node_vec, node_len = self.get_textual_feat(node_feat, token_vd)
            else:
                node_vec, node_len = self.get_structure_feat(node_feat, node_vd)
            return node_vec, node_len
        else:
            raise NotImplementedError

    def get_structure_feat(self, node_feat, node_vd, padded=False):
        """
        Sample Node Feature: ['Function', '', 0, False]
        :param node_feat: The four elements of Node Item
        :param node_vd: Node Vocab Dictionary
        :param padded: Specify if you want to pad the sequnece
        :return:
        """
        # We only use Node Type as Features
        node_type = node_feat[0]
        nidxes, nlen = self.tokenize_sentence(node_type, node_vd, eos=False)
        if padded:
            nidxes = pad_to_max(nidxes, self.config.max_code_token_len, pad_token=None)
            if nlen > self.config.max_code_token_len:
                nlen = self.config.max_code_token_len
        return nidxes, nlen

    def get_textual_feat(self, node_feat, token_vd):
        """
        Sample Node Feature: ['Function', '', 0, False]
        :param node_feat: The four elements of Node Item
        :param token_vd: Token Vocab Dictionary
        :return:
        """
        # Use the Code String as the feature
        code_str = node_feat[1]
        node_type = node_feat[0]
        tk_code_str = self.t3_parser.tokenize(code_str)
        nidxes, nlen = self.tokenize_sentence(tk_code_str, token_vd, eos=False)
        if not nidxes:
            # If there is no code in the node, we use an empty code placeholder
            nidxes.append(token_vd.get_w2i("<EMPTY_CODE>"))
            nlen = 1
        nidxes = pad_to_max(nidxes, self.config.max_code_token_len, pad_token=None)
        if nlen > self.config.max_code_token_len:
            nlen = self.config.max_code_token_len
        return nidxes, nlen

    def get_edges_based_on_config(self, edges):
        """
        Get the Edges by filtering out the non-needed stuff
        {'IS_AST_PARENT': 0, 'isCFGParent': 1, "POST_DOM": 2,
        "FLOWS_TO": 3, "USE": 4, "DEF": 5, 'REACHES': 6, "CONTROLS": 7}
        :param edges: List of Edges
        :return: Return three list, src_edges, dst_edges and edges_type
        """
        edge_types = []
        src_nodes = []
        dst_nodes = []
        for src, dst, etype in edges:
            if etype in self.config.edge_type_list:
                edge_types.append(etype)
                src_nodes.append(src)
                dst_nodes.append(dst)
        return src_nodes, dst_nodes, edge_types

    @staticmethod
    def normalize_node_id(src_edges, dst_edges, node_features):
        """
        One of the key problem is that after you get the nodes by
        different edge type, the node id is not sequential anymore. We need
        to normalize it back to 0 to len(total number of nodes)
        :param src_edges: Src Node
        :param dst_edges: Dest Node
        :param node_features: Node Features
        :return: Return the norm_src, norm_dst and norm_feats
        """
        normalized_dict = dict()
        norm_src = []
        norm_dst = []
        norm_feat = dict()
        for node_id in src_edges:
            if node_id not in normalized_dict:
                normalized_dict[node_id] = len(normalized_dict)
            norm_src.append(normalized_dict[node_id])

        for node_id in dst_edges:
            if node_id not in normalized_dict:
                normalized_dict[node_id] = len(normalized_dict)
            norm_dst.append(normalized_dict[node_id])

        for key, value in node_features.items():
            norm_feat[normalized_dict[int(key)]] = value

        return norm_src, norm_dst, norm_feat

    @staticmethod
    def get_norm_edge_type(edge_list):
        """
        Make sure the edge list start from 0
        :param edge_list: List of Edge Type
        :return:
        """
        norm_list = []
        normalized_dict = dict()
        for etype in edge_list:
            if etype not in normalized_dict:
                normalized_dict[etype] = len(normalized_dict)
            norm_list.append(normalized_dict[etype])
        assert len(norm_list) == len(edge_list)
        return norm_list

    @staticmethod
    def get_sub_node_features(node_features, src_nodes, dst_nodes):
        """
        Only get the sub node features from the node features
        Narrow down first to ensure that no problem later
        :param node_features: A dictionary of node features, with node id as key
        :param src_nodes: Src Nodes
        :param dst_nodes: Dst Nodes
        :return:
        """
        narrowed = dict()
        for key in set(src_nodes + dst_nodes):
            narrowed[key] = node_features.get(int(key), ["UNKNOWN", ''])
        # for key, value in node_features.items():
        #     if int(key) in src_nodes or int(key) in dst_nodes:
        #         narrowed[key] = value
        return narrowed
