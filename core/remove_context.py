import pickle

import pandas as pd

"""
去除bug和fix的公共context
"""

dataset = "/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/pdg/no-context-preserved/data.pkl"

if __name__ == '__main__':
    f = open(dataset, 'rb')
    t = pd.read_pickle(f)
    for tt in t:
        g1_nodes = tt['jsgraph1']['node_features']
        g2_nodes = tt['jsgraph2']['node_features']
        # g1_nodes_new = g1_nodes
        # g2_nodes_new = g2_nodes
        g1_nodes_new = dict()
        g2_nodes_new = dict()
        for k1 in g1_nodes.keys():
            is_in = False
            for k2 in g2_nodes.keys():
                if g1_nodes[k1][0] == g2_nodes[k2][0] and g1_nodes[k1][1] == g2_nodes[k2][1] and k1==k2:
                    is_in = True
                    break
            if not is_in:
                g1_nodes_new[k1] = g1_nodes[k1]

        for k2 in g2_nodes.keys():
            is_in = False
            for k1 in g1_nodes.keys():
                if g1_nodes[k1][0] == g2_nodes[k2][0] and g1_nodes[k1][1] == g2_nodes[k2][1] and k1==k2:
                    is_in = True
                    break
            if not is_in:
                g2_nodes_new[k2] = g2_nodes[k2]

        g1_nodes_added = dict()
        g2_nodes_added = dict()
        # for e in tt['jsgraph1']['graph']:
        #     if e[0] in g1_nodes_new.keys() and e[1] not in g1_nodes_new.keys():
        #         g1_nodes_added[e[1]] = g1_nodes[e[1]]
        #         for k2 in g2_nodes.keys():
        #             if g2_nodes[k2][0] == g1_nodes[e[1]][0] and g2_nodes[k2][1] == g1_nodes[e[1]][1]:
        #                 g2_nodes_added[k2] = g2_nodes[k2]
        #                 break
        #     if e[1] in g1_nodes_new.keys() and e[0] not in g1_nodes_new.keys():
        #         g1_nodes_added[e[0]] = g1_nodes[e[0]]
        #         for k2 in g2_nodes.keys():
        #             if g2_nodes[k2][0] == g1_nodes[e[0]][0] and g2_nodes[k2][1] == g1_nodes[e[0]][1]:
        #                 g2_nodes_added[k2] = g2_nodes[k2]
        #                 break
        # for e in tt['jsgraph2']['graph']:
        #     if e[0] in g2_nodes_new.keys() and e[1] not in g2_nodes_new.keys():
        #         g2_nodes_added[e[1]] = g2_nodes[e[1]]
        #         for k1 in g1_nodes.keys():
        #             if g1_nodes[k1][0] == g2_nodes[e[1]][0] and g1_nodes[k1][1] == g2_nodes[e[1]][1]:
        #                 g1_nodes_added[k1] = g1_nodes[k1]
        #                 break
        #     if e[1] in g2_nodes_new.keys() and e[0] not in g2_nodes_new.keys():
        #         g2_nodes_added[e[0]] = g2_nodes[e[0]]
        #         for k1 in g1_nodes.keys():
        #             if g1_nodes[k1][0] == g2_nodes[e[0]][0] and g1_nodes[k1][1] == g2_nodes[e[0]][1]:
        #                 g1_nodes_added[k1] = g1_nodes[k1]
        #                 break
        g1_nodes_new.update(g1_nodes_added)
        g2_nodes_new.update(g2_nodes_added)

        g1_edges_new = []
        g2_edges_new = []
        for e in tt['jsgraph1']['graph']:
            if e[0] in g1_nodes_new.keys() or e[1] in g1_nodes_new.keys():
                g1_edges_new.append(e)
        for e in tt['jsgraph2']['graph']:
            if e[0] in g2_nodes_new.keys() or e[1] in g2_nodes_new.keys():
                g2_edges_new.append(e)

        # for e in g1_edges_new:
        #     if e[0] not in g1_nodes_new.keys():
        #         g1_nodes_new[e[0]] = g1_nodes[e[0]]
        #     if e[1] not in g1_nodes_new.keys():
        #         g1_nodes_new[e[1]] = g1_nodes[e[1]]
        # for e in g2_edges_new:
        #     if e[0] not in g2_nodes_new.keys():
        #         g2_nodes_new[e[0]] = g2_nodes[e[0]]
        #     if e[1] not in g2_nodes_new.keys():
        #         g2_nodes_new[e[1]] = g2_nodes[e[1]]
        tt['jsgraph1']['node_features'] = g1_nodes_new
        tt['jsgraph2']['node_features'] = g2_nodes_new
        tt['graph_size1'] = len(tt['jsgraph1']['node_features'])
        tt['graph_size2'] = len(tt['jsgraph2']['node_features'])
        tt['jsgraph1']['graph'] = g1_edges_new
        tt['jsgraph2']['graph'] = g2_edges_new

    f = open(dataset, 'wb')
    pickle.dump(t, f, 2)
    f.close()
