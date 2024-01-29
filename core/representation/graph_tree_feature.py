import os
import pickle
import re

import pandas as pd

from dot_parser import parse_dot_to_graph
from feature_utils import features

"""
构建bug和fix代码对应的属性图、抽象语法树和特征
"""


class GraphModel:
    def __init__(self):
        self.function_id = 0
        self.target = -1
        self.jsgraph1 = None
        self.jsgraph_file_path1 = ""
        self.function1 = ""
        self.graph_size1 = 0
        self.jsgraph2 = None
        self.jsgraph_file_path2 = ""
        self.function2 = ""
        self.graph_size2 = 0
        self.java_file_path1 = ""
        self.java_file_path2 = ""

    def len(self):
        if self.jsgraph1 is None:
            l1 = 0
        else:
            l1 = len(self.jsgraph1.nodes)
        if self.jsgraph2 is None:
            l2 = 0
        else:
            l2 = len(self.jsgraph2.nodes)
        return l1, l2

    def __dict__(self):
        return {
            'function_id': self.function_id,
            'target': self.target,
            'jsgraph1': self.jsgraph1.__dict__(),
            'jsgraph_file_path1': self.jsgraph_file_path1,
            'function1': self.function1,
            'graph_size1': self.graph_size1,
            'java_file_path1': self.java_file_path1,
            'jsgraph2': self.jsgraph2.__dict__(),
            'jsgraph_file_path2': self.jsgraph_file_path2,
            'function2': self.function2,
            'graph_size2': self.graph_size2,
            'java_file_path2': self.java_file_path2
        }


class Node:
    def __init__(self, type_str: str, code_str: str, id: int):
        self.type_str = type_str
        self.code_str = code_str
        self.id = id

    def __dict__(self):
        return {self.id: [self.type_str, self.code_str]}


class JsGraph:
    def __init__(self):
        self.edges = []
        self.nodes = []

    def __dict__(self):
        _node_features = dict()
        for n in self.nodes:
            _node_features.update(n.__dict__())

        return {
            'graph': self.edges,
            'node_features': _node_features
        }


def make_graph(nodes1, edges1, nodes2, edges2, function_id: int, target: int, jsgraph_file_path1: str,
               jsgraph_file_path2: str, function1: str, function2: str,
               java_file_path1: str, java_file_path2: str):
    graph1 = JsGraph()
    graph2 = JsGraph()
    graph1.nodes = nodes1
    graph1.edges = edges1
    graph2.nodes = nodes2
    graph2.edges = edges2
    gm = GraphModel()
    gm.function_id = function_id
    gm.target = target
    gm.jsgraph_file_path1 = jsgraph_file_path1
    gm.function1 = function1
    gm.java_file_path1 = java_file_path1
    gm.jsgraph1 = graph1
    gm.jsgraph_file_path2 = jsgraph_file_path2
    gm.function2 = function2
    gm.java_file_path2 = java_file_path2
    gm.jsgraph2 = graph2
    return gm


def parse_node(path):
    nodes = []
    labels = []
    with open(path, 'r', encoding="utf8") as fff:
        for line in fff.readlines():
            mm = re.match("id:([0-9]+) type:(.*) code:(.*)", line)
            nodes.append(Node(mm.group(2), mm.group(3), int(mm.group(1))))
            labels.append(mm.group(2))
        fff.close()
    return nodes, labels


def parse_edge(path):
    result = []
    with open(path, 'r', encoding="utf8") as fff:
        for line in fff.readlines():
            mm = re.match("([0-9]+) -> ([0-9]+)", line)
            # default is ast
            result.append([int(mm.group(1)), int(mm.group(2)), 0])
        f.close()
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ttype')
    args = parser.parse_args()
    ttype_value = args.ttype

    ttype = "cfg"
    # for ddg, cdg and cpg
    joern_path = "/Users/tom/joern-master"
    # for cfg and pdg
    property_graph_jar = "../jar/property-graph/demo.jar"
    # for ast diff
    jdiff_jar = "../jar/jdiff.jar"
    # for extracting feature
    add_jar = "../jar/add.jar"
    # for extracting feature
    coming_jar = "/Users/tom/Downloads/learning-program-representation-master/coming_jar/coming.jar"
    # output
    output_path = "/Users/tom/Downloads/learning-program-representation-master/data/output"
    coming_output = "/Users/tom/Downloads/learning-program-representation-master/coming_results"
    add_output = "/Users/tom/Downloads/learning-program-representation-master/add_results"
    # java8
    java8_path = "/Library/Java/JavaVirtualMachines/jdk1.8.0_191.jdk/Contents/Home/bin/java"

    # data.pkl
    source_data = "/Users/tom/Downloads/learning-program-representation-master/data/custom/sequence/data.pkl"
    # node vocabulary
    dst_vocab = "/Users/tom/Downloads/learning-program-representation-master/data/custom/tree/dict/node_vocab_dict.pkl"
    # dataset
    dst_datasets = {
        "ast": "/Users/tom/Downloads/learning-program-representation-master/data/custom/tree/context-preserved/data.pkl",
        "cfg": "/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/cfg/context-preserved/data.pkl",
        "cdg": "/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/cdg/context-preserved/data.pkl",
        "ddg": "/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/ddg/context-preserved/data.pkl",
        "cpg": "/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/cpg/context-preserved/data.pkl",
        "pdg": "/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/pdg/context-preserved/data.pkl",
        "feature": "/Users/tom/Downloads/learning-program-representation-master/data/custom/feature/data.pkl",
    }

    data = []
    node_vocab_dict = dict()
    node_vocab_dict["name"] = "NodeVocabDictionary"
    node_vocab_dict["word2index"] = {"SOS": 0, "EOS": 1, "OOV": 2, "PAD": 3}
    node_vocab_dict["word2count"] = {"SOS": 1, "EOS": 1, "OOV": 1, "PAD": 1}
    node_vocab_dict["index2word"] = {0: "SOS", 1: "EOS", 2: "OOV", 3: "PAD"}
    label_id = 4
    with open(source_data, 'rb') as f:
        t = pd.read_pickle(f)
        length1 = len(t)
        for i in range(length1):

            fid = t[i]['function_id']
            print(fid)

            bug_code = t[i]["function1"]
            repair_code = t[i]["function2"]
            tar = t[i]["target"]
            patch_info = t[i]["patch_info"]

            name_pattern = "{}_{}_{}_{}_{}".format(patch_info["patch_tool"], patch_info["project"],
                                                   patch_info["bug_id"], patch_info["patch_name"],
                                                   patch_info["is_correct"])

            # save code
            target_dir = "{}/patch/{}".format(output_path, name_pattern)
            bug_path = "{}/bug.java".format(target_dir)
            fix_path = "{}/fixed.java".format(target_dir)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            with open(bug_path, 'w') as ff:
                ff.write(bug_code)
                ff.flush()
                ff.close()
            with open(fix_path, 'w') as ff:
                ff.write(repair_code)
                ff.flush()
                ff.close()

            if ttype == "feature":
                # coming

                os.system("rm -rf {} && mkdir {}".format(
                    "{}/{}".format(coming_output, name_pattern),
                    "{}/{}".format(coming_output, name_pattern)
                ))
                os.system("{} -jar {} -input filespair -location {}:{} -output {}/{}".format(java8_path,
                                                                                             coming_jar,
                                                                                             bug_path,
                                                                                             fix_path,
                                                                                             coming_output,
                                                                                             name_pattern))

                # add
                # git diff
                diff_path = "{}/diff/diff_{}.patch".format(output_path, name_pattern)
                os.system("git diff {} {} > {}".format(bug_path,
                                                       fix_path,
                                                       diff_path))
                os.system(
                    "java -jar {} --launcherMode REPAIR_PATTERNS --buggySourceDirectory {} --diff {} --bugId {} --output {} ".format(
                        add_jar,
                        bug_path,
                        diff_path,
                        name_pattern,
                        add_output
                    ))

                ffeatures = features(coming_path="{}/{}/change_frequency.json".format(coming_output, name_pattern),
                                     add_path="{}/{}_repair_patterns.json".format(add_output, name_pattern))

                for fe in ffeatures.keys():
                    for label in ffeatures[fe].keys():
                        num = node_vocab_dict["word2count"].get(label, 0)
                        if num > 0:
                            node_vocab_dict["word2count"][label] = num + 1
                        else:
                            node_vocab_dict["word2count"][label] = 1
                            node_vocab_dict["word2index"][label] = label_id
                            node_vocab_dict["index2word"][label_id] = label
                            label_id += 1
                ffeatures['target'] = tar
                data.append(ffeatures)
                continue
            if ttype == "ast":
                if not os.path.exists("{}/ast/{}".format(output_path, name_pattern)):
                    os.makedirs("{}/ast/{}".format(output_path, name_pattern))
                if not os.path.exists("{}/ast/{}/bug.node".format(output_path, name_pattern)):
                    os.system("java -jar {} 4 {} {}".format(jdiff_jar, bug_path, fix_path))
                    if not os.path.exists("{}/bug.node".format(target_dir)):
                        continue
                    os.system("mv {} {}".format("{}/bug.node".format(target_dir),
                                                "{}/ast/{}/bug.node".format(output_path, name_pattern)))
                    os.system("mv {} {}".format("{}/bug.edge".format(target_dir),
                                                "{}/ast/{}/bug.edge".format(output_path, name_pattern)))
                    os.system("mv {} {}".format("{}/fixed.node".format(target_dir),
                                                "{}/ast/{}/fixed.node".format(output_path, name_pattern)))
                    os.system("mv {} {}".format("{}/fixed.edge".format(target_dir),
                                                "{}/ast/{}/fixed.edge".format(output_path, name_pattern)))


            elif  ttype == "pdg" or ttype=="cfg":
                if not os.path.exists(os.path.join(output_path, ttype, name_pattern)):
                    os.makedirs(os.path.join(output_path, ttype, name_pattern))
                c1 = "java -jar {} -d {} -{}".format(property_graph_jar, bug_path, ttype[0])
                c2 = "mv {} {}".format(os.path.join(target_dir, ttype.upper(), "bug_{}.dot".format(ttype)),
                                       os.path.join(output_path, ttype.upper(), name_pattern,
                                                    "bug.dot"))
                c3 = "rm -rf {}".format(os.path.join(target_dir, ttype.upper()))

                c4 = "java -jar {} -d {} -{}".format(property_graph_jar, fix_path, ttype[0])
                c5 = "mv {} {}".format(os.path.join(target_dir, ttype.upper(), "fixed_{}.dot".format(ttype)),
                                       os.path.join(output_path, ttype, name_pattern,
                                                    "fixed.dot"))
                c6 = "rm -rf {}".format(os.path.join(target_dir, ttype.upper()))

                os.system(
                    "{} && {} && {} && {} && {} && {}".format(c1, c2, c3, c4, c5, c6))

                parse_dot_to_graph(os.path.join(output_path, ttype, name_pattern, "bug.dot"))
                parse_dot_to_graph(os.path.join(output_path, ttype, name_pattern, "fixed.dot"))

            elif ttype == "ddg" or ttype == "cdg" or ttype == "cpg": # can also be used on cfg
                temp_path = '_'.join([ttype.upper(), "TEMP"])
                # joern-parse
                if not os.path.exists("{}/joern-bin/{}/bug.bin".format(output_path, name_pattern)):
                    os.system(
                        "cd {} && ./joern-parse {}".format(joern_path, bug_path))
                    if os.path.exists("{}/cpg.bin".format(joern_path)):
                        if not os.path.exists("{}/joern-bin/{}".format(output_path, name_pattern)):
                            os.makedirs("{}/joern-bin/{}".format(output_path, name_pattern))
                        os.system("cp {}/cpg.bin {}/joern-bin/{}/bug.bin".format(joern_path, output_path, name_pattern))
                        os.system("rm -rf {}/cpg.bin".format(joern_path))
                    else:
                        print(name_pattern)
                        continue

                if not os.path.exists("{}/joern-bin/{}/fixed.bin".format(output_path, name_pattern)):
                    os.system(
                        "cd {} && ./joern-parse {}".format(joern_path, fix_path))
                    if os.path.exists("{}/cpg.bin".format(joern_path)):
                        if not os.path.exists("{}/joern-bin/{}".format(output_path, name_pattern)):
                            os.makedirs("{}/joern-bin/{}".format(output_path, name_pattern))
                        os.system(
                            "cp {}/cpg.bin {}/joern-bin/{}/fixed.bin".format(joern_path, output_path, name_pattern))
                        os.system("rm -rf {}/cpg.bin".format(joern_path))
                    else:
                        print(name_pattern)
                        continue
                # joern-export
                if not os.path.exists("{}/{}/{}/bug.dot".format(output_path, ttype, name_pattern)):
                    os.system(
                        "cd {} && ./joern-export {}/joern-bin/{}/bug.bin --repr {} --out {}".format(joern_path,
                                                                                                    output_path,
                                                                                                    name_pattern,
                                                                                                    ttype if ttype != "cpg" else ttype + "14",
                                                                                                    "{}/{}".format(
                                                                                                        output_path,
                                                                                                        temp_path)))
                    if not os.path.exists("{}/{}/0-{}.dot".format(output_path, temp_path, ttype)):
                        os.system("rm -rf {}".format("{}/{}".format(output_path, temp_path)))
                        print(name_pattern)
                        continue
                    if not os.path.exists("{}/{}/{}".format(output_path, ttype, name_pattern)):
                        os.makedirs("{}/{}/{}".format(output_path, ttype, name_pattern))
                    os.system("cp {} {}".format("{}/{}/0-{}.dot".format(output_path, temp_path, ttype),
                                                "{}/{}/{}/bug.dot".format(output_path, ttype, name_pattern)))
                    os.system("rm -rf {}".format("{}/{}".format(output_path, temp_path)))
                if not os.path.exists("{}/{}/{}/fixed.dot".format(output_path, ttype, name_pattern)):
                    os.system(
                        "cd {} && ./joern-export {}/joern-bin/{}/fixed.bin --repr {} --out {}".format(joern_path,
                                                                                                      output_path,
                                                                                                      name_pattern,
                                                                                                      ttype if ttype != "cpg" else ttype + "14",
                                                                                                      "{}/{}".format(
                                                                                                          output_path,
                                                                                                          temp_path)))
                    if not os.path.exists("{}/{}/0-{}.dot".format(output_path, temp_path, ttype)):
                        os.system("rm -rf {}".format("{}/{}".format(output_path, temp_path)))
                        print(name_pattern)
                        continue
                    if not os.path.exists("{}/{}/{}".format(output_path, ttype, name_pattern)):
                        os.makedirs("{}/{}/{}".format(output_path, ttype, name_pattern))
                    os.system("cp {} {}".format("{}/{}/0-{}.dot".format(output_path, temp_path, ttype),
                                                "{}/{}/{}/fixed.dot".format(output_path, ttype, name_pattern)))
                    os.system("rm -rf {}".format("{}/{}".format(output_path, temp_path)))
                parse_dot_to_graph(os.path.join(output_path, ttype, name_pattern, "bug.dot"), is_joern=True)
                parse_dot_to_graph(os.path.join(output_path, ttype, name_pattern, "fixed.dot"), is_joern=True)

            ns1, ls1 = parse_node("{}/{}/{}/bug.node".format(output_path, ttype, name_pattern))
            eds1 = parse_edge("{}/{}/{}/bug.edge".format(output_path, ttype, name_pattern))
            ns2, ls2 = parse_node("{}/{}/{}/fixed.node".format(output_path, ttype, name_pattern))
            eds2 = parse_edge("{}/{}/{}/fixed.edge".format(output_path, ttype, name_pattern))
            if len(ns1) == 0 or len(ns2) == 0 or len(eds1) == 0 or len(eds2) == 0:
                print(name_pattern)
                continue

            gg = make_graph(ns1, eds1, ns2, eds2, fid, tar, "", "", bug_code, repair_code, "", "")
            gg.graph_size1, gg.graph_size2 = gg.len()

            data.append(gg.__dict__())

            for label in ls1:
                num = node_vocab_dict["word2count"].get(label, 0)
                if num > 0:
                    node_vocab_dict["word2count"][label] = num + 1
                else:
                    node_vocab_dict["word2count"][label] = 1
                    node_vocab_dict["word2index"][label] = label_id
                    node_vocab_dict["index2word"][label_id] = label
                    label_id += 1
            for label in ls2:
                num = node_vocab_dict["word2count"].get(label, 0)
                if num > 0:
                    node_vocab_dict["word2count"][label] = num + 1
                else:
                    node_vocab_dict["word2count"][label] = 1
                    node_vocab_dict["word2index"][label] = label_id
                    node_vocab_dict["index2word"][label_id] = label
                    label_id += 1

    old_dict = {}
    # if exist
    if os.path.exists(dst_vocab):
        old_dict = pd.read_pickle(dst_vocab)
    # update
    if old_dict.get("word2index", -1) != -1:
        for k in old_dict["word2index"].keys():
            if node_vocab_dict["word2index"].get(k, -1) < 0:
                node_vocab_dict["word2index"][k] = label_id
                node_vocab_dict["index2word"][label_id] = k
                node_vocab_dict["word2count"][k] = old_dict["word2count"][k]
                label_id += 1
            else:
                node_vocab_dict["word2count"][k] += old_dict["word2count"][k]
    # save node_vocab_dict
    f = open(dst_vocab, 'wb')
    pickle.dump(node_vocab_dict, f, 2)
    f.close()

    # save data
    f = open(dst_datasets[ttype], 'wb')
    pickle.dump(data, f, 2)
    f.close()
