import openpyxl

header = [
    # sequence 3
    "lstm", "bilstm", "transformer-encoder",
    # feature 6
    "naivebayes-feature", "svm-feature", "xgboost-feature",
    "naivebayes-tfidf", "svm-tfidf", "xgboost-tfidf",
    # tree 12
    "treelstm-textual-context-preserved", "treelstm-structure-context-preserved", "treelstm-both-context-preserved",
    "treelstm-textual-no-context-preserved", "treelstm-structure-no-context-preserved",
    "treelstm-both-no-context-preserved",
    "tbcnn-textual-context-preserved", "tbcnn-structure-context-preserved", "tbcnn-both-context-preserved",
    "tbcnn-textual-no-context-preserved", "tbcnn-structure-no-context-preserved", "tbcnn-both-no-context-preserved",
    # graph 6*3*3=54
    "gat-ast-textual", "gat-ast-structure", "gat-ast-both", "gcn-ast-textual", "gcn-act-structure", "gcn-ast-both",
    "ggnn-ast-textual", "ggnn-ast-structure", "ggnn-ast-both",
    "gat-cfg-textual", "gat-cfg-structure", "gat-cfg-both", "gcn-cfg-textual", "gcn-cfg-structure", "gcn-cfg-both",
    "ggnn-cfg-textual", "ggnn-cfg-structure", "ggnn-cfg-both",
    "gat-cdg-textual", "gat-cdg-structure", "gat-cdg-both", "gcn-cdg-textual", "gcn-cdg-structure", "gcn-cdg-both",
    "ggnn-cdg-textual", "ggnn-cdg-structure", "ggnn-cdg-both",
    "gat-ddg-textual", "gat-ddg-structure", "gat-ddg-both", "gcn-ddg-textual", "gcn-ddg-structure", "gcn-ddg-both",
    "ggnn-ddg-textual", "ggnn-ddg-structure", "ggnn-ddg-both",
    "gat-pdg-textual", "gat-pdg-structure", "gat-pdg-both", "gcn-pdg-textual", "gcn-pdg-structure", "gcn-pdg-both",
    "ggnn-pdg-textual", "ggnn-pdg-structure", "ggnn-pdg-both",
    "gat-cpg-textual", "gat-cpg-structure", "gat-cpg-both", "gcn-cpg-textual", "gcn-cpg-structure", "gcn-cpg-both",
    "ggnn-cpg-textual", "ggnn-cpg-structure", "ggnn-cpg-both",
    # multimodal 5*9=45
    "cfg-textual + ddg-textual + max_pooling", "cfg-textual + ddg-textual + weighted_sum",
    "cfg-textual + ddg-textual + concat",
    "cfg-structure + ddg-structure + max_pooling", "cfg-structure + ddg-structure + weighted_sum",
    "cfg-structure + ddg-structure + concat",
    "cfg-both + ddg-both + max_pooling", "cfg-both + ddg-both + weighted_sum", "cfg-both + ddg-both + concat",
    "cfg-textual + cdg-textual + max_pooling", "cfg-textual + cdg-textual + weighted_sum",
    "cfg-textual + cdg-textual + concat",
    "cfg-structure + cdg-structure + max_pooling", "cfg-structure + cdg-structure + weighted_sum",
    "cfg-structure + cdg-structure + concat",
    "cfg-both + cdg-both + max_pooling", "cfg-both + cdg-both + weighted_sum", "cfg-both + cdg-both + concat",
    "ddg-textual + cdg-textual + max_pooling", "ddg-textual + cdg-textual + weighted_sum",
    "ddg-textual + cdg-textual + concat",
    "ddg-structure + cdg-structure + max_pooling", "ddg-structure + cdg-structure + weighted_sum",
    "ddg-structure + cdg-structure + concat",
    "ddg-both + cdg-both + max_pooling", "ddg-both + cdg-both + weighted_sum", "ddg-both + cdg-both + concat",
    "cfg-textual + ddg-textual+ cdg-textual + max_pooling", "cfg-textual + ddg-textual+ cdg-textual + weighted_sum",
    "cfg-textual + ddg-textual+ cdg-textual + concat",
    "cfg-structure + ddg-structure + cdg-structure + max_pooling",
    "cfg-structure + ddg-structure + cdg-structure + weighted_sum",
    "cfg-structure + ddg-structure + cdg-structure + concat",
    "cfg-both + ddg-both + cdg-both + max_pooling", "cfg-both + ddg-both + cdg-both + weighted_sum",
    "cfg-both + ddg-both + cdg-both + concat",
    "cfg-textual + ddg-textual + cdg-textual + tree-textual + sequence + max_pooling",
    "cfg-textual + ddg-textual + cdg-textual + tree-textual + sequence + weighted_sum",
    "cfg-textual + ddg-textual + cdg-textual + tree-textual + sequence + concat",
    "cfg-structure + ddg-structure + cdg-structure + tree-structure + sequence + max_pooling",
    "cfg-structure + ddg-structure + cdg-structure + tree-structure + sequence + weighted_sum",
    "cfg-structure + ddg-structure + cdg-structure + tree-structure + sequence + concat",
    "cfg-both + ddg-both + cdg-both + tree-both + sequence + max_pooling",
    "cfg-both + ddg-both + cdg-both + tree-both + sequence + weighted_sum",
    "cfg-both + ddg-both + cdg-both + tree-both + sequence + concat"]

# k_fold = [True] * 75 + [False] * 45

# rq1 四个类型的比较
rq1_header = ["sequence", "feature", "tree", "graph"]
rq1_kfold = [True] * 4

# rq2 类内部的比较
rq2_header = ["tfidf-naivebayes", "tfidf-svm", "tfidf-xgboost",
              "code-naivebayes", "code-svm", "code-xgboost",
              "context-naivebayes", "context-svm", "context-xgboost",
              "pattern-naivebayes", "pattern-svm", "pattern-xgboost",
              "code-context-naivebayes", "code-context-svm", "code-context-xgboost",
              "code-pattern-naivebayes", "code-pattern-svm", "code-pattern-xgboost",
              "context-pattern-naivebayes", "context-pattern-svm", "context-pattern-xgboost",
              "code-context-pattern-naivebayes", "code-context-pattern-svm", "code-context-pattern-xgboost",
              ]
rq2_kfold = [True] * 24

# rq3 node embedding
rq3_header = [
    # tree 6
    "treelstm-textual", "treelstm-structure", "treelstm-both",
    "tbcnn-textual", "tbcnn-structure", "tbcnn-both",
    # graph 5*3*3=45
    "gat-cfg-textual", "gat-cfg-structure", "gat-cfg-both", "gcn-cfg-textual", "gcn-cfg-structure", "gcn-cfg-both",
    "ggnn-cfg-textual", "ggnn-cfg-structure", "ggnn-cfg-both",
    "gat-cdg-textual", "gat-cdg-structure", "gat-cdg-both", "gcn-cdg-textual", "gcn-cdg-structure", "gcn-cdg-both",
    "ggnn-cdg-textual", "ggnn-cdg-structure", "ggnn-cdg-both",
    "gat-ddg-textual", "gat-ddg-structure", "gat-ddg-both", "gcn-ddg-textual", "gcn-ddg-structure", "gcn-ddg-both",
    "ggnn-ddg-textual", "ggnn-ddg-structure", "ggnn-ddg-both",
    "gat-pdg-textual", "gat-pdg-structure", "gat-pdg-both", "gcn-pdg-textual", "gcn-pdg-structure", "gcn-pdg-both",
    "ggnn-pdg-textual", "ggnn-pdg-structure", "ggnn-pdg-both",
    "gat-cpg-textual", "gat-cpg-structure", "gat-cpg-both", "gcn-cpg-textual", "gcn-cpg-structure", "gcn-cpg-both",
    "ggnn-cpg-textual", "ggnn-cpg-structure", "ggnn-cpg-both",
]

rq3_kfold = [True] * 51

# rq4 graph type
rq4_header = [
    "gat-cfg", "gcn-cfg", "ggnn-cfg",
    "gat-ddg", "gcn-ddg", "ggnn-ddg",
    "gat-pdg", "gcn-pdg", "ggnn-pdg",
    "gat-cdg", "gcn-cdg", "ggnn-cdg",
    "gat-cpg", "gcn-cpg", "ggnn-cpg"
]
rq4_kfold = [True] * 15

# rq5 context
rq5_1_header = [
    "treelstm-textual-context-preserved",
    "treelstm-structure-context-preserved",
    "treelstm-both-context-preserved",
    "treelstm-textual-no-context-preserved",
    "treelstm-structure-no-context-preserved",
    "treelstm-both-no-context-preserved",
    "tbcnn-textual-context-preserved",
    "tbcnn-structure-context-preserved",
    "tbcnn-both-context-preserved",
    "tbcnn-textual-no-context-preserved",
    "tbcnn-structure-no-context-preserved",
    "tbcnn-both-no-context-preserved"
]
rq5_1_kfold = [True] * 6 + [False] * 6
rq5_2_header = [
    "cfg-gat-textual",
    "cfg-gat-structure",
    "cfg-gat-both",
    "cfg-gcn-textual",
    "cfg-gcn-structure",
    "cfg-gcn-both",
    "cfg-ggnn-textual",
    "cfg-ggnn-structure",
    "cfg-ggnn-both",
    "cdg-gat-textual",
    "cdg-gat-structure",
    "cdg-gat-both",
    "cdg-gcn-textual",
    "cdg-gcn-structure",
    "cdg-gcn-both",
    "cdg-ggnn-textual",
    "cdg-ggnn-structure",
    "cdg-ggnn-both",
    "ddg-gat-textual",
    "ddg-gat-structure",
    "ddg-gat-both",
    "ddg-gcn-textual",
    "ddg-gcn-structure",
    "ddg-gcn-both",
    "ddg-ggnn-textual",
    "ddg-ggnn-structure",
    "ddg-ggnn-both",
    "pdg-gat-textual",
    "pdg-gat-structure",
    "pdg-gat-both",
    "pdg-gcn-textual",
    "pdg-gcn-structure",
    "pdg-gcn-both",
    "pdg-ggnn-textual",
    "pdg-ggnn-structure",
    "pdg-ggnn-both",
    "cpg-gat-textual",
    "cpg-gat-structure",
    "cpg-gat-both",
    "cpg-gcn-textual",
    "cpg-gcn-structure",
    "cpg-gcn-both",
    "cpg-ggnn-textual",
    "cpg-ggnn-structure",
    "cpg-ggnn-both",
]
rq5_2_kfold = [True] * 45


# rq6 fusion
rq6_header = [
    "feature-sequence",
    "feature-tree",
    "feature-graph",
    "sequence-tree",
    "sequence-graph",
    "tree-graph",
    "feature-sequence-tree",
    "feature-sequence-graph",
    "feature-tree-graph",
    "sequence-tree-graph",
    "feature-sequence-tree-graph"
]
rq6_kfold = [True] * 11


def recognize(s: str):
    # accuracy: 0.8062 | recall: 0.8281 | precision: 0.7854 | f1: 0.8062 | auc: 0.8067 |
    t = s.split("|")
    acc, recall, precision, f1, auc = t[0], t[1], t[2], t[3], t[4]
    return float(acc.split(":")[-1].strip()), float(recall.split(":")[-1].strip()), float(
        precision.split(":")[-1].strip()), float(f1.split(":")[-1].strip()), float(auc.split(":")[-1].strip())


def get_data(k_fold, root=""):
    results = []
    with open(root, 'r') as f:
        lines = list(f.readlines())
        i = 0
        num = 0
        while i < len(lines):
            if lines[i].startswith("accuracy"):
                print(num, lines[i])
                if k_fold[num]:
                    acc1, recall1, precision1, f11, auc1 = recognize(lines[i])
                    acc2, recall2, precision2, f12, auc2 = recognize(lines[i + 1])
                    acc3, recall3, precision3, f13, auc3 = recognize(lines[i + 2])
                    acc4, recall4, precision4, f14, auc4 = recognize(lines[i + 3])
                    acc5, recall5, precision5, f15, auc5 = recognize(lines[i + 4])
                    acc, recall, precision, f1, auc = round(sum([acc1, acc2, acc3, acc4, acc5]) * 100 / 5, 2), round(
                        sum(
                            [recall1, recall2, recall3, recall4, recall5, ]) * 100 / 5, 2), round(sum(
                        [precision1, precision2, precision3, precision4, precision5]) * 100 / 5, 2), round(sum(
                        [f11, f12, f13, f14, f15]) * 100 / 5, 2), round(sum([auc1, auc2, auc3, auc4, auc5]) * 100 / 5,
                                                                        2)
                    i += 5
                else:
                    acc, recall, precision, f1, auc = recognize(lines[i])
                    acc, recall, precision, f1, auc = round(100 * acc, 2), round(100 * recall, 2), round(
                        100 * precision, 2), round(100 * f1, 2), round(100 * auc, 2)
                    i += 1

                results.append([acc, precision, recall, f1, auc])
                num += 1
            else:
                i += 1
    f.close()
    return results


def write_to_excel(header, data,
                   output="/Users/tom/Downloads/learning-program-representation-master/data/custom/statistics/result.xlsx"):
    xls = openpyxl.Workbook()
    sheet = xls.create_sheet("Result", 0)
    sheet.append(["", "accuracy", "precision", "recall", "f1", "auc"])
    for i in range(len(header)):
        sheet.append([header[i]] + data[i])
    xls.save(output)


if __name__ == '__main__':
    write_to_excel(data=get_data(k_fold=rq5_2_kfold,
                                 root="/Users/tom/Downloads/learning-program-representation-master/data/custom/statistics/rq5/graph.txt"),
                   output="/Users/tom/Downloads/learning-program-representation-master/data/custom/statistics/rq5/rq5_2.xlsx",
                   header=rq5_2_header)
