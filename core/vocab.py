import pandas as pd

from tokenizer.code_tokenizer import CodeTokenizer
from tokenizer.vocab_dict import VocabDict


# for tree and graph
def build_vocab_for_tree_graph(dst_token_vocab_path, dst_node_vocab_path, data_path):
    tokenizer = CodeTokenizer(data=[])
    dst_token_vocab = VocabDict(
        file_name=dst_token_vocab_path,
        name="JavaTokenVocabDictionary")
    dst_node_vocab = VocabDict(
        file_name=dst_node_vocab_path,
        name="JavaTokenVocabDictionary")
    sents = []
    sents2 = []

    with open(data_path, 'rb') as f:
        t = pd.read_pickle(f)
        for s in t:
            for ss in s['jsgraph1']['node_features'].values():
                sents.append(tokenizer.tokenize(ss[1]))
                sents2.append(ss[0])
            for ss in s['jsgraph2']['node_features'].values():
                sents.append(tokenizer.tokenize(ss[1]))
                sents2.append(ss[0])

        f.close()

    for sent in sents:
        dst_token_vocab.add_sentence(str(sent))
        print(sent)
    for sent in sents2:
        dst_node_vocab.add_sentence(str(sent))
        print(sent)
    dst_token_vocab.save()
    dst_node_vocab.save()


# for sequence
def build_vocab_for_sequence(dst_token_vocab_path, data_path):
    tokenizer = CodeTokenizer(data=[])
    sents = []
    d = VocabDict(
        file_name=dst_token_vocab_path,
        name="JavaTokenVocabDictionary")

    with open(data_path, 'rb') as f:
        t = pd.read_pickle(f)
        for s in t:
            sents.append(tokenizer.tokenize(s['function1']))
            sents.append(tokenizer.tokenize(s['function2']))
        f.close()
    for sent in sents:
        d.add_sentence(str(sent))
        print(sent)
    d.save()


if __name__ == '__main__':
    build_vocab_for_sequence(
        dst_token_vocab_path="/Users/tom/Downloads/learning-program-representation-master/data/custom/feature/fusion/token_vocab_dict.pkl",
        data_path="/Users/tom/Downloads/learning-program-representation-master/data/custom/feature/fusion/data.pkl")
