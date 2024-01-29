import os
import pickle

import pandas as pd

from core.whitelist import read_whitelist

"""
构建bug和fix代码对应token序列
"""

root = "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/methods"
dst_data_pkl = "/Users/tom/Downloads/learning-program-representation-master/data/custom/sequence/data.pkl"
found = set()
not_found = set()


def find_bugfix():
    result = []
    fid=0
    for patch_info in read_whitelist(
            path="/Users/tom/Downloads/learning-program-representation-master/data/custom/raw/new/whitelist_v2.txt"):
        buggy_path = os.path.sep.join([root, patch_info["is_correct"].upper(), patch_info["patch_tool"],
                                       patch_info["patch_tool"] + "_" + patch_info["project"] + "_" + patch_info[
                                           "bug_id"], "defects4j_" + patch_info["patch_name"], "buggy.java"])
        fix_path = os.path.sep.join([root, patch_info["is_correct"].upper(), patch_info["patch_tool"],
                                     patch_info["patch_tool"] + "_" + patch_info["project"] + "_" + patch_info[
                                         "bug_id"], "defects4j_" + patch_info["patch_name"], "fixed.java"])
        with open(buggy_path, 'r', encoding="utf8") as f:
            buggy_code = "".join(list(filter(lambda x: not x.strip().startswith("//"), list(f.readlines()))))
            print(buggy_code)
            f.close()
        with open(fix_path, 'r', encoding="utf8") as f:
            fixed_code = "".join(list(filter(lambda x: not x.strip().startswith("//"), list(f.readlines()))))
            print(fixed_code)
            f.close()
        result.append({'function1': buggy_code, 'function2': fixed_code,
                       'target': 0 if patch_info["is_correct"].upper() == "CORRECT" else 1,
                       'patch_info': patch_info,
                       'function_id':fid})
        fid+=1
    return result


def remove_context(buggy_code, fixed_code):
    buggy_code = buggy_code.strip().split(" ")
    fixed_code = fixed_code.strip().split(" ")
    length1 = len(buggy_code)
    length2 = len(fixed_code)
    offset_head = 0
    offset_tail = 0
    while offset_head < min(length1, length2):
        if buggy_code[offset_head] == fixed_code[offset_head]:
            offset_head += 1
        else:
            break
    while offset_tail < min(length1, length2):
        if buggy_code[length1 - 1 - offset_tail] == fixed_code[length2 - 1 - offset_tail]:
            offset_tail += 1
        else:
            break
    s1 = ""
    s2 = ""
    if offset_head + offset_tail < length1:
        s1 = ' '.join(buggy_code[offset_head:length1 - offset_tail]).strip()
    if offset_head + offset_tail < length2:
        s2 = ' '.join(fixed_code[offset_head:length2 - offset_tail]).strip()
    return s1, s2


# remove context
def remove_context_main():
    with open(dst_data_pkl, 'rb') as f:
        t = pd.read_pickle(f)
        for d in t:
            buggy = d["function1"]
            fixed = d["function2"]
            a, b = remove_context(buggy, fixed)
            d["function1"], d["function2"] = a, b
            print(a, b)
    f = open(dst_data_pkl, 'wb')
    pickle.dump(t, f, 2)
    f.close()


if __name__ == '__main__':
    data = find_bugfix()

    # save data
    f = open(dst_data_pkl, 'wb')
    pickle.dump(data, f, 2)
    f.close()

