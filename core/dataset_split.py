import functools
import os
import pickle
import random

import tqdm

from data.custom.code.whitelist import read_whitelist

featureNum = 4495
train_set_header = ','.join(
    ['id', 'label'] + [str(x) for x in range(featureNum)])  # id,label,feature1,feature2,...,featureN
test_set_header = ','.join(['id', ] + [str(x) for x in range(featureNum)])  # id,feature1,feature2,...,featureN


# 给csv格式的特征文件添加label,header
def ods_format_trainset(
        out="/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/ods_train.csv",
        feature_csv="/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/features_patchsim_excluded.csv"):
    f = open(feature_csv, 'r', encoding='utf8')
    out_f = open(out, 'w')
    out_f.write(train_set_header + "\n")

    for line in f.readlines():
        t = line.split(',')
        if "INCORRECT" in t[0]:
            t.insert(1, "1")
        else:
            t.insert(1, "0")
        out_f.write(','.join(t))
        out_f.flush()
    out_f.close()


# 给csv格式的特征文件添加header
def ods_format_testset(
        out="/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/ods_test.csv",
        feature_csv="/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/features_patchsim.csv"):
    f = open(feature_csv, 'r', encoding='utf8')
    out_f = open(out, 'w')
    out_f.write(test_set_header + "\n")

    for line in f.readlines():
        out_f.write(line)
        out_f.flush()
    out_f.close()


def random_dropout(np=0.4,
                   dataset="/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/ods_train.csv"):
    line_num = 0
    f = open(dataset, "r")
    header = None
    positives = []
    negatives = []
    for line in f.readlines():
        if line_num == 0:
            header = line
        else:
            if "INCORRECT" in line:
                # 过拟合补丁是正样本
                positives.append(line)
            else:
                negatives.append(line)
        line_num += 1
    if len(negatives) > len(positives) * np:
        negatives = random.sample(negatives, int(len(positives) * np))
    if len(negatives) < len(positives) * np:
        positives = random.sample(positives, int(len(negatives) / np))
    f.close()
    f = open(dataset, "w")
    f.write(header)
    for negative in negatives:
        f.write(negative)
    for positive in positives:
        f.write(positive)
    f.flush()
    f.close()


# 采用k比例系数分割数据集
def k_fold_cross_validation(k_split=5, root_dir=""):
    f = open(root_dir + os.path.sep + "data.pkl", 'rb')
    dataset = pickle.load(f)
    print(len(dataset))

    for k in range(k_split):
        correct = 0
        overfit = 0
        train_set = []
        val_set = []
        train_output = open(root_dir + os.path.sep + "train{}.pkl".format(k), 'wb')
        val_output = open(root_dir + os.path.sep + "val{}.pkl".format(k), 'wb')
        for line in dataset:
            if line["target"] == 1:
                overfit += 1
                if overfit % k_split == k:
                    val_set.append(line)
                else:
                    train_set.append(line)
            else:
                correct += 1
                if correct % k_split == k:
                    val_set.append(line)
                else:
                    train_set.append(line)
        pickle.dump(train_set, train_output)
        pickle.dump(val_set, val_output)


# 根据whitelist分割数据集
def whitelist_split(whitelist, source_root, target_root):
    patches = read_whitelist(whitelist)
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    for p_info in tqdm.tqdm(patches):
        relative_dir = os.path.sep.join([p_info["is_correct"].upper(), p_info["patch_tool"],
                                         p_info["patch_tool"] + "_" + p_info["project"] + "_" + p_info["bug_id"],
                                         "defects4j_" + p_info["patch_name"]])
        abs_dst_dir = os.path.sep.join([target_root, relative_dir])
        abs_source_dir = os.path.sep.join([source_root, relative_dir])
        if not os.path.exists(abs_dst_dir):
            os.makedirs(abs_dst_dir)
        bug_source_path = os.path.sep.join([abs_source_dir, "buggy.java"])
        fix_source_path = os.path.sep.join([abs_source_dir, "fixed.java"])
        bug_dst_path = os.path.sep.join([abs_dst_dir, "buggy.java"])
        fix_dst_path = os.path.sep.join([abs_dst_dir, "fixed.java"])
        os.system("cp {} {}".format(bug_source_path, bug_dst_path))
        os.system("cp {} {}".format(fix_source_path, fix_dst_path))


# 根据fid提取数据集公共部分
def extract_by_fid(datasets, outputs=None):
    new_dataset = [[] for _ in range(len(datasets))]
    if outputs is None:
        outputs = datasets

    fp = [open(dataset, 'rb') for dataset in datasets]
    fids = [set() for _ in range(len(datasets))]
    for i in range(len(fp)):
        f = fp[i]
        for data_line in pickle.load(f):
            fids[i].add(data_line["function_id"])
    common_fid = functools.reduce(set.intersection, fids)
    [f.close() for f in fp]
    fp = [open(dataset, 'rb') for dataset in datasets]
    print(len(common_fid))
    for i in range(len(fp)):
        f = fp[i]
        for data_line in pickle.load(f):
            if data_line["function_id"] in common_fid:
                new_dataset[i].append(data_line)
    [f.close() for f in fp]

    outputs_f = [open(output, 'wb') for output in outputs]
    [pickle.dump(new_dataset[i], outputs_f[i]) for i in range(len(new_dataset))]
    [f.close() for f in outputs_f]


if __name__ == '__main__':
    # whitelist_split("/Users/tom/Downloads/learning-program-representation-master/data/custom/raw/new/patchsim_excluded_whitelist.txt",
    #                 "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/methods",
    #                 "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/methods_patchsim_excluded")
    # whitelist_split(
    #     "/Users/tom/Downloads/learning-program-representation-master/data/custom/raw/new/patchsim_whitelist.txt",
    #     "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/methods",
    #     "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/methods_patchsim")

    # ods_format_trainset()
    # ods_format_testset()

    k_fold_cross_validation(
        root_dir="/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/pdg/no-context-preserved")

     # extract_by_fid(
     #    ["/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/cfg/context-preserved/data.pkl",
     #     "/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/ddg/context-preserved/data.pkl",
     #     "/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/cdg/context-preserved/data.pkl",
     #     "/Users/tom/Downloads/learning-program-representation-master/data/custom/tree/context-preserved/data.pkl",
     #     "/Users/tom/Downloads/learning-program-representation-master/data/custom/sequence/data.pkl",
     #     "/Users/tom/Downloads/learning-program-representation-master/data/custom/feature/fusion/data.pkl"],
     #    outputs=["/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/cfg/multimodal/data.pkl",
     #             "/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/ddg/multimodal/data.pkl",
     #             "/Users/tom/Downloads/learning-program-representation-master/data/custom/graph/cdg/multimodal/data.pkl",
     #             "/Users/tom/Downloads/learning-program-representation-master/data/custom/tree/multimodal/data.pkl",
     #             "/Users/tom/Downloads/learning-program-representation-master/data/custom/sequence/multimodal/data.pkl",
     #             "/Users/tom/Downloads/learning-program-representation-master/data/custom/feature/fusion/data.pkl"])

    # random_dropout()

    # 完全一样的Patch, PatchSim不重复的只有130个
    # Patch31 Patch30 Math,8,INCORRECT,./Total_final/incorrect/jGenProg/Math/8/patch2.patch,jGenProg
    # Patch23 Patch151 Lang,51,INCORRECT,./Total_final/incorrect/Nopol2017/Lang/51/patch1.patch,Nopol2017
    # Patch83  Patch82 Time,11,INCORRECT,./Total_final/incorrect/jGenProg/Time/11/patch1.patch,jGenProg
    # Patch65  Patch66 Math,82,INCORRECT,./Total_final/incorrect/jGenProg/Math/82/patch1.patch,jGenProg
    # Patch55 Patch170 Math,73,INCORRECT,./Total_final/incorrect/Nopol2017/Math/73/patch3.patch,Nopol2017
    # Patch181 Patch185 Time,7,INCORRECT,./Total_final/incorrect/Nopol2017/Time/7/patch1.patch,Nopol2017
    # PatchHDRepair7 Patch51 Math,70,CORRECT,./Total_final/correct/HDRepair/Math/70/patch2.patch,HDRepair
    # Patch45 Patch44 Math,50,CORRECT,./Total_final/correct/jGenProg/Math/50/patch1.patch,jGenProg
    # Patch90 Patch88 Chart,9,INCORRECT,./Total_final/incorrect/Nopol2017/Chart/9/patch1.patch,Nopol2017

    # 预测结果
    # 76,Math_8_jGenProg_INCORRECT_patch2_patch_t,1
    # 36,Lang_51_Nopol2017_INCORRECT_patch1_patch_t,1
    # 87,Time_11_jGenProg_INCORRECT_patch1_patch_t,0
    # 43,Math_82_jGenProg_INCORRECT_patch1_patch_t,1
    # 18,Math_73_Nopol2017_INCORRECT_patch3_patch_t,1
    # 2,Time_7_Nopol2017_INCORRECT_patch1_patch_t,0
    # 126,Math_70_HDRepair_CORRECT_patch2_patch_t,0
    # 37,Math_50_jGenProg_CORRECT_patch1_patch_t,0
    # 103,Chart_9_Nopol2017_INCORRECT_patch1_patch_t,1
    # TP=5 FP=2 TN=2 FN=0
    # TP=70 TN=17 FP=16 FN=36
