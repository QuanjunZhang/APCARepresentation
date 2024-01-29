"""
收集所有patch的信息，包括项目名称、bugid、是否正确、patch文件位置、修复工具名称
"""
import json
import os

import tqdm

root = "/Users/tom/Downloads/learning-program-representation-master/data/custom/raw/new/Total_final"
name = "Total_final"
excluded_list = "/Users/tom/Downloads/learning-program-representation-master/data/custom/raw/new/filtered_list.txt"
ods_csv = "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/features.csv"


# Math,34,CORRECT,./Total_final/correct/HDRepair/Math/34/patch1.patch,HDRepair
# 从原始的patch数据集中收集patch信息
def extract():
    patch_list = []
    for is_correct in os.listdir(root):
        if is_correct.endswith(".DS_Store"):
            continue
        path1 = os.path.sep.join([root, is_correct])
        for tool_name in os.listdir(path1):
            if tool_name.endswith(".DS_Store"):
                continue
            path2 = os.path.sep.join([path1, tool_name])
            for proj in os.listdir(path2):
                if proj.endswith(".DS_Store"):
                    continue
                path3 = os.path.sep.join([path2, proj])
                for bug_id in os.listdir(path3):
                    if bug_id.endswith(".DS_Store"):
                        continue
                    path4 = os.path.sep.join([path3, bug_id])
                    for patch_name in os.listdir(path4):
                        if patch_name.endswith(".DS_Store"):
                            continue
                        patch_path = os.path.sep.join([".", name, is_correct, tool_name, proj, bug_id, patch_name])
                        patch_list.append(
                            "{},{},{},{},{}".format(proj, bug_id, is_correct.upper(), patch_path, tool_name))
    return patch_list


# 去除不能被cache脚本还原的patch
def exclude_absent(patches_all):
    with open(excluded_list, 'r', encoding='utf8') as f:
        patches_excluded = set([x.replace("\n", "") for x in f.readlines()])
    patches_all = set(patches_all)
    return list(patches_all - patches_excluded)


# 去除不能被ods利用的patch
def exclude_ods(patches_all):
    patch_files = []
    with open(ods_csv, 'r', encoding="utf8") as csv:
        for line in csv.readlines():
            # Closure_133_TBar_INCORRECT_patch2_patch_t
            proj, bug_id, tool_name, is_correct, patch_name, _, _ = line[:line.index(',')].split("_")
            is_correct = is_correct.lower()
            patch_files.append(
                os.path.sep.join([".", name, is_correct, tool_name, proj, bug_id, patch_name + ".patch"]))
    csv.close()
    patches_filtered = []
    for p in patches_all:
        if p.split(",")[3] in patch_files:
            patches_filtered.append(p)
    return patches_filtered


def write_to_whitelist(patches, where):
    dst_file = open(where, 'w', encoding="utf8")
    dst_file.writelines([patches[i] + ("\n" if i != len(patches) - 1 else "") for i in range(len(patches))])
    dst_file.close()


def read_whitelist(path):
    f = open(path, 'r', encoding="utf8")
    patches = []
    for line in f.readlines():
        # Math,34,CORRECT,./Total_final/correct/HDRepair/Math/34/patch2.patch,HDRepair
        line_split = line.strip().split(",")
        patches.append({
            "project": line_split[0],
            "bug_id": line_split[1],
            "is_correct": line_split[2],
            "patch_tool": line_split[-1],
            "patch_name": line_split[3].split(os.path.sep)[-1].replace(".patch", ""),
            "relative_path": line_split[3],
        })
    return patches


patch_ids = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 36, 37, 38, 44, 45, 46, 47, 48, 49, 51, 53, 54, 55, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69,
             72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 88, 89, 90, 91, 92, 93, 150, 151, 152, 153, 154, 155,
             157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 180,
             181, 182, 183, 184, 185, 186, 187, 188, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 201, 202, 203,
             204, 205, 206, 207, 208, 209, 210, 'HDRepair1', 'HDRepair3', 'HDRepair4', 'HDRepair5', 'HDRepair6',
             'HDRepair7', 'HDRepair8', 'HDRepair9', 'HDRepair10']
patch_ids = ["Patch" + str(x) for x in patch_ids]


# 找出patchsim中的patch
def exclude_patchsim(
        whitelist="/Users/tom/Downloads/learning-program-representation-master/data/custom/raw/new/whitelist_v2.txt",
        patch_root="/Users/tom/Downloads/learning-program-representation-master/data/custom/raw/new",
        patchsim_root="/Users/tom/Downloads/learning-program-representation-master/data/custom/raw/patchsim"):
    wl = read_whitelist(whitelist)
    patchsim_patches = []
    reflection={}
    other_patches = []
    for p_info in wl:
        p_line = ','.join([p_info["project"], p_info["bug_id"], p_info["is_correct"], p_info["relative_path"],
                           p_info["patch_tool"], ])
        other_patches.append(p_line)

    for patchsim_patch in os.listdir(patchsim_root):
        if patchsim_patch not in patch_ids:
            continue
        found = False
        patchsim_patch = patchsim_root + os.path.sep + patchsim_patch

        for p_info in wl:
            p_line = ','.join([p_info["project"], p_info["bug_id"], p_info["is_correct"], p_info["relative_path"],
                               p_info["patch_tool"], ])
            patch_path = patch_root + os.path.sep + p_info["relative_path"]
            if same_file(patch_path, patchsim_patch):
                if p_line not in patchsim_patches:
                    found = True
                    patchsim_patches.append(p_line)
                    reflection[p_line]=patchsim_patch
                    break
                else:
                    print("duplicate patches {}-{}-{}".format(patchsim_patch,reflection[p_line],p_line))

        # if not found:
        #     print(patchsim_patch)

    return patchsim_patches, list(set(other_patches)-set(patchsim_patches))


def same_file(file1, file2):
    f1 = open(file1, 'r')
    f2 = open(file2, 'r')
    return remove_redundant(f1.read()) == remove_redundant(f2.read())


def remove_redundant(s):
    ss = s.split("\n")
    sss = []
    for t in ss:
        if t.startswith("---") or t.startswith("+++") or t.startswith("@@"):
            continue
        if t.startswith("diff"):
            continue
        sss.append(t)
    return '\n'.join(sss).replace(" ", "").replace("-", "").replace("+", "").replace("\n", "")


if __name__ == '__main__':
    # write_to_whitelist(exclude_ods(extract()),
    #                    "/Users/tom/Downloads/learning-program-representation-master/data/custom/raw/new/whitelist.txt")
    p1, p2 = exclude_patchsim()
    write_to_whitelist(p1,
                       "/Users/tom/Downloads/learning-program-representation-master/data/custom/raw/new/patchsim_whitelist.txt")
    write_to_whitelist(p2,
                       "/Users/tom/Downloads/learning-program-representation-master/data/custom/raw/new/patchsim_excluded_whitelist.txt")
