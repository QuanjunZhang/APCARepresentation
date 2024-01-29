import os

"""
将还原后的bug fix代码整理为coming的目录格式
"""

# rules Math_70/BisectionSolver/Math_70_BisectionSolver_s.java Math_70/BisectionSolver/Math_70_BisectionSolver_t.java

def rename(root,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    count = 0
    for is_correct in os.listdir(root):
        if is_correct==".DS_Store":
            continue
        p0 = os.path.sep.join([root, is_correct])
        for tool_name in os.listdir(p0):
            if tool_name == ".DS_Store":
                continue
            p1 = os.path.sep.join([p0, tool_name])
            for p in os.listdir(p1):
                if p == ".DS_Store":
                    continue
                _, proj, bug_id = p.split("_")
                p2 = os.path.sep.join([p1, p])
                for pp in os.listdir(p2):
                    if pp == ".DS_Store":
                        continue
                    _, patch_name = pp.split("_")
                    p3 = os.path.sep.join([p2, pp])

                    if not os.path.exists(os.path.sep.join([p3, "buggy.java"])):
                        continue
                    count += 1
                    target_dir = os.path.sep.join(
                        [output_dir, proj + "_" + bug_id + "_" + tool_name + "_" + is_correct+"_"+patch_name, "patch"])
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    os.system("cp {} {}".format(os.path.sep.join([p3, "buggy.java"]), os.path.sep.join(
                        [target_dir, proj + "_" + bug_id + "_" + tool_name + "_" + is_correct+"_"+patch_name+ "_patch_s.java"])))
                    os.system("cp {} {}".format(os.path.sep.join([p3, "fixed.java"]), os.path.sep.join(
                        [target_dir,
                         proj + "_" + bug_id + "_" + tool_name + "_" + is_correct+"_"+patch_name+ "_patch_t.java"])))

    print(count)


if __name__ == '__main__':
    rename("/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/methods_patchsim",
           "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/name_as_coming_rules_patchsim")
    rename("/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/methods_patchsim_excluded",
           "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/name_as_coming_rules_patchsim_excluded")
# java -classpath ./coming_jar/coming.jar fr.inria.coming.main.ComingMain -input files -mode features -location ./pairsD4j -output ./out
