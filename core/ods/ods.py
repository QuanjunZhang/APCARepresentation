import os
import sys

feature_csv = "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/features_patchsim.csv"
root = "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/name_as_coming_rules_patchsim"
tmp = "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/tmp"
out = "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/out"
coming_jar = "/Users/tom/Downloads/learning-program-representation-master/coming_jar/coming.jar"
test_csv = "/Users/tom/Downloads/learning-program-representation-master/data/custom/code/test.csv"
featureNum = 4495
header = ','.join(['id', ] + [str(x) for x in range(featureNum)])
"""使用coming.jar提取ods特征"""


def ods_main():
    f = open(feature_csv, 'w', encoding="utf8")
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    for d in os.listdir(root):
        os.system("cp -r {} {}".format(os.path.sep.join([root, d]), tmp + "/"))
        os.system(
            "java -classpath {} fr.inria.coming.main.ComingMain -input files -mode features -location {} -output {}".format(
                coming_jar, tmp, out))
        if os.path.exists(test_csv):
            with open(test_csv, 'r', encoding='utf8') as csv:
                f.write(list(csv.readlines())[1])
                f.flush()
            csv.close()
            os.system("rm -rf {}".format(test_csv))
        else:
            # give a default value
            f.write(','.join([d + "_patch_t"] + ['0'] * featureNum) + "\n")
            f.flush()

        os.system("rm -rf {}".format(out))
        os.system("rm -rf {}".format(tmp + os.path.sep + d))

    os.system("rm -rf {}".format(tmp))
    f.close()


"""正确/错误补丁数量统计"""


def ods_count():
    f = open(feature_csv, 'r', encoding="utf8")
    correct = 0
    overfit = 0
    for line in f.readlines():
        if "INCORRECT" in line:
            overfit += 1
        else:
            correct += 1
    f.close()
    print(correct, overfit)


if __name__ == '__main__':
    ods_main()
    ods_count()
