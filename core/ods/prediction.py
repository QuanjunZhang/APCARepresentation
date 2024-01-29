predict_csv = "/Users/tom/Downloads/learning-program-representation-master/data/custom/methods-new/prediction.csv"

"根据ODS输出的prediction.csv计算各种metrics"


def ods_prediction():
    p_csv = open(predict_csv, 'r', encoding='utf8')
    TP, TN, FP, FN = 0, 0, 0, 0

    line_num = 0

    for line in p_csv.readlines():
        if line_num == 0:
            line_num += 1
            continue
        line = line.replace("\n", "")
        ll = line.split(",")
        if "INCORRECT" in ll[1]:
            if ll[2] == '1':
                TP += 1
            elif ll[2] == '0':
                FN += 1
        else:
            if ll[2] == '1':
                FP += 1
            elif ll[2] == '0':
                TN += 1
        line_num += 1
    p_csv.close()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return TP, TN, FP, FN, precision, recall, f1


if __name__ == '__main__':
    r = ods_prediction()
    print("TP={}, TN={}, FP={}, FN={}, Precision={}, Recall={}, F1={}".format(r[0], r[1], r[2], r[3], r[4], r[5], r[6]))


