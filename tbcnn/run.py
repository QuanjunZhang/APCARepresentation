import math
import os

if __name__ == '__main__':
    tree_sizes = [30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    orders=[]
    for tsize in tree_sizes:
        batch_size = 10 * 1024 // tsize
        batch_size=int(pow(2,int(math.log2(batch_size))))
        orders.append("python data_formatter.py {} {} && python train.py {} {} > log{}.txt".format(tsize,batch_size, tsize,batch_size, tsize))
    os.system("&&".join(orders))
