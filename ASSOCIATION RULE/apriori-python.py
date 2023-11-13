#Ví dụ này chạy trên Python 2.7. Tải Python 2.7.0 về cài đặt sau đó tạo intepreter tương ứng

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apriori_python import apriori

def ReadData(filename):
    D = []
    with open("DataSet6.txt", "r") as f:
        for line in f:
            T = []
            for word in line.split():
                T.append(word)
            D.append(T)
    return D

store_data = ReadData('DataSet6.txt')
print(store_data)
freqItemSet, rules = apriori(store_data, minSup=0.3, minConf=0.3)
for x in freqItemSet:
    print('Freq: ', x, freqItemSet[x])
i = 0
for x in rules:
    i = i + 1
    print('Rules: ',i, x)