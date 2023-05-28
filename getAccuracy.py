#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Get Accuracy of d-DNNF classifiers
#   Author: Xuanxiang Huang
#
################################################################################
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
from xddnnf.xpddnnf import XpdDnnf
################################################################################

# python3 XXX.py -bench pmlb_bool.txt
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 1 and args[0] == '-bench':
        bench_name = args[1]
        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()
        for item in name_list:
            name = item.strip()
            ################## read d-DNNF ##################
            xpddnnf = XpdDnnf.from_file(f'examples/{name}/ddnnf_2_fanin/{name}.dnnf', verb=1)
            xpddnnf.parse_feature_map(f'examples/{name}/{name}.map')
            ################## read d-DNNF ##################
            train_df = pd.read_csv(f"datasets/{name}/train.csv")
            train_xs = train_df.iloc[:, :-1].to_numpy()
            train_ys = train_df.iloc[:, -1].to_numpy()

            test_df = pd.read_csv(f"datasets/{name}/test.csv")
            test_xs = test_df.iloc[:, :-1].to_numpy()
            test_ys = test_df.iloc[:, -1].to_numpy()

            ##### prediction all given instances #####
            train_pred = xpddnnf.predict(train_xs)
            train_acc = accuracy_score(train_pred, train_ys)
            print(f"{name}, train acc: {round(train_acc, 3)}")
            test_pred = xpddnnf.predict(test_xs)
            test_acc = accuracy_score(test_pred, test_ys)
            print(f"{name}, test acc: {round(test_acc, 3)}")

    exit(0)
