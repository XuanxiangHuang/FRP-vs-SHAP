#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Generate all points
#   Author: Xuanxiang Huang
#
################################################################################
import itertools
import os
import sys
import numpy as np
import pandas as pd
################################################################################

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '-bench':
        bench_name = args[1]

        with open(bench_name, 'r') as fp:
            datasets = fp.readlines()

        for ds in datasets:
            name = ds.strip()
            print(f"############ {name} ############")
            train_df = pd.read_csv(f"datasets/{name}/train.csv")
            train_df.drop(train_df.columns[len(train_df.columns) - 1], axis=1, inplace=True)
            feature_names = list(train_df.columns)
            X_train = train_df.to_numpy()
            nof_insts, nof_feats = X_train.shape
            print(f"#feats: {nof_feats}, #train insts: {nof_insts}, generate(all possible points): {2**nof_feats}")
            # since all features have domain [0, 1],
            # we generate cartesian product of m domains
            tmp = list(itertools.product([0, 1], repeat=nof_feats))
            samples = [list(ele) for ele in tmp]
            samples = np.array(samples, dtype=np.uint16)
            df = pd.DataFrame(samples, columns=feature_names)
            dir_name = f"samples/{name}/all_points"
            try:
                os.stat(dir_name)
            except:
                os.makedirs(dir_name)
            fname = f"samples/{name}/all_points/train.csv"
            df.to_csv(fname, index=False)
