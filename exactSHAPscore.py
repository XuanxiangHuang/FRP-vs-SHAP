#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Compute exact SHAP-score using Barcelo's algorithm (under uniform distribution)
#   Author: Xuanxiang Huang
#
################################################################################
import sys
import pandas as pd
import numpy as np
from xddnnf.xpddnnf import XpdDnnf
from SHAPscore import SHAPdDNNF
################################################################################

if __name__ == '__main__':
    # string to bytes
    args = sys.argv[1:]
    if len(args) >= 1 and args[0] == '-bench':
        bench_name = args[1]
        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()
        for item in name_list:
            name = item.strip()
            print(f"################## {name} ##################")
            ################## read d-DNNF ##################
            xpddnnf = XpdDnnf.from_file(f'examples/{name}/ddnnf_2_fanin/{name}.dnnf', verb=0)
            xpddnnf.parse_feature_map(f'examples/{name}/{name}.map')
            ################## read d-DNNF ##################
            ################## read data ##################
            df_X = pd.read_csv(f"samples/{name}/all_points/train.csv")
            feature_names = list(df_X.columns)
            ################## read data ##################

            ################## compute SHAP-score ##################
            preds = xpddnnf.predict(df_X)
            prior_distrubution = [0.5] * xpddnnf.nf
            shapddnnf = SHAPdDNNF(prior_distrubution)
            scores = []
            for idx, line in enumerate(df_X.to_numpy()):
                xpddnnf.parse_instance(list(line))
                feats_score = [None] * xpddnnf.nf
                for feat in range(xpddnnf.nf):
                    feats_score[feat] = shapddnnf.algo1(xpddnnf, feat)
                scores.append(feats_score)
            exact_shap_scores = np.array(scores)
            abs_shap_scores = np.abs(exact_shap_scores)
            np.savetxt(f"scores/all_points/barcelo/{name}.csv", abs_shap_scores, delimiter=",", header=",".join(feature_names))
