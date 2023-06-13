#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Using the LIME tool (under uniform distribution)
#   Author: Xuanxiang Huang
#
################################################################################
import sys
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
from xddnnf.xpddnnf import XpdDnnf

np.random.seed(73)
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

            ################## invoke LIME explainer ##################
            feat_idx = list(range(len(feature_names)))
            explainer = lime.lime_tabular.LimeTabularExplainer(training_data=df_X.to_numpy(), feature_names=feature_names,
                                                               categorical_features=feat_idx, categorical_names=None,
                                                               class_names=[0, 1],
                                                               discretize_continuous=False)
            all_values = []
            for pt in df_X.to_numpy():
                exp = explainer.explain_instance(pt, xpddnnf.predict_prob, num_features=len(feature_names))
                lime_values = sorted(list(exp.as_map().values())[0], key=lambda x: x[0])
                vals = [v for idx, v in lime_values]
                all_values.append(vals)

            header_line = ",".join(feature_names)
            header_line = header_line.lstrip("#")
            np.savetxt(f"lime_scores/all_points/{name}.csv", np.array(all_values), delimiter=",", header=header_line, comments="")
