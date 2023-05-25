#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Using the shap-tools (under uniform distribution)
#   Author: Xuanxiang Huang
#
################################################################################
import sys
import pandas as pd
import numpy as np
import shap
from xddnnf.xpddnnf import XpdDnnf
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

            ################## invoke SHAP explainer ##################
            explainer = shap.Explainer(model=xpddnnf.predict, masker=df_X)  # it will use 'exact' algorithm
            approx_shap_values = explainer(df_X)
            abs_shap_values = np.abs(approx_shap_values.values)
            np.savetxt(f"scores/all_points/lundberg/{name}.csv", abs_shap_values, delimiter=",", header=",".join(feature_names))

