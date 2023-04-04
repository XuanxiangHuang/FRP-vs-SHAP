#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Plot exact SHAP-score of instance
#   Author: Xuanxiang Huang
#
################################################################################
import sys
import pandas as pd
import numpy as np
from xddnnf.xpddnnf import XpdDnnf
import matplotlib.pyplot as plt
################################################################################

# The palette with grey:
# "#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"


def plot_disorder_inst(data, instance, pred, col_names, fig_title, filename, score_type="lundberg"):
    df = pd.DataFrame(data, columns=col_names)
    ax1 = df.plot(kind='scatter', x=col_names[0], y=col_names[1], color="#E69F00", s=50, marker='^')
    ax2 = df.plot(kind='scatter', x=col_names[0], y=col_names[2], color="#56B4E9", s=50, ax=ax1)
    x = df[col_names[0]].to_numpy()
    y1 = df[col_names[1]].to_numpy()
    y2 = df[col_names[2]].to_numpy()
    for i in range(len(x)):
        if y1[i] != np.nan:
            ax2.annotate(f"{y1[i]:.3}", (x[i], y1[i]))
        if y2[i] != np.nan:
            ax2.annotate(f"{y2[i]:.3}", (x[i], y2[i]))

    ax2.set_xlabel(f"instance: {tuple(instance), pred}")
    if score_type == "lundberg":
        ax2.set_ylabel("SHAP values")
    else:
        ax2.set_ylabel("Shapley values")
    plt.title(fig_title)
    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()


# python3 XXX.py -bench pmlb_bool.txt
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == '-bench':
        bench_name = args[1]
        which_score = args[2]
        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()
        for item in name_list:
            name = item.strip()
            print(f"################## {name} ##################")
            ################## read d-DNNF ##################
            xpddnnf = XpdDnnf.from_file(f'examples/{name}/ddnnf_2_fanin/{name}.dnnf', verb=0)
            xpddnnf.parse_feature_map(f'examples/{name}/{name}.map')
            ################## read d-DNNF ##################
            df_X = pd.read_csv(f"samples/{name}/all_points/train.csv")
            feature_names = list(df_X.columns)

            b_data = pd.read_csv(f"scores/all_points/{which_score}/{name}.csv")
            b_score = b_data.to_numpy()
            n, m = df_X.shape
            col_names = ['x', 'IRR', 'REL']
            for idx, line in enumerate(df_X.to_numpy()):
                xpddnnf.parse_instance(list(line))
                feat_cnts = xpddnnf.nf * [0]
                pred = xpddnnf.get_prediction()
                axps, cxps = xpddnnf.enum_exps()
                for axp in axps:
                    for feat in axp:
                        assert feat < xpddnnf.nf
                        feat_cnts[feat] += 1
                scores_irr = []
                scores_rel = []
                data_irr = xpddnnf.nf * [np.nan]
                data_rel = xpddnnf.nf * [np.nan]
                for j in range(xpddnnf.nf):
                    if feat_cnts[j] == 0:
                        scores_irr.append(b_score[idx, j])
                        data_irr[j] = b_score[idx, j]
                    else:
                        scores_rel.append(b_score[idx, j])
                        data_rel[j] = b_score[idx, j]
                if len(scores_irr) and len(scores_rel):
                    if abs(max(scores_irr)) >= abs(min(scores_rel)):
                        arr = np.array([np.arange(len(feature_names)), data_irr, data_rel])
                        plot_disorder_inst(arr.transpose(), list(line), pred, col_names, name, f"disorder_inst/{which_score}/{name}/{name}_{idx}", which_score)
