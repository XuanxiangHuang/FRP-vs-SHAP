#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Plot the LIME scores of the disorder instances.
#   Disorder is defined as instances where the LIME score for irrelevant features (irr)
#   is greater than the LIME score for relevant features (rel).
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


def plot_disorder_insts(data, instance, pred, col_names, fig_title, filename, avg_val=None):
    df = pd.DataFrame(data, columns=col_names)
    ax1 = df.plot.scatter(x=col_names[0], y=col_names[1], color="#E69F00", s=50, marker='^')
    ax2 = df.plot.scatter(x=col_names[0], y=col_names[2], color="#56B4E9", s=50, ax=ax1)
    for i, (x, y1, y2) in enumerate(zip(df[col_names[0]], df[col_names[1]], df[col_names[2]])):
        if not np.isnan(y1):
            ax2.annotate(f"{y1:.3}", (x, y1))
        if not np.isnan(y2):
            ax2.annotate(f"{y2:.3}", (x, y2))

    ax2.set_xlabel(f"instance: {tuple(instance), pred}")
    ax2.set_ylabel("Lime scores")
    if avg_val is not None:
        ax2.axhline(avg_val, color='r', linestyle='--')

    plt.title(fig_title)
    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()


# python3 XXX.py -bench pmlb_bool.txt
if __name__ == '__main__':
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
            df_X = pd.read_csv(f"samples/{name}/all_points/train.csv")
            feature_names = list(df_X.columns)

            b_data = pd.read_csv(f"lime_scores/all_points/{name}.csv")
            b_score = b_data.to_numpy()
            n, m = df_X.shape
            col_names = ['x', 'IRR', 'REL']
            for idx, line in enumerate(df_X.to_numpy()):
                xpddnnf.parse_instance(list(line))
                feat_cnts = xpddnnf.nf * [0]
                pred = xpddnnf.get_prediction()
                univ = [True] * xpddnnf.nf
                avg_output = (xpddnnf.model_counting(univ) / (2 ** xpddnnf.nf))
                axps, cxps = xpddnnf.enum_exps()
                for axp in axps:
                    for feat in axp:
                        assert feat < xpddnnf.nf
                        feat_cnts[feat] += 1
                scores_irr = [b_score[idx, j] for j in range(xpddnnf.nf) if feat_cnts[j] == 0]
                scores_rel = [b_score[idx, j] for j in range(xpddnnf.nf) if feat_cnts[j] != 0]
                data_irr = [b_score[idx, j] if feat_cnts[j] == 0 else np.nan for j in range(xpddnnf.nf)]
                data_rel = [b_score[idx, j] if feat_cnts[j] != 0 else np.nan for j in range(xpddnnf.nf)]
                if len(scores_irr) and len(scores_rel):
                    if max([abs(x) for x in scores_irr]) >= min([abs(x) for x in scores_rel]):
                        arr = np.array([np.arange(len(feature_names)), data_irr, data_rel])
                        plot_disorder_insts(arr.transpose(), list(line), pred, col_names, name,
                                            f"lime_disorder_insts/{name}/{name}_{idx}", avg_output)
