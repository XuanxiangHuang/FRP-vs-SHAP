#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   The maximum LIME score of the irrelevant feature and the minimum LIME score of the relevant feature.
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


def plot_max_irr_min_rel(data, col_names, fig_title, filename):
    df = pd.DataFrame(data, columns=col_names)
    ax1 = df.plot.scatter(x=col_names[0], y=col_names[1], color="#E69F00")
    ax2 = df.plot.scatter(x=col_names[0], y=col_names[2], color="#56B4E9", ax=ax1)
    ax2.set_xlabel("Instances")
    ax2.set_ylabel("Lime values")
    plt.title(fig_title)
    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()


def plot_irr_rel_summary(data, len_X, fig_title, filename):
    plt.rcdefaults()
    plt.bar(len_X, data, align='center', color=["#CC79A7", "#009E73"])
    plt.annotate(f'{data[0]}', xy=(0, data[0]), ha='center', va='bottom')
    plt.annotate(f'{data[1]}', xy=(1, data[1]), ha='center', va='bottom')
    plt.ylabel('#Instances')
    plt.xlabel('Lime values')
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
            max_irr_and_min_rel = []
            max_ir_lt_min_r = 0
            max_ir_ge_min_r = 0
            diff_max_ir_min_r = []
            column_names = ["#Instance", "Max Scores of IR-Feat", "Min Scores of R-Feat"]
            count = 0
            for idx, line in enumerate(df_X.to_numpy()):
                xpddnnf.parse_instance(list(line))
                feat_cnts = xpddnnf.nf * [0]
                pred = xpddnnf.get_prediction()
                axps, cxps = xpddnnf.enum_exps()
                for axp in axps:
                    for feat in axp:
                        assert feat < xpddnnf.nf
                        feat_cnts[feat] += 1
                scores_irr = [b_score[idx, j] for j in range(xpddnnf.nf) if feat_cnts[j] == 0]
                scores_rel = [b_score[idx, j] for j in range(xpddnnf.nf) if feat_cnts[j] != 0]
                if len(scores_irr) and len(scores_rel):
                    if abs(max(scores_irr)) >= abs(min(scores_rel)):
                        max_irr_and_min_rel.append([count, max(scores_irr), min(scores_rel)])
                        diff_max_ir_min_r.append([count, max(scores_irr) - min(scores_rel)])
                        max_ir_ge_min_r += 1
                        count += 1
                    else:
                        max_ir_lt_min_r += 1
                else:
                    max_ir_lt_min_r += 1
            plot_max_irr_min_rel(np.array(max_irr_and_min_rel), column_names, name,
                                 f"lime_scores/all_points/IrrRel/values/{name}.png")
            plot_irr_rel_summary(np.asarray([max_ir_ge_min_r, max_ir_lt_min_r]), ['Abnormal', 'Normal'], name,
                                 f"lime_scores/all_points/IrrRel/summary/{name}.png")
