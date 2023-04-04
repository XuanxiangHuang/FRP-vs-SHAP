#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Max exact SHAP-score not in AXp and min exact SHAP-score in AXp
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


def plot_max_not_in_min_in(data, col_names, fig_title, filename, score_type="lundberg"):
    df = pd.DataFrame(data, columns=col_names)
    ax1 = df.plot(kind='scatter', x=col_names[0], y=col_names[1], color="#E69F00")
    ax2 = df.plot(kind='scatter', x=col_names[0], y=col_names[2], color="#56B4E9", ax=ax1)
    ax2.set_xlabel("Instances")
    if score_type == "lundberg":
        ax2.set_ylabel("SHAP values")
    else:
        ax2.set_ylabel("Shapley values")
    plt.title(fig_title)
    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()


def plot_comp_irr_rel(data, len_X, fig_title, filename, score_type="lundberg"):
    plt.rcdefaults()
    plt.bar(len_X, data, align='center', color=["#CC79A7", "#009E73"])
    plt.annotate(f'{data[0]}', xy=(0, data[0]), ha='center', va='bottom')
    plt.annotate(f'{data[1]}', xy=(1, data[1]), ha='center', va='bottom')
    plt.ylabel('#Instances')
    if score_type == "lundberg":
        plt.xlabel('SHAP values')
    else:
        plt.xlabel('Shapley values')
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
            counter = 0
            n, m = df_X.shape
            # max shapley value of irrelevant feature and min shapley value of relevant feature
            max_not_in_and_min_in = []
            # max shapley value of irrelevant feature < min shapley value of relevant feature (perfect separation)
            max_ir_lt_min_r = 0
            # max shapley value of irrelevant feature >= min shapley value of relevant feature (ambiguous)
            max_ir_ge_min_r = 0
            # max IR - min R
            diff_max_ir_min_r = []
            column_names = ["#Instance", "Max Scores of IR-Feat", "Min Scores of R-Feat"]
            # find the maximal score of feature not involved in AXp
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
                scores_irr = []
                scores_rel = []
                for j in range(xpddnnf.nf):
                    if feat_cnts[j] == 0:
                        scores_irr.append(b_score[idx, j])
                    else:
                        scores_rel.append(b_score[idx, j])
                if len(scores_irr) and len(scores_rel):
                    if abs(max(scores_irr)) >= abs(min(scores_rel)):
                        max_not_in_and_min_in.append([count, max(scores_irr), min(scores_rel)])
                        diff_max_ir_min_r.append([count, max(scores_irr) - min(scores_rel)])
                        max_ir_ge_min_r += 1
                        count += 1
                    else:
                        max_ir_lt_min_r += 1
                else:
                    max_ir_lt_min_r += 1
            plot_max_not_in_min_in(np.array(max_not_in_and_min_in), column_names, name, f"scores/all_points/max_notin_min_in/{which_score}_svs_irr_rel/{name}.png", which_score)
            plot_comp_irr_rel(np.asarray([max_ir_ge_min_r, max_ir_lt_min_r]), ['Abnormal', 'Normal'], name, f"scores/all_points/max_notin_min_in/{which_score}_svs_diff/{name}.png", which_score)
