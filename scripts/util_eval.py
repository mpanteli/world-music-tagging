# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:33:23 2017

@author: mariapanteli
"""
import numpy as np
from sklearn.metrics import roc_curve, auc

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def evaluate_predictions(y_test, y_score, tags=None):
    n_classes = y_test.shape[1]
    if tags is None:
        tags = np.arange(n_classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        tag = tags[i]
        fpr[tag], tpr[tag], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[tag] = auc(fpr[tag], tpr[tag])    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])    
    return fpr, tpr, roc_auc


def tag_precision(true_binary, pred_binary, tags, K=10):
    n_items = len(true_binary)
    precision_list = []
    true_y_list = []
    for i in range(n_items):
        #print true_binary[i, :], pred_binary[i, :]
        true_y = tags[np.where(true_binary[i, :]==1)[0]]
        if len(true_y)==0:
            # no tags found for this sample
            continue
        top_K_idx = np.argsort(pred_binary[i, :])[-K:][::-1]
        pred_y = tags[top_K_idx]
        n_tags = len(true_y)  # ranges between 1-3
        pp = 0
        for yy in true_y:
            if yy in pred_y:
                pp += 1
        precision = np.float(pp) / np.float(n_tags) 
        true_y_list.append(true_y[0])
        precision_list.append(precision)
    true_y_list = np.array(true_y_list)
    precision_list = np.array(precision_list)
    return precision_list, true_y_list


def plot_auc(fpr, tpr, auc_score, output_fig=None):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.3f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if output_fig is not None:
        plt.savefig(output_fig, bbox_inches='tight')


def plot_multiple_auc(fpr_list, tpr_list, auc_score_list, label_list=None, output_fig=None):
    n_models = len(auc_score_list)
    if label_list is None:
        label_list = ['model_'+str(i+1) for i in range(n_models)]
    plt.figure()
    for i in range(n_models):
        fpr, tpr, auc_score = fpr_list[i], tpr_list[i], auc_score_list[i]
        label = label_list[i]
        plt.plot(fpr['micro'], tpr['micro'], lw=2, label='%s ROC curve (area = %0.3f)' % (label, auc_score['micro']))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if output_fig is not None:
        plt.savefig(output_fig, bbox_inches='tight')


def plot_auc_per_class(auc_score, labels=None, color='blue', plot_legend=False, title=None, output_fig=None):
    sort_inds = np.argsort(auc_score) 
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(auc_score)), auc_score[sort_inds], 0.3, color=color[sort_inds])
    plt.ylabel('AUC score')
    plt.xlabel('Tags')
    if labels is not None:
        plt.xticks(range(len(labels)), labels[sort_inds], rotation=90, fontsize=4.5, ha='center')
    if plot_legend:
        # manual legend
        legend_labels=['country', 'language', 'decade']
        line_c = matplotlib.lines.Line2D([0], [0], linestyle='-', color='red')
        line_l = matplotlib.lines.Line2D([0], [0], linestyle='-', color='blue')
        line_d = matplotlib.lines.Line2D([0], [0], linestyle='-', color='black')
        plt.legend([line_c, line_l, line_d], legend_labels)
    if title is not None:
        plt.title(title)
    if output_fig is not None:
        plt.savefig(output_fig, bbox_inches='tight')