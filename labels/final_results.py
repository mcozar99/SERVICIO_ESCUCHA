import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt
from pathlib import Path
import os
from config import model_label as model, corpus, percent, n_samples, relabel
from BERTclassifier import getTrueLabels
from UliPlot.XLSX import auto_adjust_xlsx_column_width
from subprocess import call
from labels.escenario_multilabel.performance import ml_score, correct_ml_preds

mode = relabel.split('_')[0]
print(mode)

call('mkdir ./results/%s/evaluation/definitive'%model, shell=True)

label_set = list(dict.fromkeys(getTrueLabels(corpus)))
paths = sorted(Path('./results/%s/evaluation'%model).iterdir(), key=os.path.getmtime)
i=0
while i < len(paths):
    if mode not in paths[i].name or '%s_%s'%(n_samples, percent) not in paths[i].name:
        paths.remove(paths[i])
        i = 0
    i += 1

if mode not in paths[0].name:
    paths.pop(0)
print(paths)

df = pd.DataFrame(index = ['ACC', 'PREC', 'RECALL', 'F1'])

for path in paths:
    path = path.name
    data = pd.read_excel('./results/%s/evaluation/%s'%(model, path), header=None)
    if 'monolabel' in path:
        acc = data.loc[label_set.__len__() + 2].tolist()[2]
        prec = data.loc[label_set.__len__() + 3].tolist()[2]
        recall = data.loc[label_set.__len__() + 4].tolist()[2]
        f1 = data.loc[label_set.__len__() + 5].tolist()[2]
    if 'multilabel' in path:
        if 'kneighbors' not in path:
            acc = data.loc[0].tolist()[3]
            prec = data.loc[1].tolist()[3]
            recall = data.loc[2].tolist()[3]
            f1 = data.loc[3].tolist()[3]
        else:
            acc = data.loc[0].tolist()[2]
            prec = data.loc[1].tolist()[2]
            recall = data.loc[2].tolist()[2]
            f1 = data.loc[3].tolist()[2]
    path = path.replace('%s_'%mode, '')
    path = path.replace('_evaluation_%s_%s.xls'%(n_samples, percent), '')
    df[path] = [acc, prec, recall, f1]
    print(path)
    print('ACC: %s \t PREC: %s \t RECALL: %s \t F1: %s \n'%(acc, prec, recall, f1))
print('MULTILABEL CLUSTERING SCORE: %s'%ml_score)
print('PERCENT OF WELL PREDICTED MULTILABEL CLUSTERS: %s'%correct_ml_preds)
df['ML_SCORE'] = [ml_score, correct_ml_preds,0,0]
#df.to_excel('./results/%s/final_evaluation_%s_%s_%s_%s.xlsx'%(model, model, mode, n_samples, percent), sheet_name='Final Evaluation', float_format="%.5f", startrow=1, startcol=1, merge_cells=True, verbose=True)

# Export dataset to XLSX
#with pd.ExcelWriter('./results/%s/evaluation/definitive/final_evaluation_%s_%s_%s_%s.xlsx'%(model, model, mode, n_samples, percent)) as writer:
with pd.ExcelWriter('./TRAIN/final_evaluation_%s_%s_%s_%s.xlsx'%(model, mode, n_samples, percent)) as writer:
    df.to_excel(writer, sheet_name="Final Evaluation")
    auto_adjust_xlsx_column_width(df, writer, sheet_name="Final Evaluation", margin=0)
