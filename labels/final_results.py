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
from config import model_label as model, corpus, percent, n_samples
from BERTclassifier import getTrueLabels
from UliPlot.XLSX import auto_adjust_xlsx_column_width

label_set = list(dict.fromkeys(getTrueLabels(corpus)))
paths = sorted(Path('./results/%s/evaluation'%model).iterdir(), key=os.path.getmtime)
for i in range(len(paths)):
    paths[i] = paths[i].name.replace('_evaluation.xls', '')

df = pd.DataFrame(index = ['ACC', 'PREC', 'RECALL', 'F1'], columns=paths)

for path in paths:
    print(path)
    data = pd.read_excel('./results/%s/evaluation/%s_evaluation.xls'%(model, path), header=None)
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
    df[path] = [acc, prec, recall, f1]
    print('ACC: %s \t PREC: %s \t RECALL: %s \t F1: %s \n'%(acc, prec, recall, f1))

#df.to_excel('./results/%s/final_evaluation.xlsx'%model, sheet_name='Final Evaluation', float_format="%.5f", startrow=1, startcol=1, merge_cells=True, verbose=True)


# Export dataset to XLSX
with pd.ExcelWriter('./results/%s/final_evaluation_%s_%s_%s.xlsx'%(model, model, n_samples, percent)) as writer:
    df.to_excel(writer, sheet_name="Final Evaluation")
    auto_adjust_xlsx_column_width(df, writer, sheet_name="Final Evaluation", margin=0)
