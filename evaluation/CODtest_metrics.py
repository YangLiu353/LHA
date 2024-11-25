# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os

import cv2
from tqdm import tqdm

txtpath = '/home/ipal/Home/LY/run4/eval/'+'result4'
data_root = "/home/ipal/Home/LY/run4/eval/eval-data"
dataname = ['LSNet-1']
#dataname = ['CMINet', 'ATSA', 'BTS', 'CDNet', 'CoNet', 'DANet', 'DSA2F', 'EFNet', 'FRDT', 'HAINet', 'PGAR', 'S2MA', 'SSF', 'UCNet']
#methods = ['COME-E', 'COME-H']
#methodmasks = ['COME-E', 'COME-H']

#methods = ['SIP']
#methodmasks = ['SIP']

#methods = ['COME-E', 'COME-H', 'DES', 'DUT-RGBD', 'LFSD', 'NJU2K', 'NLPR', 'ReDWeb-S', 'SIP', 'STERE']
#methodmasks = ['COME-E', 'COME-H', 'DES', 'DUT-RGBD', 'LFSD', 'NJU2K', 'NLPR', 'ReDWeb-S', 'SIP', 'STERE']

#methods = ['NLPR', 'SIP', 'STERE', 'SSD']
#methodmasks = ['NLPR', 'SIP', 'STERE', 'SSD']
methods = ['CAMO','CHAMELEON', 'COD10K', 'NC4K']
methodmasks = ['CAMO','CHAMELEON', 'COD10K', 'NC4K']

#methods = ['NJU2K']
#methodmasks = ['NJU2K']
for name in dataname:
    for mthm in methodmasks:
        for mths in methods:
           if not mthm == mths:
              continue
           datasetname = mthm.split("_")[0]
           mask_root = os.path.join(data_root, "gt", mthm)
           pred_root = os.path.join(data_root, "preds", name, mths)
		
           from sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure
           FM = Fmeasure()
           WFM = WeightedFmeasure()
           SM = Smeasure()
           EM = Emeasure()
           MAE = MAE()
           mask_name_list = sorted(os.listdir(mask_root))
           for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
               mask_path = os.path.join(mask_root, mask_name)
               pred_path = os.path.join(pred_root, mask_name)
               mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
               pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            #    print ('NAME',mask_name)
               if not mask.shape == pred.shape:
                      rows, cols = mask.shape[:2]
                      pred = cv2.resize(pred,(cols,rows),interpolation=cv2.INTER_CUBIC)
               FM.step(pred=pred, gt=mask)
               #WFM.step(pred=pred, gt=mask)
               SM.step(pred=pred, gt=mask)
               EM.step(pred=pred, gt=mask)
               MAE.step(pred=pred, gt=mask)
		
           fm = FM.get_results()["fm"]
           #wfm = WFM.get_results()["wfm"]
           sm = SM.get_results()["sm"]
           em = EM.get_results()["em"]
           mae = MAE.get_results()["mae"]
		
           results = {
                      "dataset": name,
                      "Subclass": datasetname,
                      "Methods": mths,
                      "Smeasure": sm.round(3),
                      #"wFmeasure": wfm.round(3),
                      #"adpEm": em["adp"].round(3),
                      "meanEm": em["curve"].mean().round(3),
                      #"maxEm": em["curve"].max().round(3),
                      #"adpFm": fm["adp"].round(3),
                      "meanFm": fm["curve"].mean().round(3),
                      #"maxFm": fm["curve"].max().round(3),
                      "MAE": mae.round(3),
                     }
		
           print(results)
		
           with open(txtpath, 'a+') as fp:
                fp.write((str(results).replace('{', '')).replace('}', ''))
                fp.write('\n')
