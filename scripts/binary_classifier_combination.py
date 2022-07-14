from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawROCCurve, DrawDistributions, DrawBDTScoreDistributions, DrawFeatureImportance, DrawVarDistribution,DrawMultipleROCCurves
import argparse
import pickle as pkl
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score
from collections import OrderedDict
import seaborn as sns
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--channel', help= 'Channel to train BDT for', default='tt')
parser.add_argument('--year', help= 'Year to train BDT for', default='2018')
parser.add_argument('--load', help= 'Load dataframe from file',  action='store_true')
parser.add_argument('--stop', help= 'Stop before training',  action='store_true')
args = parser.parse_args()

variables = ["pt_1", "pt_2","pt_3","pt_4","dphi_12","dphi_13","dphi_14","dphi_23","dphi_24","dphi_34","dR_12","dR_13","dR_14","dR_23",
"dR_24","dR_34","mt_1","mt_2","mt_3","mt_4","mt_lep_12","mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34",
"mvis_12","mvis_13","mvis_14","mvis_23","mvis_24","mvis_34","q_1","q_2","q_3","q_4"]

abs_variables = ["dphi_12","dphi_13","dphi_14","dphi_23","dphi_24","dphi_34","dR_12","dR_13","dR_14","dR_23",
"dR_24","dR_34"]

channels = ["eett","emtt","ettt","mmtt","mttt","tttt"]
#channels = ["eett","tttt"]
if not (args.load and os.path.isfile("dataframes/binary_combination_{}.pkl".format(args.year))):
  print "<< Making dataframe >>"  

  dataframes = []
  for ch in channels:
     # Add signal dataframe
     sig_df = Dataframe()
     sig_df.LoadRootFilesFromJson("json_selection/{}_{}_sig.json".format(ch,args.year),variables)
     sig_df.NormaliseWeights()
     sig_df.dataframe.loc[:,"{}".format(ch)] = 1
     sig_df.dataframe.loc[:,"y"] = 1
     # Add background data
     bkg_df = Dataframe()
     bkg_df.LoadRootFilesFromJson("json_selection/{}_{}_bkg.json".format(ch,args.year),variables)
     bkg_df.NormaliseWeights()
     bkg_df.dataframe.loc[:,"{}".format(ch)] = 1
     bkg_df.dataframe.loc[:,"y"] = 0

     dataframes.append(sig_df.dataframe)
     dataframes.append(bkg_df.dataframe)
  
  df_total = pd.concat(dataframes,ignore_index=True,sort=False)
  df_total[channels] = df_total[channels].fillna(value=0)

  #print "Signal Dataframe"
  #print sig_df.dataframe.head()
  #print "Length =",len(sig_df.dataframe)
  #print "Weight Normalisation =",sig_df.dataframe.loc[:,"weights"].sum()

  for i in abs_variables:
     df_total[i] = df_total[i].abs()

  print "Total Dataframe"
  print df_total.head()
  print "Length =",len(df_total)

  df_total.to_pickle("dataframes/binary_combination_{}.pkl".format(args.year))
else:
  print "<< Loading in dataframe >>"
  df_total = pd.read_pickle("dataframes/binary_combination_{}.pkl".format(args.year))


train_dedicated = OrderedDict()
X_train_dedicated = OrderedDict()
wt_train_dedicated = OrderedDict()
y_train_dedicated = OrderedDict()

test_dedicated = OrderedDict()
X_test_dedicated = OrderedDict()
wt_test_dedicated = OrderedDict()
y_test_dedicated = OrderedDict()


for ch in channels:
  dedicated = df_total[(df_total["{}".format(ch)] != 0)]
  dedicated = dedicated.loc[:, (dedicated != 0).any(axis=0)]
  dedicated = dedicated.drop("{}".format(ch),axis=1)
  
  signal_only_df = dedicated[dedicated.y == 1]
  bkg_only_df = dedicated[dedicated.y == 0]
  variables_to_plot=["pt_1","pt_2","pt_3"]
  for var in variables_to_plot:  
    DrawDistributions(signal_only_df,bkg_only_df,var,[0,200],100,ch)
  
  train,test = train_test_split(dedicated,test_size=0.5, random_state=42)
  train_dedicated["{}".format(ch)] = train
  test_dedicated["{}".format(ch)] = test

  y_train = train.loc[:,"y"]
  wt_train = train.loc[:,"weights"]
  X_train = train.drop(["y","weights"],axis=1)
  y_train_dedicated["{}".format(ch)] = y_train
  wt_train_dedicated["{}".format(ch)] = wt_train
  X_train_dedicated["{}".format(ch)] = X_train

  y_test = test.loc[:,"y"]
  wt_test = test.loc[:,"weights"]
  X_test = test.drop(["y","weights"],axis=1)
  y_test_dedicated["{}".format(ch)] = y_test
  wt_test_dedicated["{}".format(ch)] = wt_test
  X_test_dedicated["{}".format(ch)] = X_test


# set up train and test separated dataframes
# channels = ["eett","emtt","ettt","mmtt","mttt","tttt"]
selection = "eett"
for i in ["Cross_Channel","Cross_Channel_eett","Cross_Channel_emtt","Cross_Channel_ettt","Cross_Channel_mmtt","Cross_Channel_mttt","Cross_Channel_tttt"]:
    train, test = train_test_split(df_total,test_size=0.5, random_state=42)
    train_dedicated["{}".format(i)] = train
    test_dedicated["{}".format(i)] = test

    y_train = train.loc[:,"y"]
    wt_train = train.loc[:,"weights"]
    X_train = train.drop(["y","weights"],axis=1)
    y_train_dedicated["{}".format(i)] = y_train
    wt_train_dedicated["{}".format(i)] = wt_train
    X_train_dedicated["{}".format(i)] = X_train

    if i == "Cross_Channel_eett":
        selection = "eett"
        test = test[test[selection] != 0]
    if i == "Cross_Channel_emtt":
        selection = "emtt"
        test = test[test[selection] != 0]
    if i == "Cross_Channel_ettt":
        selection = "ettt"
        test = test[test[selection] != 0]
    if i == "Cross_Channel_mmtt":
        selection = "mmtt"
        test = test[test[selection] != 0]
    if i == "Cross_Channel_mttt":
        selection = "mttt"
        test = test[test[selection] != 0]
    if i == "Cross_Channel_tttt":
        selection = "tttt"
        test = test[test[selection] != 0]        
    y_test = test.loc[:,"y"]
    wt_test = test.loc[:,"weights"]
    X_test = test.drop(["y","weights"],axis=1)
    y_test_dedicated["{}".format(i)] = y_test
    wt_test_dedicated["{}".format(i)] = wt_test
    X_test_dedicated["{}".format(i)] = X_test


if args.stop:
    sys.exit("Exit before training")

# Train BDT
print "<< Running training >>"

xgb_model = xgb.XGBClassifier(
                             learning_rate =0.2,
                             n_estimators=100,
                             max_depth=3,
                             min_child_weight=5,
                             subsample=0.9,
                             colsample_bytree=0.8,
                             reg_alpha = 1.1,
                             reg_lambda = 1.3
                             )

# xgb_model = xgb.XGBClassifier()

preds=OrderedDict()

for key in train_dedicated:
 
   xgb_model.fit(X_train_dedicated[key], y_train_dedicated[key], sample_weight=wt_train_dedicated[key])
   probs = xgb_model.predict_proba(X_test_dedicated[key])
   preds[key] = probs[:,1]
   DrawFeatureImportance(xgb_model,"gain","bc_feature_importance_gain_{}_HMP".format(key))
   #DrawFeatureImportance(xgb_model,"weight","bc_feature_importance_weight{}".format(key))
   
#DrawMultipleROCCurves(y_test_dedicated,preds,wt_test_dedicated,output="roc_curve_HighMassPoint")

print "<< Training finished >>"



