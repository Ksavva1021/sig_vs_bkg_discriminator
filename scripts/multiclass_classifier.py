from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawROCCurve, DrawBDTScoreDistributions, DrawFeatureImportance, DrawConfusionMatrix
import argparse
import pickle as pkl
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='tt')
parser.add_argument('--year', help= 'Year to train BDT for', default='2018')
parser.add_argument('--load', help= 'Load dataframe from file',  action='store_true')
args = parser.parse_args()

# Set up variables to use for training
variables = ["mt_tot","pt_1","pt_2","met","m_vis","n_deepbjets","pt_tt","mt_lep","met_dphi_1","met_dphi_2","jet_pt_1/pt_1","jet_pt_2/pt_2"]

if not (args.load and os.path.isfile("dataframes/{}_{}.pkl".format(args.channel,args.year))):
  print "<< Making dataframe >>"  

  # Add interfence dataframe
  int_df = Dataframe()
  int_df.LoadRootFilesFromJson("json_selection/{}_{}_sig_interference.json".format(args.channel,args.year),variables)
  int_df.NormaliseWeights()
  int_df.dataframe.loc[:,"y"] = 2
  print "Interference Dataframe"
  print int_df.dataframe.head()
  print "Length =",len(int_df.dataframe)
  print "Weight Normalisation =",int_df.dataframe.loc[:,"weights"].sum()

  # Add signal dataframe
  sig_df = Dataframe()
  sig_df.LoadRootFilesFromJson("json_selection/{}_{}_sig.json".format(args.channel,args.year),variables)
  sig_df.NormaliseWeights()
  sig_df.dataframe.loc[:,"y"] = 1
  print "Signal Dataframe"
  print sig_df.dataframe.head()
  print "Length =",len(sig_df.dataframe)
  print "Weight Normalisation =",sig_df.dataframe.loc[:,"weights"].sum()
  
  # Add background data
  bkg_df = Dataframe()
  bkg_df.LoadRootFilesFromJson("json_selection/{}_{}_bkg.json".format(args.channel,args.year),variables)
  bkg_df.NormaliseWeights()
  bkg_df.dataframe.loc[:,"y"] = 0
  print "Background Dataframe"
  print bkg_df.dataframe.head()
  print "Length =",len(bkg_df.dataframe)
  print "Weight Normalisation =",bkg_df.dataframe.loc[:,"weights"].sum()
  
  # Combine dataframes
  df_total = pd.concat([bkg_df.dataframe,sig_df.dataframe,int_df.dataframe],ignore_index=True, sort=False)
  print "Total Dataframe"
  print df_total.head()
  print "Length =",len(df_total)
  
  df_total.to_pickle("dataframes/multiclass_{}_{}.pkl".format(args.channel,args.year))
else:
  print "<< Loading in dataframe >>"
  df_total = pd.read_pickle("dataframes/multiclass_{}_{}.pkl".format(args.channel,args.year))

# Set up train and test separated dataframes
train, test = train_test_split(df_total,test_size=0.5, random_state=42)

y_train = train.loc[:,"y"]
wt_train = train.loc[:,"weights"]
X_train = train.drop(["y","weights"],axis=1)

y_test = test.loc[:,"y"]
wt_test = test.loc[:,"weights"]
X_test = test.drop(["y","weights"],axis=1)

# Train BDT
print "<< Running training >>"

#xgb_model = xgb.XGBClassifier(
#                              learning_rate =0.1,
#                              n_estimators=1000,
#                              max_depth=5,
#                              min_child_weight=1,
#                              gamma=0,
#                              subsample=0.8,
#                              colsample_bytree=0.8,
#                              objective= 'binary:logistic',
#                              nthread=4,
#                              scale_pos_weight=1,
#                              seed=27
#                              )

xgb_model = xgb.XGBClassifier()

xgb_model.fit(X_train, y_train, sample_weight=wt_train)

pkl.dump(xgb_model,open("BDTs/multiclass_{}_{}.pkl".format(args.channel,args.year), "wb"))

print "<< Training finished >>"

# Test output

# BDT scores
t0 = test.loc[(test.loc[:,"y"]==0)]
t1 = test.loc[(test.loc[:,"y"]==1)]
t2 = test.loc[(test.loc[:,"y"]==2)]
wt0 = t0.loc[:,"weights"]
wt1 = t1.loc[:,"weights"]
wt2 = t2.loc[:,"weights"]
xt0 = t0.drop(["y","weights"],axis=1)
xt1 = t1.drop(["y","weights"],axis=1)
xt2 = t2.drop(["y","weights"],axis=1)
probs0 = xgb_model.predict_proba(xt0)
probs1 = xgb_model.predict_proba(xt1)
probs2 = xgb_model.predict_proba(xt2)
preds0 = probs0[:,0]
preds1 = probs1[:,1]
preds2 = probs2[:,2]

DrawBDTScoreDistributions({"background":{"preds":preds0,"weights":wt0},"signal":{"preds":preds1,"weights":wt1},"interference":{"preds":preds2,"weights":wt2}})

# Feature importance
DrawFeatureImportance(xgb_model)

# Confustion matrix
preds = xgb_model.predict(X_test)
probs = xgb_model.predict_proba(X_test)
DrawConfusionMatrix(y_test,preds,wt_test,["background","signal","interference"])

