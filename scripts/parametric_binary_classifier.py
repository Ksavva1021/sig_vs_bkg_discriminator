from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawROCCurve, DrawBDTScoreDistributions, DrawFeatureImportance, DrawVarDistribution
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mttt')
parser.add_argument('--year', help= 'Year to train BDT for', default='2018')
parser.add_argument('--load', help= 'Load dataframe from file',  action='store_true')
parser.add_argument('--duplicate_bkg', help= 'Will duplicate the background events per mass point tested rather than randomise.',  action='store_true')
args = parser.parse_args()

# Set up variables to use for training

variables = [
             "pt_1", "pt_2","pt_3","pt_4",
             "fabs(dphi_12)","fabs(dphi_13)","fabs(dphi_14)","fabs(dphi_23)","fabs(dphi_24)","fabs(dphi_34)",
             "fabs(dR_12)","fabs(dR_13)","fabs(dR_14)","fabs(dR_23)","fabs(dR_24)","fabs(dR_34)",
             "mt_1","mt_2","mt_3","mt_4",
             "mt_lep_12","mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34",
             "mvis_12","mvis_13","mvis_14","mvis_23","mvis_24","mvis_34",
             "q_1","q_2","q_3","q_4",
             "pt_tt_12","pt_tt_13","pt_tt_14","pt_tt_23","pt_tt_24","pt_tt_34",
             "n_jets","n_bjets"
             ]

#################### Make dataframe #################################

#phi_to_train_on = [100,300]
#A_to_train_on = [60,150]

phi_to_train_on = [100,200,300]
A_to_train_on = [60,100,150]

phi_to_test_on = [100,200,300]
A_to_test_on = [60,100,150]

#phi_to_train_on = [200]
#A_to_train_on = [100]

#phi_to_test_on = [200]
#A_to_test_on = [100]


if not (args.load and os.path.isfile("dataframes/{}_{}.pkl".format(args.channel,args.year))):
  print "<< Making dataframe >>"  

  # Add signal dataframes

  sig_df = {}
  if not args.duplicate_bkg: bkg_df = {}

  for mphi in list(set(phi_to_train_on+phi_to_test_on)):
    for mA in list(set(A_to_train_on+A_to_test_on)):
      filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars() 
      sig_df[filename] = Dataframe()
      sig_df[filename].LoadRootFilesFromJson("json_selection/{}_{}_sig_all.json".format(args.channel,args.year),variables,specific_file=filename)
      sig_df[filename].NormaliseWeights()
      sig_df[filename].dataframe.loc[:,"y"] = 1
      sig_df[filename].dataframe.loc[:,"mphi"] = mphi
      sig_df[filename].dataframe.loc[:,"mA"] = mA
      if mphi in phi_to_train_on and mA in A_to_train_on:
        sig_df[filename].dataframe.loc[:,"train"] = np.random.choice([0,1], sig_df[filename].dataframe.shape[0])
        sig_df[filename].dataframe.loc[:,"weights"] = sig_df[filename].dataframe.loc[:,"weights"]/(len(phi_to_train_on)*len(A_to_train_on))
      else:
        sig_df[filename].dataframe.loc[:,"train"] = 0
      print "Signal %(filename)s Dataframe" % vars()
      print sig_df[filename].dataframe.head()
      print "Length =",len(sig_df[filename].dataframe)
      print "Weight Normalisation =",sig_df[filename].dataframe.loc[:,"weights"].sum()

  # Add background dataframes
  if args.duplicate_bkg:
    bkg_df = {}
    for mphi in phi_to_train_on:
      for mA in A_to_train_on:
        filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
        bkg_df[filename] = Dataframe()
        bkg_df[filename].LoadRootFilesFromJson("json_selection/{}_{}_bkg.json".format(args.channel,args.year),variables)
        bkg_df[filename].NormaliseWeights()
        bkg_df[filename].dataframe.loc[:,"y"] = 0
        bkg_df[filename].dataframe.loc[:,"mphi"] = mphi
        bkg_df[filename].dataframe.loc[:,"mA"] = mA
        bkg_df[filename].dataframe.loc[:,"weights"] = bkg_df[filename].dataframe.loc[:,"weights"]/(len(phi_to_train_on)*len(A_to_train_on))
        bkg_df[filename].dataframe.loc[:,"train"] = np.random.choice([0,1], bkg_df[filename].dataframe.shape[0])
        print "Background %(filename)s Dataframe" % vars()
        print bkg_df[filename].dataframe.head()
        print "Length =",len(bkg_df[filename].dataframe)
        print "Weight Normalisation =",bkg_df[filename].dataframe.loc[:,"weights"].sum()  
  else:
    bkg_df = Dataframe()
    bkg_df.LoadRootFilesFromJson("json_selection/{}_{}_bkg.json".format(args.channel,args.year),variables)
    bkg_df.NormaliseWeights()
    bkg_df.dataframe.loc[:,"y"] = 0
    bkg_df.dataframe.loc[:,"mphi"] = np.random.choice(phi_to_train_on, bkg_df.dataframe.shape[0])
    bkg_df.dataframe.loc[:,"mA"] = np.random.choice(A_to_train_on, bkg_df.dataframe.shape[0])
    bkg_df.dataframe.loc[:,"train"] = np.random.choice([0,1], bkg_df.dataframe.shape[0])
    print "Background Dataframe"
    print bkg_df.dataframe.head()
    print "Length =",len(bkg_df.dataframe)
    print "Weight Normalisation =",bkg_df.dataframe.loc[:,"weights"].sum()
  
  # Combine dataframes
  combine_list = []
  for mphi in phi_to_train_on:
    for mA in A_to_train_on:
      filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
      combine_list.append(sig_df[filename].dataframe)
      if args.duplicate_bkg: combine_list.append(bkg_df[filename].dataframe)
  if not args.duplicate_bkg: combine_list.append(bkg_df.dataframe)
  df_total = pd.concat(combine_list,ignore_index=True, sort=False)
  print "Total Dataframe"
  print df_total.head()
  print "Length =",len(df_total)
  
  df_total.to_pickle("dataframes/binary_{}_{}.pkl".format(args.channel,args.year))
else:
  print "<< Loading in dataframe >>"
  df_total = pd.read_pickle("dataframes/binary_{}_{}.pkl".format(args.channel,args.year))

#DrawVarDistribution(df_total,2,'n_jets',0,10,"bc_n_jets_distribution",bin_edges=[0,1,2,3,4,5,6,7,8,9,10])

# Set up train and test separated dataframes
train = df_total.loc[(df_total.loc[:,'train']==1)]
y_train = train.loc[:,"y"]
wt_train = train.loc[:,"weights"]
X_train = train.drop(["y","weights","train"],axis=1)

#################### Train BDT #################################
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

xgb_model.fit(X_train, y_train, sample_weight=wt_train)

pkl.dump(xgb_model,open("BDTs/binary_{}_{}.pkl".format(args.channel,args.year), "wb"))

print "<< Training finished >>"

#################### Test BDT #################################

# ROC curve

for mphi in phi_to_test_on:
  for mA in A_to_test_on:
    bkg_test = Dataframe()
    sig_test = Dataframe()
    filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
    print "Scoring",filename
    sig_test.dataframe =  sig_df[filename].dataframe.loc[(sig_df[filename].dataframe.loc[:,'train']==0)]
    sig_test.NormaliseWeights()

    if args.duplicate_bkg:
      bkg_test.dataframe = bkg_df["phi{}A{}To4Tau".format(phi_to_train_on[0],A_to_train_on[0])].dataframe.loc[(bkg_df["phi{}A{}To4Tau".format(phi_to_train_on[0],A_to_train_on[0])].dataframe.loc[:,'train']==0)]
    else:
      bkg_test.dataframe = bkg_df.dataframe.loc[(bkg_df.dataframe.loc[:,"train"]==0)]
    bkg_test.dataframe.loc[:,"mphi"] = mphi
    bkg_test.dataframe.loc[:,"mA"] = mA
    bkg_test.NormaliseWeights()

    #print "Signal Test %(filename)s Dataframe" % vars()
    #print sig_test.dataframe.head()
    #print "Length =",len(sig_test.dataframe)
    #print "Weight Normalisation =",sig_test.dataframe.loc[:,"weights"].sum()

    #print "Background Test %(filename)s Dataframe" % vars()
    #print bkg_test.dataframe.head()
    #print "Length =",len(bkg_test.dataframe)
    #print "Weight Normalisation =",bkg_test.dataframe.loc[:,"weights"].sum()

    test = pd.concat([sig_test.dataframe,bkg_test.dataframe],ignore_index=True, sort=False)
    y_test = test.loc[:,"y"]
    wt_test = test.loc[:,"weights"]
    X_test = test.drop(["y","weights","train"],axis=1)
    probs = xgb_model.predict_proba(X_test)
    preds = probs[:,1]
    DrawROCCurve(y_test,preds,wt_test,"roc_curve_phi%(mphi)iA%(mA)i" % vars())
    print ""

## BDT scores
#t0 = test.loc[(test.loc[:,"y"]==0)]
#t1 = test.loc[(test.loc[:,"y"]==1)]
#wt0 = t0.loc[:,"weights"]
#wt1 = t1.loc[:,"weights"]
#xt0 = t0.drop(["y","weights"],axis=1)
#xt1 = t1.drop(["y","weights"],axis=1)
#probs0 = xgb_model.predict_proba(xt0)
#probs1 = xgb_model.predict_proba(xt1)
#preds0 = probs0[:,1]
#preds1 = probs1[:,1]
#
#DrawBDTScoreDistributions({"background":{"preds":preds0,"weights":wt0},"signal":{"preds":preds1,"weights":wt1}})
#
## Feature importance
#
#DrawFeatureImportance(xgb_model,"gain","bc_feature_importance_gain")
#DrawFeatureImportance(xgb_model,"weight","bc_feature_importance_weight")
