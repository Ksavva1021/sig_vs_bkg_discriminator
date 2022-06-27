from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawROCCurve, DrawBDTScoreDistributions, DrawFeatureImportance, DrawVarDistribution, DrawMultipleROCCurves
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mttt')
parser.add_argument('--year', help= 'Year to train BDT for', default='2018')
parser.add_argument('--verboscity', help= 'Changes how much is printed', default='0')
parser.add_argument('--load', help= 'Load dataframe from file',  action='store_true')
parser.add_argument('--duplicate_bkg', help= 'Will duplicate the background events per mass point tested rather than randomise.',  action='store_true')
parser.add_argument('--use_deeptau', help= 'Use the deeptau scores for training',  action='store_true')
parser.add_argument('--train_dedicated', help= 'Train dedicated MVA to test against parametric MVA.',  action='store_true')
parser.add_argument('--grid_search', help= 'Do grid search to find best hyperparameters',  action='store_true')
parser.add_argument('--grid_search_dedicated', help= 'Do grid search to find best hyperparameters for the dedicated bdt',  action='store_true')
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
             "n_jets","n_bjets",
             ]

if args.use_deeptau:
  variables += [
                "deepTauVsJets_iso_1","deepTauVsJets_iso_2","deepTauVsJets_iso_3","deepTauVsJets_iso_4",
                "deepTauVsEle_iso_1","deepTauVsEle_iso_2","deepTauVsEle_iso_3","deepTauVsEle_iso_4",
                "deepTauVsMu_iso_1","deepTauVsMu_iso_2","deepTauVsMu_iso_3","deepTauVsMu_iso_4",
                ]


#################### Make dataframe #################################

phi_to_train_on = [100,200,300]
A_to_train_on = [60,100,150]

phi_to_test_on = [100,200,300]
A_to_test_on = [60,100,150]


if not (args.load and os.path.isfile("dataframes/{}_{}.pkl".format(args.channel,args.year))):
  print "<< Making dataframe >>"  

  # Add signal dataframes

  sig_df = {}
  if not args.duplicate_bkg: bkg_df = {}

  for mphi in list(set(phi_to_train_on+phi_to_test_on)):
    for mA in list(set(A_to_train_on+A_to_test_on)):
      filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars() 
      sig_df[filename] = Dataframe()
      sig_df[filename].LoadRootFilesFromJson("json_selection/{}_{}_sig_all.json".format(args.channel,args.year),variables,specific_file=filename,quiet=True)
      sig_df[filename].dataframe.loc[:,"y"] = 1
      if args.train_dedicated:
        sig_df[filename+"_dedicated"] = sig_df[filename].Copy()
        sig_df[filename+"_dedicated"].TrainTestSplit()
        sig_df[filename+"_dedicated"].NormaliseWeights(train_frac=1.0,test_frac=1.0)
      if mphi in phi_to_train_on and mA in A_to_train_on:
        sig_df[filename].TrainTestSplit()
      else:
        sig_df[filename].dataframe.loc[:,"train"] = 0
      sig_df[filename].NormaliseWeights(test_frac=1.0,train_frac=1.0/(len(phi_to_train_on)*len(A_to_train_on)))
      sig_df[filename].dataframe.loc[:,"mphi"] = mphi
      sig_df[filename].dataframe.loc[:,"mA"] = mA

  # Add background dataframes
  if args.duplicate_bkg:
    bkg_df = {}
    for mphi in phi_to_train_on:
      for mA in A_to_train_on:
        filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
        bkg_df[filename] = Dataframe()
        bkg_df[filename].LoadRootFilesFromJson("json_selection/{}_{}_bkg.json".format(args.channel,args.year),variables,quiet=True)
        bkg_df[filename].TrainTestSplit()
        bkg_df[filename].dataframe.loc[:,"y"] = 0
        if mphi == phi_to_train_on[0] and mA == A_to_train_on[0] and args.train_dedicated:
          bkg_df_dedicated = bkg_df[filename].Copy()
          bkg_df_dedicated.NormaliseWeights(train_frac=1.0,test_frac=1.0)
        bkg_df[filename].NormaliseWeights(test_frac=1.0,train_frac=1.0/(len(phi_to_train_on)*len(A_to_train_on)))
        bkg_df[filename].dataframe.loc[:,"mphi"] = mphi
        bkg_df[filename].dataframe.loc[:,"mA"] = mA
        
  else:
    bkg_df = Dataframe()
    bkg_df.LoadRootFilesFromJson("json_selection/{}_{}_bkg.json".format(args.channel,args.year),variables,quiet=True)
    bkg_df.TrainTestSplit()
    bkg_df.NormaliseWeights(test_frac=1.0,train_frac=1.0)
    bkg_df.dataframe.loc[:,"y"] = 0
    if args.train_dedicated:
      bkg_df_dedicated = bkg_df.Copy()
    bkg_df.dataframe.loc[:,"mphi"] = np.random.choice(phi_to_train_on, bkg_df.dataframe.shape[0])
    bkg_df.dataframe.loc[:,"mA"] = np.random.choice(A_to_train_on, bkg_df.dataframe.shape[0])

  
  # Combine dataframes
  combine_list = []
  for mphi in phi_to_train_on:
    for mA in A_to_train_on:
      filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
      combine_list.append(sig_df[filename].dataframe)
      if args.duplicate_bkg: combine_list.append(bkg_df[filename].dataframe)
  if not args.duplicate_bkg: combine_list.append(bkg_df.dataframe)
  df_total = pd.concat(combine_list,ignore_index=True, sort=False)
  
  df_total.to_pickle("dataframes/binary_{}_{}.pkl".format(args.channel,args.year))
else:
  print "<< Loading in dataframe >>"
  df_total = pd.read_pickle("dataframes/binary_{}_{}.pkl".format(args.channel,args.year))

#DrawVarDistribution(df_total,2,'n_jets',0,10,"bc_n_jets_distribution",bin_edges=[0,1,2,3,4,5,6,7,8,9,10])

# Set up train and test separated dataframes
train = df_total.loc[(df_total.loc[:,'train']==1)].copy(deep=True)
train = train.sample(frac=1)
y_train = train.loc[:,"y"]
wt_train = train.loc[:,"weights"]
X_train = train.drop(["y","weights","train"],axis=1)

#################### Train BDT #################################
print "<< Running parametric training >>"

if args.verboscity == "1":
  print "Parametric training dataset"
  print train.head(10)
  print "Bkg Length = {}, Bkg Sum of Weights = {}".format(len(train.loc[(train.loc[:,'y']==0)]),train.loc[(train.loc[:,'y']==0)].loc[:,"weights"].sum())
  print "Sig Length = {}, Sig Sum of Weights = {}".format(len(train.loc[(train.loc[:,'y']==1)]),train.loc[(train.loc[:,'y']==1)].loc[:,"weights"].sum())
  print "Total Length = {}, Total Sum of Weights = {}".format(len(train),train.loc[:,"weights"].sum())

if args.grid_search:
  param_grid = {
                'learning_rate': [.2,.3,.4],
                'n_estimators': [100],
                #'colsample_bytree': [0.7, 0.8],
                'max_depth': [3,4,5],
                #'reg_alpha': [1.1, 1.3],
                #'reg_lambda': [1.1,1.3],
                #'subsample': [0.7, 0.9],
                'min_child_weight': [3]
                }
  model = xgb.XGBClassifier()
  gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=4,
                    n_jobs=8,
                    scoring="roc_auc",
                    verbose=2
                    )
  xgb_model = gs.fit(X_train, y_train, wt_train)

  print "Best Score =",xgb_model.best_score_
  print "Best Hyperparameters =",xgb_model.best_params_

else:

  xgb_model = xgb.XGBClassifier(
                                learning_rate =0.05,
                                n_estimators=400,
                                max_depth=5,
                                min_child_weight=5,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                reg_alpha = 1.1,
                                reg_lambda = 1.3
                                )
  
  xgb_model.fit(X_train, y_train, sample_weight=wt_train)

pkl.dump(xgb_model,open("BDTs/binary_{}_{}.pkl".format(args.channel,args.year), "wb"))

print "<< Training parametric finished >>"

if args.train_dedicated:
  ded_mva = {}
  for mphi in phi_to_test_on:
    for mA in A_to_test_on:
      filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
      print "<< Running dedicated training for %(filename)s >>" % vars()
      train_dedicated = pd.concat([bkg_df_dedicated.dataframe,sig_df[filename+"_dedicated"].dataframe],ignore_index=True, sort=False) 
      train_dedicated = train_dedicated.loc[(train_dedicated.loc[:,'train']==1)]
      train_dedicated = train_dedicated.sample(frac=1)
      y_train_dedicated = train_dedicated.loc[:,"y"]
      wt_train_dedicated = train_dedicated.loc[:,"weights"]
      X_train_dedicated = train_dedicated.drop(["y","weights","train"],axis=1)

      if args.verboscity == "1":
        print "Dedicated training dataset"
        print train_dedicated.head(10)
        print "Bkg Length = {}, Bkg Sum of Weights = {}".format(len(train_dedicated.loc[(train_dedicated.loc[:,'y']==0)]),train_dedicated.loc[(train_dedicated.loc[:,'y']==0)].loc[:,"weights"].sum())
        print "Sig Length = {}, Sig Sum of Weights = {}".format(len(train_dedicated.loc[(train_dedicated.loc[:,'y']==1)]),train_dedicated.loc[(train_dedicated.loc[:,'y']==1)].loc[:,"weights"].sum())
        print "Total Length = {}, Total Sum of Weights = {}".format(len(train_dedicated),train_dedicated.loc[:,"weights"].sum())

      if args.grid_search_dedicated:
        param_grid = {
                      'learning_rate': [.1,.2],
                      'n_estimators': [100, 200],
                      #'colsample_bytree': [0.7, 0.8],
                      'max_depth': [1,2,3],
                      #'reg_alpha': [1.1, 1.3],
                      #'reg_lambda': [1.1,1.3],
                      #'subsample': [0.7, 0.9],
                      'min_child_weight': [3,5]
                      }
        model = xgb.XGBClassifier()
        gs = GridSearchCV(
                          estimator=model,
                          param_grid=param_grid, 
                          cv=5, 
                          n_jobs=-1, 
                          scoring="roc_auc",
                          verbose=0
                          )
        ded_mva[filename] = gs.fit(X_train_dedicated, y_train_dedicated, wt_train_dedicated)
        print "Best Score =",ded_mva[filename].best_score_
        print "Best Hyperparameters =",ded_mva[filename].best_params_

      else:
        ded_mva[filename] = xgb.XGBClassifier(
                                              learning_rate =0.2,
                                              n_estimators=100,
                                              max_depth=3,
                                              min_child_weight=5,
                                              subsample=0.9,
                                              colsample_bytree=0.8,
                                              reg_alpha = 1.1,
                                              reg_lambda = 1.3
                                              )
        ded_mva[filename].fit(X_train_dedicated, y_train_dedicated, sample_weight=wt_train_dedicated)

      print "<< Training dedicated finished for %(filename)s >>" % vars()

#################### Test BDT #################################

# ROC curve

# we want to draw a ROC curve and BDT scores for every pBDT mass point on each specific sample

for mphi_samp in phi_to_test_on:
  for mA_samp in A_to_test_on:
    filename = "phi%(mphi_samp)iA%(mA_samp)iTo4Tau" % vars()
    print "------------------------------------------------------------------------------------------------------------------------------------------------"
    print "<< Scoring file %(filename)s >>" % vars()

    # test performance of pBDT
    y_test = OrderedDict()
    wt_test = OrderedDict()
    X_test = OrderedDict()
    probs = OrderedDict()
    preds = OrderedDict()
    for mphi_mva in phi_to_test_on:
      for mA_mva in A_to_test_on:
        print "<< Setting mphi = {} and mA = {} in MVA >>".format(mphi_mva,mA_mva)
        mva_name = "pBDT: m_{\phi}=%(mphi_mva)i, m_{A}=%(mA_mva)i" % vars()
        sig_test =  sig_df[filename].dataframe.loc[(sig_df[filename].dataframe.loc[:,'train']==0)]
        sig_test.loc[:,"mphi"] = mphi_mva
        sig_test.loc[:,"mA"] = mA_mva

        if args.duplicate_bkg:
          bkg_test = bkg_df["phi{}A{}To4Tau".format(phi_to_train_on[0],A_to_train_on[0])].dataframe.loc[(bkg_df["phi{}A{}To4Tau".format(phi_to_train_on[0],A_to_train_on[0])].dataframe.loc[:,'train']==0)]
        else:
          bkg_test = bkg_df.dataframe.loc[(bkg_df.dataframe.loc[:,"train"]==0)]
        bkg_test.loc[:,"mphi"] = mphi_mva
        bkg_test.loc[:,"mA"] = mA_mva

        test = pd.concat([sig_test,bkg_test],ignore_index=True, sort=False)

        if args.verboscity == "1":
          print "Testing Dataset"
          print test.head(10)
          print "Bkg Length = {}, Bkg Sum of Weights = {}".format(len(test.loc[(test.loc[:,'y']==0)]),test.loc[(test.loc[:,'y']==0)].loc[:,"weights"].sum())
          print "Sig Length = {}, Sig Sum of Weights = {}".format(len(test.loc[(test.loc[:,'y']==1)]),test.loc[(test.loc[:,'y']==1)].loc[:,"weights"].sum())
          print "Total Length = {}, Total Sum of Weights = {}".format(len(test),test.loc[:,"weights"].sum())

        y_test[mva_name] = test.loc[:,"y"]
        wt_test[mva_name] = test.loc[:,"weights"]
        X_test[mva_name] = test.drop(["y","weights","train"],axis=1)
        probs[mva_name] = xgb_model.predict_proba(X_test[mva_name])
        preds[mva_name] = probs[mva_name][:,1]

        # test performance of dedicated
        ded_name = "Dedicated: m_{\phi}=%(mphi_mva)i, m_{A}=%(mA_mva)i" % vars()

        test_dedicated = pd.concat([bkg_df_dedicated.dataframe,sig_df[filename+"_dedicated"].dataframe],ignore_index=True, sort=False)
        test_dedicated = test_dedicated.loc[(test_dedicated.loc[:,'train']==0)]

        if args.verboscity == "1":
          print "<< Testing dedicated BDT >>"
          print "Testing Dataset"
          print test_dedicated.head(10)
          print "Bkg Length = {}, Bkg Sum of Weights = {}".format(len(test_dedicated.loc[(test_dedicated.loc[:,'y']==0)]),test_dedicated.loc[(test_dedicated.loc[:,'y']==0)].loc[:,"weights"].sum())
          print "Sig Length = {}, Sig Sum of Weights = {}".format(len(test_dedicated.loc[(test_dedicated.loc[:,'y']==1)]),test_dedicated.loc[(test_dedicated.loc[:,'y']==1)].loc[:,"weights"].sum())
          print "Total Length = {}, Total Sum of Weights = {}".format(len(test_dedicated),test_dedicated.loc[:,"weights"].sum())

        y_test[ded_name] = test_dedicated.loc[:,"y"]
        wt_test[ded_name] = test_dedicated.loc[:,"weights"]
        X_test[ded_name] = test_dedicated.drop(["y","weights","train"],axis=1)
        probs[ded_name] = ded_mva["phi%(mphi_mva)iA%(mA_mva)iTo4Tau" % vars()].predict_proba(X_test[ded_name])
        preds[ded_name] = probs[ded_name][:,1]

    DrawMultipleROCCurves(y_test,preds,wt_test,"roc_curve_%(filename)s" % vars(),name=filename)

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
