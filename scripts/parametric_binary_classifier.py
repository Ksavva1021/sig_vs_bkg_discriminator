"""
To Do:
- Fix copy warning
- Add increased priority to border sample masses
- Add option to use all tau decay channels in training
- one hot encode channel
- Does using all channels improve performance in each channel?

"""

from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawROCCurve, DrawBDTScoreDistributions, DrawFeatureImportance, DrawVarDistribution, DrawMultipleROCCurves, DrawMultipleROCCurvesOnOnePage
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import itertools
import copy
import json
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mttt')
parser.add_argument('--year', help= 'Year to train BDT for', default='2018')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
parser.add_argument('--load', help= 'Load dataframe from file',  action='store_true')
parser.add_argument('--load_hyperparameters', help= 'Load hyperparameters from file',  action='store_true')
parser.add_argument('--batch', help= 'Run on batch',  action='store_true')
parser.add_argument('--duplicate_bkg', help= 'Will duplicate the background events per mass point tested rather than randomise.',  action='store_true')
parser.add_argument('--use_deeptau', help= 'Use the deeptau scores for training',  action='store_true')
parser.add_argument('--train_dedicated', help= 'Train dedicated MVA to test against parametric MVA.',  action='store_true')
parser.add_argument('--grid_search', help= 'Do grid search to find best hyperparameters',  action='store_true')
parser.add_argument('--grid_search_dedicated', help= 'Do grid search to find best hyperparameters for the dedicated bdt',  action='store_true')
parser.add_argument('--edge_ratios',help= 'Colon separated list of weights to give depending on whether sample masses are fully surrounded, partialy surrounded or on a corner', default='1:1:1')
args = parser.parse_args()

#################### Functions #################################


def CreateBatchJob(name,cmssw_base,cmd_list):
  if os.path.exists(job_file): os.system('rm %(name)s' % vars())
  os.system('echo "#!/bin/bash" >> %(name)s' % vars())
  os.system('echo "ulimit -c 0" >> %(name)s' % vars())
  for cmd in cmd_list:
    os.system('echo "%(cmd)s" >> %(name)s' % vars())
  os.system('chmod +x %(name)s' % vars())
  print "Created job:",name

def SubmitBatchJob(name,time=180,memory=24,cores=1):
  error_log = name.replace('.sh','_error.log')
  output_log = name.replace('.sh','_output.log')
  if os.path.exists(error_log): os.system('rm %(error_log)s' % vars())
  if os.path.exists(output_log): os.system('rm %(output_log)s' % vars())
  if cores>1: os.system('qsub -e %(error_log)s -o %(output_log)s -V -q hep.q -pe hep.pe %(cores)s -l h_rt=0:%(time)s:0 -l h_vmem=%(memory)sG -cwd %(name)s' % vars())
  else: os.system('qsub -e %(error_log)s -o %(output_log)s -V -q hep.q -l h_rt=0:%(time)s:0 -l h_vmem=%(memory)sG -cwd %(name)s' % vars())


def PrintDatasetSummary(name,dataset):
  print name
  print dataset.head(10)
  print "Bkg Length = {}, Bkg Sum of Weights = {}".format(len(dataset.loc[(dataset.loc[:,'y']==0)]),dataset.loc[(dataset.loc[:,'y']==0)].loc[:,"weights"].sum())
  print "Sig Length = {}, Sig Sum of Weights = {}".format(len(dataset.loc[(dataset.loc[:,'y']==1)]),dataset.loc[(dataset.loc[:,'y']==1)].loc[:,"weights"].sum())
  print "Total Length = {}, Total Sum of Weights = {}".format(len(dataset),dataset.loc[:,"weights"].sum())
  print ""

def SampleAndSplit_X_y_wt(dataset):
  y = dataset.loc[:,"y"]
  wt = dataset.loc[:,"weights"]
  X = dataset.drop(["y","weights","train"],axis=1)
  return X, y, wt


# submit job to batch
if args.batch:
  cmd = "python scripts/parametric_binary_classifier.py"
  for key, val in vars(args).items():
    if not key in ["batch"]:
      if type(val) == str: val = "'{}'".format(val)
      if not val in [False,True]:
        cmd += " --{}={}".format(key,val)
      elif val == True:
        cmd += " --{}".format(key)
  job_file = "jobs/batch_job_pBDT_{}_{}.sh".format(args.channel,args.year)
  cmssw_base = os.getcwd().replace('src/UserCode/sig_vs_bkg_discriminator','')
  CreateBatchJob(job_file,cmssw_base,[cmd])
  SubmitBatchJob(job_file,time=180,memory=24,cores=1)
  exit()


#################### Variables #################################

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


if not args.load:
  print "<< Making dataframe >>"  

  # Add signal dataframes
  sig_df = {}
  if not args.duplicate_bkg: bkg_df = {}

  for mphi in list(set(phi_to_train_on+phi_to_test_on)):
    for mA in list(set(A_to_train_on+A_to_test_on)):
      filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars() 
      sig_df[filename] = Dataframe()
      sig_df[filename].LoadRootFilesFromJson("json_selection/{}_{}_sig_all.json".format(args.channel,args.year),variables,specific_file=filename,quiet=(args.verbosity<2))
      sig_df[filename].dataframe.loc[:,"y"] = 1
      if args.train_dedicated:
        sig_df[filename+"_dedicated"] = sig_df[filename].Copy()
        sig_df[filename+"_dedicated"].TrainTestSplit()
        sig_df[filename+"_dedicated"].NormaliseWeights(train_frac=1.0,test_frac=1.0)
      if mphi in phi_to_train_on and mA in A_to_train_on:
        sig_df[filename].TrainTestSplit()
      else:
        sig_df[filename].dataframe.loc[:,"train"] = 0
      
      count_edges = (mphi == phi_to_train_on[0] or mphi == phi_to_train_on[-1]) + (mA == A_to_train_on[0] or mA == A_to_train_on[-1])
      scale = float(args.edge_ratios.split(":")[count_edges])
      sum_scale = (4*float(args.edge_ratios.split(":")[2])) + (((2*(len(phi_to_train_on)-2)) + 2*(len(A_to_train_on)-2))*float(args.edge_ratios.split(":")[1])) + (((len(phi_to_train_on)-2)*(len(A_to_train_on)-2))*float(args.edge_ratios.split(":")[0]))
      print filename, scale, sum_scale
      sig_df[filename].NormaliseWeights(test_frac=1.0,train_frac=scale/sum_scale)
      sig_df[filename].dataframe.loc[:,"mphi"] = mphi
      sig_df[filename].dataframe.loc[:,"mA"] = mA

  # Add background dataframes
  if args.duplicate_bkg:
    bkg_df = {}
    for mphi in phi_to_train_on:
      for mA in A_to_train_on:
        filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
        bkg_df[filename] = Dataframe()
        bkg_df[filename].LoadRootFilesFromJson("json_selection/{}_{}_bkg.json".format(args.channel,args.year),variables,quiet=(args.verbosity<2))
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
    bkg_df.LoadRootFilesFromJson("json_selection/{}_{}_bkg.json".format(args.channel,args.year),variables,quiet=(args.verbosity<2))
    bkg_df.TrainTestSplit()
    bkg_df.NormaliseWeights(test_frac=1.0,train_frac=1.0)
    bkg_df.dataframe.loc[:,"y"] = 0
    if args.train_dedicated:
      bkg_df_dedicated = bkg_df.Copy()
    bkg_df.dataframe.loc[:,"mphi"] = np.random.choice(phi_to_train_on, bkg_df.dataframe.shape[0])
    bkg_df.dataframe.loc[:,"mA"] = np.random.choice(A_to_train_on, bkg_df.dataframe.shape[0])

 

  # Set up train separated pBDT dataframes 
  combine_list = []
  for mphi in phi_to_train_on:
    for mA in A_to_train_on:
      filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
      combine_list.append(sig_df[filename].dataframe)
      if args.duplicate_bkg: combine_list.append(bkg_df[filename].dataframe)
  if not args.duplicate_bkg: combine_list.append(bkg_df.dataframe)
  pBDT_total = pd.concat(combine_list,ignore_index=True, sort=False)
  train = pBDT_total.loc[(pBDT_total.loc[:,'train']==1)].copy(deep=True)
  X_train, y_train, wt_train = SampleAndSplit_X_y_wt(train)

  X_train.to_pickle("dataframes/pBDT_X_train_{}_{}.pkl".format(args.channel,args.year))
  y_train.to_pickle("dataframes/pBDT_y_train_{}_{}.pkl".format(args.channel,args.year))
  wt_train.to_pickle("dataframes/pBDT_wt_train_{}_{}.pkl".format(args.channel,args.year))
  
  # Set up test separated pBDT dataframes dictionary
  test = {}
  X_test = {}
  wt_test = {}
  y_test = {}
  for mphi in phi_to_test_on:
    for mA in A_to_test_on:
      filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
      sig_test =  sig_df[filename].dataframe.loc[(sig_df[filename].dataframe.loc[:,'train']==0)]
      if args.duplicate_bkg:
        bkg_test = bkg_df[filename].dataframe.loc[(bkg_df[filename].dataframe.loc[:,'train']==0)]
      else:
        bkg_test = bkg_df.dataframe.loc[(bkg_df.dataframe.loc[:,"train"]==0)]
        bkg_test.loc[:,"mphi"] = mphi_mva
        bkg_test.loc[:,"mA"] = mA_mva
  
      test[filename] = pd.concat([sig_test,bkg_test],ignore_index=True, sort=False)
      X_test[filename], y_test[filename], wt_test[filename] = SampleAndSplit_X_y_wt(test[filename])
  
      X_test[filename].to_pickle("dataframes/pBDT_{}_X_test_{}_{}.pkl".format(filename,args.channel,args.year))
      y_test[filename].to_pickle("dataframes/pBDT_{}_y_test_{}_{}.pkl".format(filename,args.channel,args.year))
      wt_test[filename].to_pickle("dataframes/pBDT_{}_wt_test_{}_{}.pkl".format(filename,args.channel,args.year))
   
  if args.verbosity > 0: 
    PrintDatasetSummary("Parametric training dataset",train)
  
    for mphi in phi_to_test_on:
      for mA in A_to_test_on:
        filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
        PrintDatasetSummary("Parametric %(filename)s testing dataset" % vars(),test[filename])
   
    # Set dedicated BDT dataframes
    if args.train_dedicated:
      train_dedicated = {}
      X_train_dedicated = {}
      wt_train_dedicated = {}
      y_train_dedicated = {}
      test_dedicated = {}
      X_test_dedicated = {}
      wt_test_dedicated = {}
      y_test_dedicated = {}
      for mphi in phi_to_test_on:
        for mA in A_to_test_on:
          filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
          dedicated = pd.concat([bkg_df_dedicated.dataframe,sig_df[filename+"_dedicated"].dataframe],ignore_index=True, sort=False)
          train_dedicated[filename] = dedicated.loc[(dedicated.loc[:,'train']==1)]
          test_dedicated[filename] = dedicated.loc[(dedicated.loc[:,'train']==0)]
 
          X_train_dedicated[filename], y_train_dedicated[filename], wt_train_dedicated[filename] = SampleAndSplit_X_y_wt(train_dedicated[filename])
          X_test_dedicated[filename], y_test_dedicated[filename], wt_test_dedicated[filename] = SampleAndSplit_X_y_wt(test_dedicated[filename])
  
          X_train_dedicated[filename].to_pickle("dataframes/dedicated_{}_X_train_{}_{}.pkl".format(filename,args.channel,args.year))
          y_train_dedicated[filename].to_pickle("dataframes/dedicated_{}_y_train_{}_{}.pkl".format(filename,args.channel,args.year))
          wt_train_dedicated[filename].to_pickle("dataframes/dedicated_{}_wt_train_{}_{}.pkl".format(filename,args.channel,args.year))
  
          X_test_dedicated[filename].to_pickle("dataframes/dedicated_{}_X_test_{}_{}.pkl".format(filename,args.channel,args.year))    
          y_test_dedicated[filename].to_pickle("dataframes/dedicated_{}_y_test_{}_{}.pkl".format(filename,args.channel,args.year))
          wt_test_dedicated[filename].to_pickle("dataframes/dedicated_{}_wt_test_{}_{}.pkl".format(filename,args.channel,args.year))
  
    if args.verbosity > 0:
      for mphi in phi_to_test_on:
        for mA in A_to_test_on:
          filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
          PrintDatasetSummary("Dedicated %(filename)s training dataset" % vars(),train_dedicated[filename])
          PrintDatasetSummary("Dedicated %(filename)s testing dataset" % vars(),test_dedicated[filename])

else:
  print "<< Loading in dataframes >>"
  X_train = pd.read_pickle("dataframes/pBDT_X_train_{}_{}.pkl".format(args.channel,args.year))
  y_train = pd.read_pickle("dataframes/pBDT_y_train_{}_{}.pkl".format(args.channel,args.year))
  wt_train = pd.read_pickle("dataframes/pBDT_wt_train_{}_{}.pkl".format(args.channel,args.year))
  X_train_dedicated = {}
  wt_train_dedicated = {}
  y_train_dedicated = {}
  X_test_dedicated = {}
  wt_test_dedicated = {}
  y_test_dedicated = {}
  X_test = {}
  wt_test = {}
  y_test = {}
  for mphi in phi_to_test_on:
    for mA in A_to_test_on:
      filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
      X_test[filename] = pd.read_pickle("dataframes/pBDT_{}_X_test_{}_{}.pkl".format(filename,args.channel,args.year))
      y_test[filename] = pd.read_pickle("dataframes/pBDT_{}_y_test_{}_{}.pkl".format(filename,args.channel,args.year))
      wt_test[filename] = pd.read_pickle("dataframes/pBDT_{}_wt_test_{}_{}.pkl".format(filename,args.channel,args.year))
      X_train_dedicated[filename] = pd.read_pickle("dataframes/dedicated_{}_X_train_{}_{}.pkl".format(filename,args.channel,args.year))
      y_train_dedicated[filename] = pd.read_pickle("dataframes/dedicated_{}_y_train_{}_{}.pkl".format(filename,args.channel,args.year))
      wt_train_dedicated[filename] = pd.read_pickle("dataframes/dedicated_{}_wt_train_{}_{}.pkl".format(filename,args.channel,args.year))
      X_test_dedicated[filename] = pd.read_pickle("dataframes/dedicated_{}_X_test_{}_{}.pkl".format(filename,args.channel,args.year))
      y_test_dedicated[filename] = pd.read_pickle("dataframes/dedicated_{}_y_test_{}_{}.pkl".format(filename,args.channel,args.year))
      wt_test_dedicated[filename] = pd.read_pickle("dataframes/dedicated_{}_wt_test_{}_{}.pkl".format(filename,args.channel,args.year))


#################### Train BDT #################################
print "<< Running parametric training >>"
if args.grid_search:
  param_grid = {
                'learning_rate': [.1],
                'n_estimators': [300,400,500,600,700,800,900,1000],
                'colsample_bytree': [0.7],
                'max_depth': [3,4,5,6,7,8,9],
                'reg_alpha': [0.9],
                'reg_lambda': [0.4],
                'subsample': [1.0],
                'min_child_weight': [3]
                }

  param_grid = {"objective": ["binary:logistic"], "seed": [0], "n_estimators": [750], "learning_rate": [0.05], "gamma": [0.5], "max_depth": [4], "min_child_weight": [0.0], "subsample": [1.0], "colsample_bytree": [1.0], "reg_alpha": [0.0], "reg_lambda": [1.0]}

  print "Performing grid search for parameters:"
  print param_grid

  keys, values = zip(*param_grid.items())
  permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
  best_ave_auc = 0
  for ind, val in enumerate(permutations_dicts):
    # fit model
    model = xgb.XGBClassifier()
    model.set_params(**val)
    model.fit(X_train, y_train, sample_weight=wt_train)

    # score model
    auc_scores = []
    for mphi in phi_to_test_on:
      for mA in A_to_test_on:
        filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
        probs = model.predict_proba(X_test[filename])
        preds = probs[:,1]
        auc_scores.append(roc_auc_score(y_test[filename], preds, sample_weight=wt_test[filename]))
    ave_auc = sum(auc_scores)/len(auc_scores) 
    if args.verbosity > 1:
      print "Testing:", val 
      print "AUC score:", ave_auc
      print str(ind)+"/"+str(len(permutations_dicts))
    if ave_auc > best_ave_auc:
      best_ave_auc = copy.deepcopy(ave_auc)
      xgb_model = copy.deepcopy(model)
      best_hyperparameters = copy.deepcopy(val)

  print "Best Hyperparameters =", best_hyperparameters
  print "Best Score =", best_ave_auc

  with open('hyperparameters/pBDT.json', 'w') as outfile:
    json.dump(best_hyperparameters, outfile)

  for key, val in best_hyperparameters.items():
    if val == min(param_grid[key]) or val == max(param_grid[key]):
      print "WARNING: Best hyperparameter for %(key)s found on edge of search" % vars()

else:

  if args.load_hyperparameters:
    with open('hyperparameters/pBDT.json') as json_file:
      data = json.load(json_file)
    xgb_model = xgb.XGBClassifier()
    xgb_model.set_params(**data)
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

pkl.dump(xgb_model,open("BDTs/pBDT_{}_{}.pkl".format(args.channel,args.year), "wb"))

print "<< Training parametric finished >>"

if args.train_dedicated:
  ded_mva = {}
  for mphi in phi_to_test_on:
    for mA in A_to_test_on:
      filename = "phi%(mphi)iA%(mA)iTo4Tau" % vars()
      print "<< Running dedicated training for %(filename)s >>" % vars()

      if args.grid_search_dedicated:
        param_grid = {
                      'learning_rate': [.1,.2,.3],
                      'n_estimators': [50,100,150],
                      #'colsample_bytree': [0.6,0.7,0.8,0.9,1.0],
                      'max_depth': [6,7,8],
                      #'reg_alpha': [0.9,1.1,1.3,1.5],
                      #'reg_lambda': [1.1,1.3,1.5],
                      #'subsample': [0.5, 0.7, 0.9],
                      #'min_child_weight': [1,2,3]
                      }

        print "Performing grid search for parameters:"
        print param_grid

        keys, values = zip(*param_grid.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        best_auc = 0
        for val in permutations_dicts:
          # fit model
          model = xgb.XGBClassifier()
          model.set_params(**val)
          model.fit(X_train_dedicated[filename], y_train_dedicated[filename], sample_weight=wt_train_dedicated[filename])

          # score model
          probs = model.predict_proba(X_test_dedicated[filename])
          preds = probs[:,1]
          auc = roc_auc_score(y_test_dedicated[filename], preds, sample_weight=wt_test_dedicated[filename])
          if args.verbosity > 1:
            print "Testing:", val
            print "AUC score:", auc
          if auc > best_auc:
            best_auc = copy.deepcopy(auc)
            ded_mva[filename] = copy.deepcopy(model)
            best_hyperparameters = copy.deepcopy(val)

        print "Best Hyperparameters =", best_hyperparameters
        print "Best Score =", best_auc

        with open('hyperparameters/dedicated_%(filename)s.json' % vars(), 'w') as outfile:
          json.dump(best_hyperparameters, outfile)

        for key, val in best_hyperparameters.items():
          if val == min(param_grid[key]) or val == max(param_grid[key]):
            print "WARNING: Best hyperparameter for %(key)s found on edge of search" % vars()

      else:

        if args.load_hyperparameters:
          with open('hyperparameters/dedicated_%(filename)s.json' % vars()) as json_file:
            data = json.load(json_file)
          ded_mva[filename] = xgb.XGBClassifier()
          ded_mva[filename].set_params(**data)
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
        ded_mva[filename].fit(X_train_dedicated[filename], y_train_dedicated[filename], sample_weight=wt_train_dedicated[filename])

      print "<< Training dedicated finished for %(filename)s >>" % vars()


#################### Test BDT #################################

# ROC curve

# we want to draw a ROC curve and BDT scores for every pBDT mass point on each specific sample
y_test_plot = OrderedDict()
preds = OrderedDict()
wt_test_plot = OrderedDict()

for mphi_samp in phi_to_test_on:
  for mA_samp in A_to_test_on:
    filename = "phi%(mphi_samp)iA%(mA_samp)iTo4Tau" % vars()
    plot_name = "Testing on $m_{\phi}=%(mphi_samp)i$, $m_{A}=%(mA_samp)i$ sample" % vars()
    print "<< Scoring file %(filename)s >>" % vars()

    # test performance of pBDT
    y_test_plot[plot_name] = OrderedDict()
    wt_test_plot[plot_name] = OrderedDict()
    X_test_plot = OrderedDict()
    probs = OrderedDict()
    preds[plot_name] = OrderedDict()

    mva_name = "pBDT: m_{\phi}=%(mphi_samp)i, m_{A}=%(mA_samp)i" % vars()
    y_test_plot[plot_name][mva_name] = y_test[filename]
    wt_test_plot[plot_name][mva_name] = wt_test[filename]
    X_test_plot[mva_name] = X_test[filename]
    probs[mva_name] = xgb_model.predict_proba(X_test_plot[mva_name])
    preds[plot_name][mva_name] = probs[mva_name][:,1]


    for mphi_mva in phi_to_test_on:
      for mA_mva in A_to_test_on:
        ded_name = "Dedicated: m_{\phi}=%(mphi_mva)i, m_{A}=%(mA_mva)i" % vars()
        y_test_plot[plot_name][ded_name] = y_test_dedicated[filename]
        wt_test_plot[plot_name][ded_name] = wt_test_dedicated[filename]
        X_test_plot[ded_name] = X_test_dedicated[filename]
        probs[ded_name] = ded_mva["phi%(mphi_mva)iA%(mA_mva)iTo4Tau" % vars()].predict_proba(X_test_plot[ded_name])
        preds[plot_name][ded_name] = probs[ded_name][:,1]

    DrawMultipleROCCurves(y_test_plot[plot_name],preds[plot_name],wt_test_plot[plot_name],"roc_curve_%(filename)s" % vars(),name=plot_name)

DrawMultipleROCCurvesOnOnePage(y_test_plot,preds,wt_test_plot,len_x=len(phi_to_test_on),len_y=len(A_to_test_on))
