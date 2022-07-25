from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawDistributions,DrawROCCurve, DrawBDTScoreDistributions, DrawFeatureImportance, DrawVarDistribution
import argparse
import pickle as pkl
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np
from collections import OrderedDict
import ROOT
import math


parser = argparse.ArgumentParser()
parser.add_argument('--year', help= 'Year to train BDT for', default='2018')
parser.add_argument('--scenario', help = 'Scenario to run', default='s1')
parser.add_argument('--load', help= 'Load dataframe from file',  action='store_true')
parser.add_argument('--use_deeptauVsJets', help= 'Use the deeptau scores for training',  action='store_true')
parser.add_argument('--use_deeptauVsEle', help= 'Use the deeptau scores for training',  action='store_true')
parser.add_argument('--use_deeptauVsMu', help= 'Use the deeptau scores for training',  action='store_true')
parser.add_argument('--draw_distribution', help= 'Draw variable distributions',  action='store_true')
args = parser.parse_args()

#updated
variables = ["pt_1", "pt_2","pt_3","pt_4","dphi_12","dphi_13","dphi_14","dphi_23","dphi_24","dphi_34","dR_12","dR_13","dR_14","dR_23",
"dR_24","dR_34","mt_1","mt_2","mt_3","mt_4","mt_lep_12","mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34",
"mvis_12","mvis_13","mvis_14","mvis_23","mvis_24","mvis_34","q_1","q_2","q_3","q_4"]

if args.use_deeptauVsJets:
  variables += [
                "deepTauVsJets_iso_1","deepTauVsJets_iso_2","deepTauVsJets_iso_3","deepTauVsJets_iso_4"
                ]

if args.use_deeptauVsEle:
  variables += [
                "deepTauVsEle_iso_1","deepTauVsEle_iso_2","deepTauVsEle_iso_3","deepTauVsEle_iso_4"
                ]

if args.use_deeptauVsMu:
  variables += [
                "deepTauVsMu_iso_1","deepTauVsMu_iso_2","deepTauVsMu_iso_3","deepTauVsMu_iso_4"
                ]

abs_variables = ["dphi_12","dphi_13","dphi_14","dphi_23","dphi_24","dphi_34","dR_12","dR_13","dR_14","dR_23",
"dR_24","dR_34"]

channels = ["tttt","emtt","ettt","mmtt","mttt","eett"]
#channels = ["mttt"]
if not (args.load):
  print "<< Making dataframe >>"  
  train_dataframes = []
  test_dataframes = []
  for ch in channels:
    print "Channel:", ch
    # Add signal dataframe
    sig_df = Dataframe()
    sig_df.LoadRootFilesFromJson("json_selection_{}/{}_{}_sig.json".format(args.scenario,ch,args.year),variables)
    sig_df.dataframe.loc[:,"{}".format(ch)] = 1
    sig_df.dataframe.loc[:,"y"] = 1
    # Add background data
    bkg_df = Dataframe()
    bkg_df.LoadRootFilesFromJson("json_selection_{}/{}_{}_bkg.json".format(args.scenario,ch,args.year),variables)
    bkg_df.dataframe.loc[:,"{}".format(ch)] = 1
    bkg_df.dataframe.loc[:,"y"] = 0
    #Combine dataframes
    df_total = pd.concat([bkg_df.dataframe,sig_df.dataframe],ignore_index=True, sort=False)
  
    for i in abs_variables: 
        df_total[i] = df_total[i].abs()
        
    # Set up train and test separated dataframes
    train, test = train_test_split(df_total,test_size=0.5, random_state=42)  
    
    train_sig_df = train[train['y'] == 1]
    train_bkg_df = train[train['y'] == 0]    
  
    sig_df = Dataframe()
    bkg_df = Dataframe()
  
    sig_df.dataframe = train_sig_df.copy(deep=True)
    sig_df.NormaliseWeights()  
    bkg_df.dataframe = train_bkg_df.copy(deep=True)
    bkg_df.NormaliseWeights()

    df_train_total = pd.concat([bkg_df.dataframe,sig_df.dataframe],ignore_index=True, sort=False)
    train_dataframes.append(df_train_total)
    test_dataframes.append(test)

    
  df_train_total = pd.concat(train_dataframes,ignore_index=True,sort=False)
  df_train_total[channels] = df_train_total[channels].fillna(value=0)
  df_test_total = pd.concat(test_dataframes,ignore_index=True,sort=False)
  df_test_total[channels] = df_test_total[channels].fillna(value=0)

  df_train_total.to_pickle("dataframes/{}/df_train_total_{}.pkl".format(args.scenario,args.year))
  df_test_total.to_pickle("dataframes/{}/df_test_total_{}.pkl".format(args.scenario,args.year))
else:
  print "<< Loading in dataframe >>"
  df_train_total = pd.read_pickle("dataframes/{}/df_train_total_{}.pkl".format(args.scenario,args.year))
  df_test_total = pd.read_pickle("dataframes/{}/df_test_total_{}.pkl".format(args.scenario,args.year))

  
train_dict = OrderedDict()
X_train_dict = OrderedDict()
wt_train_dict = OrderedDict()
y_train_dict = OrderedDict()

train_sig_dict = OrderedDict()
train_bkg_dict = OrderedDict()
test_sig_dict = OrderedDict()
test_bkg_dict = OrderedDict()

test_dict = OrderedDict()
X_test_dict = OrderedDict()
wt_test_dict = OrderedDict()
y_test_dict = OrderedDict()
t0_dict = OrderedDict()
t1_dict = OrderedDict()
wt0_dict = OrderedDict()
wt1_dict = OrderedDict()
xt0_dict = OrderedDict()
xt1_dict = OrderedDict()

for ch in channels:
  
  train_dedicated = df_train_total[(df_train_total["{}".format(ch)] != 0)]
  train_dedicated = train_dedicated.loc[:, (train_dedicated != 0).any(axis=0)]
  train_dedicated = train_dedicated.drop("{}".format(ch),axis=1)

  train_sig_dict['{}'.format(ch)]  = train_dedicated[train_dedicated['y'] == 1]
  train_bkg_dict['{}'.format(ch)] = train_dedicated[train_dedicated['y'] == 0]  

  y_train = train_dedicated.loc[:,"y"]
  wt_train = train_dedicated.loc[:,"weights"]
  X_train = train_dedicated.drop(["y","weights"],axis=1)
  y_train_dict["{}".format(ch)] = y_train
  wt_train_dict["{}".format(ch)] = wt_train
  X_train_dict["{}".format(ch)] = X_train
  
  test_dedicated = df_test_total[(df_test_total["{}".format(ch)] != 0)]
  test_dedicated = test_dedicated.loc[:, (test_dedicated != 0).any(axis=0)]
  test_dedicated = test_dedicated.drop("{}".format(ch),axis=1)

  test_sig_dict['{}'.format(ch)]  = test_dedicated[test_dedicated['y'] == 1]
  test_bkg_dict['{}'.format(ch)] = test_dedicated[test_dedicated['y'] == 0]  

  y_test = test_dedicated.loc[:,"y"]
  wt_test = test_dedicated.loc[:,"weights"]
  X_test = test_dedicated.drop(["y","weights"],axis=1)
  y_test_dict["{}".format(ch)] = y_test
  wt_test_dict["{}".format(ch)] = wt_test
  X_test_dict["{}".format(ch)] = X_test
  
  # BDT scores
  t0_dict["{}".format(ch)] = test_dedicated.loc[(test_dedicated.loc[:,"y"]==0)]
  t1_dict["{}".format(ch)] = test_dedicated.loc[(test_dedicated.loc[:,"y"]==1)]
  wt0_dict["{}".format(ch)] = t0_dict["{}".format(ch)].loc[:,"weights"]
  wt1_dict["{}".format(ch)] = t1_dict["{}".format(ch)].loc[:,"weights"]
  xt0_dict["{}".format(ch)] = t0_dict["{}".format(ch)].drop(["y","weights"],axis=1)
  xt1_dict["{}".format(ch)] = t1_dict["{}".format(ch)].drop(["y","weights"],axis=1)

if args.draw_distribution:  
    for ch in channels:
        if ch == 'tttt':
            i_ = [1,2,3,4]
            for i in i_:
                DrawDistributions(train_sig_dict['{}'.format(ch)],train_bkg_dict['{}'.format(ch)],'deepTauVsJets_iso_{}'.format(i),[0,1],100,channel='{}'.format(ch),location="plots/Distributions/train_")
                DrawDistributions(test_sig_dict['{}'.format(ch)],test_bkg_dict['{}'.format(ch)],'deepTauVsJets_iso_{}'.format(i),[0,1],100,channel='{}'.format(ch),location="plots/Distributions/test_")
        if ch == "ettt" or ch == "mttt":
            i_ = [2,3,4]
            for i in i_:
                DrawDistributions(train_sig_dict['{}'.format(ch)],train_bkg_dict['{}'.format(ch)],'deepTauVsJets_iso_{}'.format(i),[0,1],100,channel='{}'.format(ch),location="plots/Distributions/train_")
                DrawDistributions(test_sig_dict['{}'.format(ch)],test_bkg_dict['{}'.format(ch)],'deepTauVsJets_iso_{}'.format(i),[0,1],100,channel='{}'.format(ch),location="plots/Distributions/test_")
        if ch == "eett" or ch == "mmtt" or ch =='emtt':
            i_ = [3,4]
            for i in i_:
                DrawDistributions(train_sig_dict['{}'.format(ch)],train_bkg_dict['{}'.format(ch)],'deepTauVsJets_iso_{}'.format(i),[0,1],100,channel='{}'.format(ch),location="plots/Distributions/train_")
                DrawDistributions(test_sig_dict['{}'.format(ch)],test_bkg_dict['{}'.format(ch)],'deepTauVsJets_iso_{}'.format(i),[0,1],100,channel='{}'.format(ch),location="plots/Distributions/test_")        
    

for ch in channels:
   print(ch)  
   xgb_model = xgb.XGBClassifier()
   eval_set = [(X_train_dict['{}'.format(ch)],y_train_dict['{}'.format(ch)]),(X_test_dict['{}'.format(ch)], y_test_dict['{}'.format(ch)])]
   weight_eval_set = [wt_train_dict['{}'.format(ch)],wt_test_dict['{}'.format(ch)]]
   xgb_model.fit(X_train_dict['{}'.format(ch)], y_train_dict['{}'.format(ch)], eval_metric = "auc",sample_weight=wt_train_dict['{}'.format(ch)], eval_set = eval_set,sample_weight_eval_set = weight_eval_set,verbose = False)
   
   #DrawFeatureImportance(xgb_model,"weight","{}/{}/feature_importance".format(args.scenario,ch))
   
   probs = xgb_model.predict_proba(X_test_dict['{}'.format(ch)])
   preds = probs[:,1]
   
   #BDT Scores
   probs0 = xgb_model.predict_proba(xt0_dict['{}'.format(ch)])
   probs1 = xgb_model.predict_proba(xt1_dict['{}'.format(ch)])
   preds0 = probs0[:,1]
   preds1 = probs1[:,1]
   wt0 = np.array(wt0_dict['{}'.format(ch)])
   wt1 = np.array(wt1_dict['{}'.format(ch)])
   
   sig_hist  = ROOT.TH1F( 'signal_{}'.format(ch), 'This is the signal distribution', 1000, 0, 1 )
   sig_hist.Sumw2()

   bkg_hist  = ROOT.TH1F( 'bkg_{}'.format(ch), 'This is the background distribution', 1000, 0, 1 )
   bkg_hist.Sumw2()

   
   for i in range(len(preds1)):
      sig_hist.Fill(preds1[i],wt1[i])
   for i in range(len(preds0)):
      bkg_hist.Fill(preds0[i],wt0[i])
      
   integral = sig_hist.Integral()
   running_integral = 0
   percentage = 0
   cut = 0
   error = 0
   for j in reversed(range(sig_hist.GetNbinsX()+1)):
      if percentage < 0.68:
         running_integral += sig_hist.GetBinContent(j)
         percentage = running_integral/integral
         #print(j,sig_hist.GetBinContent(j),running_integral,integral,percentage)
      else:
         print(j)
         cut = j
         break

   sig_error= ROOT.Double()
   sig_integral = sig_hist.IntegralAndError(cut,sig_hist.GetNbinsX()+1,sig_error)
   
   bkg_error= ROOT.Double()
   bkg_integral = bkg_hist.IntegralAndError(cut,bkg_hist.GetNbinsX()+1,bkg_error)
   
   print(sig_integral,sig_error)
   print(bkg_integral,bkg_error)
   val0 = sig_integral/np.sqrt(bkg_integral) 
   unc0 = np.sqrt((sig_error/sig_integral)**2 + (0.5*bkg_error/bkg_integral)**2)*val0 
   print(val0,unc0)
   AMS = np.sqrt(2*((sig_integral+bkg_integral)*np.log(1 + (sig_integral/bkg_integral))-sig_integral))
   print(AMS)
      
   # test 
   bins_array = np.linspace(0,1,100,endpoint=True)
   hist1,bins1 = np.histogram(preds1, bins = bins_array, weights = wt1)
   hist0,bins0 = np.histogram(preds0, bins = bins_array, weights = wt0)
   total = np.sum(hist1)
   running_total = 0
   percentage = 0
   cut1 = 0
   plt.figure()
   plt.hist(preds1,bins=bins_array,weights=wt1,alpha = 0.5,label='signal',histtype='step')
   plt.hist(preds0,bins=bins_array,weights=wt0,alpha = 0.5,label='bkg',histtype='step')
   plt.ylim(0,100)
   plt.legend(loc='best')
   plt.savefig("plots/{}/{}/BDT_Score_Distribution".format(args.scenario,ch))

   for i, item in reversed(list(enumerate(hist1))):
       if percentage < 0.68:
          running_total += item
          percentage = running_total/total
          #if ch == "mmtt":
            #print(i,item,running_total,total,percentage)
       else:
          sig_area = sum(hist1[i:])
          bkg_area = sum(hist0[i:])
          AMS = np.sqrt(2*((sig_area+bkg_area)*np.log(1 + (sig_area/bkg_area))-sig_area))
          # print("Signal Yield: ",sig_area)
          # print("Background Yield: ",bkg_area)
          # print("S/$\sqrt{B}$: ", sig_area/np.sqrt(bkg_area))
          # print("AMS: ", AMS)
          # print()
          # break
   print("Finished")
   

  

