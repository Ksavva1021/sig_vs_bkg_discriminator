from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawROCCurve, DrawBDTScoreDistributions, DrawFeatureImportance
import argparse
import pickle as pkl
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='tt')
parser.add_argument('--year', help= 'Year to train BDT for', default='2018')
parser.add_argument('--load', help= 'Load dataframe from file',  action='store_true')
args = parser.parse_args()

# Set up variables to use for training
#variables = ["mt_tot","pt_1","pt_2","met","m_vis","n_deepbjets","pt_tt","mt_lep","met_dphi_1","met_dphi_2","jet_pt_1/pt_1","jet_pt_2/pt_2"]

variables = ["pt_1", "pt_2","pt_3","pt_4","E_1","E_2","E_3","E_4","eta_1","eta_2","eta_3","eta_4",
"phi_1","phi_2","phi_3","phi_4","dphi_12","dphi_13","dphi_14","dphi_23","dphi_24","dphi_34","dR_12","dR_13","dR_14","dR_23",
"dR_24","dR_34","mt_1","mt_2","mt_3","mt_4","mt_lep_12","mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34",
"mvis_12","mvis_13","mvis_14","mvis_23","mvis_24","mvis_34","mvis_min_dphi_1", "mvis_min_dphi_2","mvis_min_dR_1", "mvis_min_dR_2", 
"mvis_min_sum_dphi_1","mvis_min_sum_dphi_2","mvis_min_sum_dR_1","mvis_min_sum_dR_2","pt_min_dphi_1","pt_min_dphi_2","pt_min_dR_1", 
"pt_min_dR_2","pt_min_sum_dphi_1","pt_min_sum_dphi_2","pt_min_sum_dR_1","pt_min_sum_dR_2","p_min_dphi_1","p_min_dphi_2","p_min_sum_dphi_1",
"p_min_sum_dphi_2","p_min_dR_1","p_min_dR_2","p_min_sum_dR_1","p_min_sum_dR_2","q_1","q_2","q_3","q_4"]

abs_variables = ["dphi_12","dphi_13","dphi_14","dphi_23","dphi_24","dphi_34","dR_12","dR_13","dR_14","dR_23",
"dR_24","dR_34"]

#variables = ["mvis_min_dphi_1"]

if not (args.load and os.path.isfile("dataframes/{}_{}.pkl".format(args.channel,args.year))):
  print "<< Making dataframe >>"  

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
  df_total = pd.concat([bkg_df.dataframe,sig_df.dataframe],ignore_index=True, sort=False)
  
  df_total['pt_1_mt_lep_12'] = df_total['pt_1']/df_total['mt_lep_12']
  df_total['pt_1_mt_lep_13'] = df_total['pt_1']/df_total['mt_lep_13']
  df_total['pt_1_mt_lep_14'] = df_total['pt_1']/df_total['mt_lep_14']
  
  df_total['pt_2_mt_lep_12'] = df_total['pt_2']/df_total['mt_lep_12']
  df_total['pt_2_mt_lep_23'] = df_total['pt_2']/df_total['mt_lep_23']
  df_total['pt_2_mt_lep_24'] = df_total['pt_2']/df_total['mt_lep_24']
  
  df_total['pt_3_mt_lep_13'] = df_total['pt_3']/df_total['mt_lep_13']
  df_total['pt_3_mt_lep_23'] = df_total['pt_3']/df_total['mt_lep_23']
  df_total['pt_3_mt_lep_34'] = df_total['pt_3']/df_total['mt_lep_34']
  
  df_total['pt_4_mt_lep_14'] = df_total['pt_4']/df_total['mt_lep_14']
  df_total['pt_4_mt_lep_24'] = df_total['pt_4']/df_total['mt_lep_24']
  df_total['pt_4_mt_lep_34'] = df_total['pt_4']/df_total['mt_lep_34']
  
  for i in abs_variables: 
    df_total[i] = df_total[i].abs()
    
  df_total.drop(["pt_1", "pt_2","pt_3","pt_4","mt_lep_12","mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34"],
  inplace = True,axis=1)
  
  print "Total Dataframe"
  print df_total.head()
  print "Length =",len(df_total)
  
  df_total.to_pickle("dataframes/binary_{}_{}.pkl".format(args.channel,args.year))
else:
  print "<< Loading in dataframe >>"
  df_total = pd.read_pickle("dataframes/binary_{}_{}.pkl".format(args.channel,args.year))

# Set up train and test separated dataframes
train, test = train_test_split(df_total,test_size=0.5, random_state=42)

y_train = train.loc[:,"y"]
wt_train = train.loc[:,"weights"]
X_train = train.drop(["y","weights"],axis=1)

y_test = test.loc[:,"y"]
wt_test = test.loc[:,"weights"]
X_test = test.drop(["y","weights"],axis=1)


# def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       # model, param_grid, cv=10, scoring_fit='roc_auc',
                       # do_probabilities = False):
    # gs = GridSearchCV(
        # estimator=model,
        # param_grid=param_grid, 
        # cv=cv, 
        # n_jobs=-1, 
        # scoring=scoring_fit,
        # verbose=2
    # )
    # fitted_model = gs.fit(X_train_data, y_train_data, wt_train)
    
    # if do_probabilities:
      # pred = fitted_model.predict_proba(X_test_data)
    # else:
      # pred = fitted_model.predict(X_test_data)
    
    # return fitted_model, pred

# model = xgb.XGBClassifier()
# param_grid = {
    # 'learning_rate': [.1,.2],
    # 'n_estimators': [100, 1000],
    # 'colsample_bytree': [0.7, 0.8],
    # 'max_depth': [1,2,3],
    # 'reg_alpha': [1.1, 1.3],
    # 'reg_lambda': [1.1,1.3],
    # 'subsample': [0.7, 0.9],
    # 'min_child_weight': [3,5]
# }

# model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                 # param_grid, cv=5)

# print(model.best_score_)
# print(model.best_params_)


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

xgb_model.fit(X_train, y_train, sample_weight=wt_train)

pkl.dump(xgb_model,open("BDTs/binary_{}_{}.pkl".format(args.channel,args.year), "wb"))

print "<< Training finished >>"

# Test output
# ROC curve
probs = xgb_model.predict_proba(X_test)
preds = probs[:,1]
DrawROCCurve(y_test,preds,wt_test)

# BDT scores
t0 = test.loc[(test.loc[:,"y"]==0)]
t1 = test.loc[(test.loc[:,"y"]==1)]
wt0 = t0.loc[:,"weights"]
wt1 = t1.loc[:,"weights"]
xt0 = t0.drop(["y","weights"],axis=1)
xt1 = t1.drop(["y","weights"],axis=1)
probs0 = xgb_model.predict_proba(xt0)
probs1 = xgb_model.predict_proba(xt1)
preds0 = probs0[:,1]
preds1 = probs1[:,1]

DrawBDTScoreDistributions({"background":{"preds":preds0,"weights":wt0},"signal":{"preds":preds1,"weights":wt1}})

# Feature importance
DrawFeatureImportance(xgb_model)
