from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawROCCurve, DrawBDTScoreDistributions, DrawFeatureImportance, DrawConfusionMatrix, DrawVarDistribution
import argparse
import pickle as pkl
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score



warnings.filterwarnings(action='ignore', category=DeprecationWarning)
parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='tt')
parser.add_argument('--year', help= 'Year to train BDT for', default='2018')
parser.add_argument('--load', help= 'Load dataframe from file',  action='store_true')
args = parser.parse_args()

# Set up variables to use for training
variables = ["q_1","q_2","q_3","q_4","dphi_12","dphi_13","dphi_14","dphi_23","dphi_24","dphi_34","dR_12","dR_13","dR_14","dR_23",
"dR_24","dR_34","pdgid_mother_1","pdgid_mother_2","pdgid_mother_3","pdgid_mother_4", "pt_1", "pt_2","pt_3","pt_4","mt_lep_12",
"mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34"]

abs_variables = ["dphi_12","dphi_13","dphi_14","dphi_23","dphi_24","dphi_34","dR_12","dR_13","dR_14","dR_23",
"dR_24","dR_34"]

# variables = ["pt_1", "pt_2","pt_3","pt_4","q_1","q_2","q_3","q_4","phi_1","phi_2","phi_3","phi_4","dphi_12","dphi_13","dphi_14","dphi_23","dphi_24","dphi_34","dR_12","dR_13","dR_14","dR_23",
# "dR_24","dR_34","eta_1","eta_2","eta_3","eta_4","mt_lep_12","mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34",
# "mvis_12","mvis_13","mvis_14","mvis_23","mvis_24","mvis_34","pdgid_mother_1","pdgid_mother_2","pdgid_mother_3","pdgid_mother_4"]

def label_target(df):

    if (df["pdgid_mother_1"] == df["pdgid_mother_2"] and df["pdgid_mother_3"] == df["pdgid_mother_4"] and np.sign(df["q_1"]) != np.sign(df["q_2"]) and np.sign(df["q_3"]) != np.sign(df["q_4"])):
        return 0
    if (df["pdgid_mother_1"] == df["pdgid_mother_3"] and df["pdgid_mother_2"] == df["pdgid_mother_4"] and np.sign(df["q_1"]) != np.sign(df["q_3"]) and np.sign(df["q_2"]) != np.sign(df["q_4"])):
        return 1
    if (df["pdgid_mother_1"] == df["pdgid_mother_4"] and df["pdgid_mother_2"] == df["pdgid_mother_3"] and np.sign(df["q_1"]) != np.sign(df["q_4"]) and np.sign(df["q_2"]) != np.sign(df["q_3"])):
        return 2
    else: 
        return 3
        


if not (args.load and os.path.isfile("dataframes/{}_{}.pkl".format(args.channel,args.year))):
  print "<< Making dataframe >>"  

  # Add signal dataframe
  sig_df = Dataframe()
  sig_df.LoadRootFilesFromJson("json_selection/{}_{}_sig_pair.json".format(args.channel,args.year),variables)
  
  sig_df.dataframe['y'] = sig_df.dataframe.apply(label_target,axis=1)
  sig_df.dataframe = sig_df.dataframe[sig_df.dataframe.y != 3]
  
  sig_df.dataframe['dphi_1234'] = abs(sig_df.dataframe['dphi_12']) +  abs(sig_df.dataframe['dphi_34'])
  sig_df.dataframe['dphi_1324'] = abs(sig_df.dataframe['dphi_13']) +  abs(sig_df.dataframe['dphi_24'])
  sig_df.dataframe['dphi_1423'] = abs(sig_df.dataframe['dphi_14']) +  abs(sig_df.dataframe['dphi_23'])
  
  sig_df.dataframe['os_12'] = sig_df.dataframe['q_1']/sig_df.dataframe['q_2']
  sig_df.dataframe['os_12'] = sig_df.dataframe['os_12'].apply(lambda x: 1 if (x < 0) else 0)
  sig_df.dataframe['os_13'] = sig_df.dataframe['q_1']/sig_df.dataframe['q_3']
  sig_df.dataframe['os_13'] = sig_df.dataframe['os_13'].apply(lambda x: 1 if (x < 0) else 0)
  sig_df.dataframe['os_14'] = sig_df.dataframe['q_1']/sig_df.dataframe['q_4']
  sig_df.dataframe['os_14'] = sig_df.dataframe['os_14'].apply(lambda x: 1 if (x < 0) else 0)
  
  sig_df.dataframe['pt_1_mt_lep_12'] = sig_df.dataframe['pt_1']/sig_df.dataframe['mt_lep_12']
  sig_df.dataframe['pt_1_mt_lep_13'] = sig_df.dataframe['pt_1']/sig_df.dataframe['mt_lep_13']
  sig_df.dataframe['pt_1_mt_lep_14'] = sig_df.dataframe['pt_1']/sig_df.dataframe['mt_lep_14']
  
  sig_df.dataframe['pt_2_mt_lep_12'] = sig_df.dataframe['pt_2']/sig_df.dataframe['mt_lep_12']
  sig_df.dataframe['pt_2_mt_lep_23'] = sig_df.dataframe['pt_2']/sig_df.dataframe['mt_lep_23']
  sig_df.dataframe['pt_2_mt_lep_24'] = sig_df.dataframe['pt_2']/sig_df.dataframe['mt_lep_24']
  
  sig_df.dataframe['pt_3_mt_lep_13'] = sig_df.dataframe['pt_3']/sig_df.dataframe['mt_lep_13']
  sig_df.dataframe['pt_3_mt_lep_23'] = sig_df.dataframe['pt_3']/sig_df.dataframe['mt_lep_23']
  sig_df.dataframe['pt_3_mt_lep_34'] = sig_df.dataframe['pt_3']/sig_df.dataframe['mt_lep_34']
  
  sig_df.dataframe['pt_4_mt_lep_14'] = sig_df.dataframe['pt_4']/sig_df.dataframe['mt_lep_14']
  sig_df.dataframe['pt_4_mt_lep_24'] = sig_df.dataframe['pt_4']/sig_df.dataframe['mt_lep_24']
  sig_df.dataframe['pt_4_mt_lep_34'] = sig_df.dataframe['pt_4']/sig_df.dataframe['mt_lep_34']
  
 
  for i in abs_variables: 
    sig_df.dataframe[i] = sig_df.dataframe[i].abs()
  
  
  print "Raw Dataframe"
  print sig_df.dataframe.head()
  print "Length =",len(sig_df.dataframe)
  
  df0 = Dataframe()
  df1 = Dataframe()
  df2 = Dataframe()
  
  df0.dataframe = sig_df.dataframe[sig_df.dataframe['y'] == 0].copy()
  df1.dataframe = sig_df.dataframe[sig_df.dataframe['y'] == 1].copy()
  df2.dataframe = sig_df.dataframe[sig_df.dataframe['y'] == 2].copy()
  
  df0.NormaliseWeights()
  print "Dataframe Target = 0"
  print df0.dataframe.head()
  print "Length =",len(df0.dataframe)
  print "Weight Normalisation =",df0.dataframe.loc[:,"weights"].sum()
  
  df1.NormaliseWeights()
  print "Dataframe Target = 1"
  print df1.dataframe.head()
  print "Length =",len(df1.dataframe)
  print "Weight Normalisation =",df1.dataframe.loc[:,"weights"].sum()
  
  
  df2.NormaliseWeights()
  print "Dataframe Target = 2"
  print df2.dataframe.head()
  print "Length =",len(df2.dataframe)
  print "Weight Normalisation =",df2.dataframe.loc[:,"weights"].sum()

  
  # Combine dataframes
  df_total = pd.concat([df0.dataframe,df1.dataframe,df2.dataframe],ignore_index=True, sort=False)
  #df_total.drop(['pdgid_mother_1','pdgid_mother_2','pdgid_mother_3','pdgid_mother_4'],inplace = True,axis=1)
  df_total.drop(['pdgid_mother_1','pdgid_mother_2','pdgid_mother_3','pdgid_mother_4','q_1','q_2','q_3','q_4',
  "pt_1", "pt_2","pt_3","pt_4","mt_lep_12","mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34"],
  inplace = True,axis=1)
  print "Total Dataframe"
  print df_total.head()
  print "Length =",len(df_total)
 
  df_total.to_pickle("dataframes/multiclass_{}_{}.pkl".format(args.channel,args.year))
else:
  print "<< Loading in dataframe >>"
  df_total = pd.read_pickle("dataframes/multiclass_{}_{}.pkl".format(args.channel,args.year))
  
DrawVarDistribution(df_total,3,'pt_3_mt_lep_34',0,100,"pc_pt_3_mt_lep_34_distribution")

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


# A parameter grid for XGBoost
       
# pipe_xgb = Pipeline([('XGB', xgb.XGBClassifier(random_state=42))])

# param_range = [1, 2, 3, 4, 5, 6]
# param_range_fl = [1.0, 0.5, 0.1]
# n_estimators = [50,100,150]
# learning_rates = [.1,.2,.3]

# xgb_param_grid = [{'XGB__learning_rate': learning_rates,
                    # 'XGB__max_depth': param_range,
                    # 'XGB__min_child_weight': param_range[:2],
                    # 'XGB__subsample': param_range_fl,
                    # 'XGB__n_estimators': n_estimators}]

# xgb_grid_search = GridSearchCV(estimator=pipe_xgb,
        # param_grid=xgb_param_grid,
        # scoring='accuracy',
        # cv=3)

# xgb_grid_search.fit(X_train, y_train, wt_train)

# print('Test Accuracy: {}'.format(xgb_grid_search.score(X_test,y_test)))
# print('Best Params: {}'.format(xgb_grid_search.best_params_))
# Best Params: {'XGB__min_child_weight': 1, 'XGB__subsample': 0.5, 'XGB__learning_rate': 0.2, 'XGB__n_estimators': 100, 'XGB__max_depth': 1}



xgb_model = xgb.XGBClassifier(
                             learning_rate =0.2,
                             n_estimators=100,
                             max_depth=1,
                             min_child_weight=1,
                             subsample=0.5
                             )

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

DrawBDTScoreDistributions({"cat1":{"preds":preds0,"weights":wt0},"cat2":{"preds":preds1,"weights":wt1},"cat3":{"preds":preds2,"weights":wt2}})

# Feature importance
DrawFeatureImportance(xgb_model,"gain","pc_feature_importance_gain")
DrawFeatureImportance(xgb_model,"weight","pc_feature_importance_weight")

# Confusion matrix

preds = xgb_model.predict(X_test)

y_test_np = np.array(y_test)
score = 0
for i in range(len(preds)):
    if preds[i] == y_test_np[i]:
        score += 1

print "<< Score:",score,"/",len(preds)," >>"
probs = xgb_model.predict_proba(X_test)
DrawConfusionMatrix(y_test,preds,wt_test,["cat 1","cat 2","cat 3"])



