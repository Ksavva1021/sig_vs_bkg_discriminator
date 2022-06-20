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
import warnings
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
  sig_df.LoadRootFilesFromJson("json_selection/{}_{}_sig.json".format(args.channel,args.year),variables)
  sig_df.NormaliseWeights()
    
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
  
  print "Signal Dataframe"
  print sig_df.dataframe.head()
  print "Length =",len(sig_df.dataframe)
  print "Weight Normalisation =",sig_df.dataframe.loc[:,"weights"].sum()
  print 
  
  #Combine dataframes
  df_total = sig_df.dataframe
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


DrawBDTScoreDistributions({"cat1":{"preds":preds0,"weights":wt0},"cat2":{"preds":preds1,"weights":wt1},"cat3":{"preds":preds2,"weights":wt2}})

# Feature importance
DrawFeatureImportance(xgb_model)

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

