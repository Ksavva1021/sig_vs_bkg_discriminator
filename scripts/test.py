from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix

variables = ["mt_tot","pt_1","pt_2","met","m_vis","chi","n_jets","n_deepbjets","pt_tt","fabs(eta_1)","fabs(eta_2)","dR","dphi","mt_1","mt_2","mt_lep","met_dphi_1","met_dphi_2","jet_pt_1/pt_1","jet_pt_2/pt_2","jpt_1","jpt_2"]

int_df = Dataframe()
int_df.LoadRootFilesFromJson("json_selection/tt_2018_sig_interference.json",variables)
int_df.NormaliseWeights()
int_df.dataframe.loc[:,"y"] = 1
print "Interference Dataframe"
print int_df.dataframe.head()
print "Length =",len(int_df.dataframe)
print "Weight Normalisation =",int_df.dataframe.loc[:,"weights"].sum()


sig_df = Dataframe()
sig_df.LoadRootFilesFromJson("json_selection/tt_2018_sig.json",variables)
sig_df.NormaliseWeights()
sig_df.dataframe.loc[:,"y"] = 1
print "Signal Dataframe"
print sig_df.dataframe.head()
print "Length =",len(sig_df.dataframe)
print "Weight Normalisation =",sig_df.dataframe.loc[:,"weights"].sum()

bkg_df = Dataframe()
bkg_df.LoadRootFilesFromJson("json_selection/tt_2018_bkg.json",variables)
bkg_df.NormaliseWeights()
bkg_df.dataframe.loc[:,"y"] = 0
print "Background Dataframe"
print bkg_df.dataframe.head()
print "Length =",len(bkg_df.dataframe)
print "Weight Normalisation =",bkg_df.dataframe.loc[:,"weights"].sum()


#df_total = pd.concat([bkg_df.dataframe,sig_df.dataframe,int_df.dataframe],ignore_index=True, sort=False)
df_total = pd.concat([bkg_df.dataframe,sig_df.dataframe],ignore_index=True, sort=False)
print "Total Dataframe"
print df_total.head()
print "Length =",len(df_total)


train, test = train_test_split(df_total,test_size=0.5, random_state=42)

y_train = train.loc[:,"y"]
wt_train = train.loc[:,"weights"]
X_train = train.drop(["y","weights"],axis=1)

y_test = test.loc[:,"y"]
wt_test = test.loc[:,"weights"]
X_test = test.drop(["y","weights"],axis=1)

xgb_model = xgb.XGBClassifier(
                              learning_rate =0.1,
                              n_estimators=1000,
                              max_depth=5,
                              min_child_weight=1,
                              gamma=0,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              objective= 'binary:logistic',
                              nthread=4,
                              scale_pos_weight=1,
                              seed=27
                              )

#xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train, sample_weight=wt_train)
#xgb_model.fit(X_train, y_train)

probs = xgb_model.predict_proba(X_test)
preds = xgb_model.predict(X_test)

#preds = probs[:,1]

print preds

fpr, tpr, threshold = metrics.roc_curve(y_test, preds,sample_weight=wt_test)
roc_auc = roc_auc_score(y_test, preds, sample_weight=wt_test)

plt.figure(0)
plt.title('ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#
#t2 = test.loc[(test.loc[:,"y"]==2)] 
#t1 = test.loc[(test.loc[:,"y"]==1)]
#
#wt2 = t2.loc[:,"weights"]
#wt1 = t1.loc[:,"weights"]
#
#xt2 = t2.drop(["y","weights"],axis=1)
#xt1 = t1.drop(["y","weights"],axis=1)
#
#prob2 = xgb_model.predict_proba(xt2)
#preds2 = prob2[:,1]
#
#prob1 = xgb_model.predict_proba(xt1)
#preds1 = prob1[:,1]
#
#plt.figure(1)
#_, bins, _ = plt.hist(preds1, weights=wt1 ,bins=100, histtype='step', label='Non-interference')
#plt.xlabel("BDT Score")
#plt.xlim(bins[0], bins[-1])
#plt.legend(loc='best')
#plt.show()
#
#plt.figure(2)
#_, bins, _ = plt.hist(preds2, weights=wt2 ,bins=100, histtype='step', label='Interference')
#plt.xlabel("BDT Score")
#plt.xlim(bins[0], bins[-1])
#plt.legend(loc='best')
#plt.show()


plot_importance(xgb_model)
plt.show()

cm = confusion_matrix(y_test,preds,sample_weight=wt_test)
new_cm = []
for ind, i in enumerate(cm):
  norm = sum(i)
  new_cm.append([])
  for j in i:
    new_cm[ind].append(j/norm)

print new_cm
