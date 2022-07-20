import xgboost as xgb
import copy
import pandas as pd
import numpy as np
import itertools

class p_xgboost(xgb.XGBClassifier):

  def __init__(self):
    self.__p_weights__ = [] # example format [[100,100,4],[200,100,1],....]
    self.p_names = [] # [mphi, mA]

  def set_params_pbdt(self,params_dict):
    if "p_weights" in params_dict.keys():
      self.__p_weights__ = params_dict["p_weights"]
      xgb_params_dict = copy.deepcopy(params_dict)
      del xgb_params_dict["p_weights"]
      self.set_params(**xgb_params_dict)
    else:
      self.set_params(**params_dict)

  def fit_pbdt(self, x, Y, sample_weight=None): 
    if sample_weight == None:
      sample_weight = pd.Series(np.ones(len(x)))
    df = pd.concat([x,Y,sample_weight],axis=1)
    df = df.loc[(df.loc[:,Y.name]==1)]

    unique_entries = []
    for n in p_names:
      unique_entries.append(df[n].unique):

    iterate_over = list(itertools.product(*unique_entries))

    for it_ind, it in enumerate(iterate_over):
      df_temp = copy.deepcopy(df)
      for ind,p in enumerate(it):
        df_temp = df.loc[(df.loc[:,self.p_names[ind]]==p)]


      for p_weight in self.__p_weights__:
        if p_weight[:-1] == it:
          df_temp.loc[:,sample_weight.name] = p_weight[-1]*df_temp.loc[:,sample_weight.name]

    if it_ind == 0:
      total_df = df_temp.copy(deep=True)
    else:
      total_df = pd.concat([total_df,df_temp], ignore_index=True, sort=False)


    Y = total_df.loc[:,Y.name]
    sample_weight = total_df.loc[:,sample_weight.name]
    x = total_df.drop([Y.name,sample_weight.name],axis=1) 

    self.fit(x,Y,sample_weight)

     
