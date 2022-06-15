import uproot
import os
import json
import pandas
import pickle
import argparse
import numpy as np
import xgboost as xgb
from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe

# python scripts/add_to_trees.py --input_location=/vols/cms/gu18/Offline/output/MSSM/vlq_2018_pre_bdt --output_location="./" --filename=TauB_tt_2018.root --channel=tt --year=2018 --splitting=100000 --offset=0

parser = argparse.ArgumentParser()
parser.add_argument('--input_location',help= 'Name of input location (not including file name)', default='/vols/cms/gu18/Offline/output/MSSM/vlq_2018_bkg_data/')
parser.add_argument('--output_location',help= 'Name of output location (not including file name)', default='./')
parser.add_argument('--filename',help= 'Name of file', default='TauB_tt_2018.root')
parser.add_argument('--channel',help= 'Name of channel', default='tt')
parser.add_argument('--year',help= 'Name of year', default='2018')
parser.add_argument('--splitting',help= 'Number of events per task', default='100000')
parser.add_argument('--offset',help= 'Offset of job', default='0')
args = parser.parse_args()

tree = uproot.open(args.input_location+'/'+args.filename, localsource=uproot.FileSource.defaults)["ntuple"]

k = 0
for small_tree in tree.iterate(entrysteps=int(args.splitting)):
  if k == int(args.offset) or int(args.offset)==-1:
    df = Dataframe()
    df.dataframe = pandas.DataFrame.from_dict(small_tree)

    xgb_model_1 = pickle.load(open("BDTs/multiclass_{}_{}.pkl".format(args.channel,args.year), "rb"))       
    new_df_1 = Dataframe()
    new_df_1.dataframe = df.dataframe.copy(deep=False)
    new_df_1.SelectColumns(xgb_model_1.get_booster().feature_names)
    df.dataframe.loc[:,"vlq_b_bdt_multiclass"] = xgb_model_1.predict_proba(new_df_1.dataframe)[:,0]  
    df.dataframe.loc[:,"vlq_s_bdt_multiclass"] = xgb_model_1.predict_proba(new_df_1.dataframe)[:,1]
    df.dataframe.loc[:,"vlq_i_bdt_multiclass"] = xgb_model_1.predict_proba(new_df_1.dataframe)[:,2]
    del new_df_1

    xgb_model_2 = pickle.load(open("BDTs/binary_{}_{}.pkl".format(args.channel,args.year), "rb"))
    new_df_2 = Dataframe()
    new_df_2.dataframe = df.dataframe.copy(deep=False)
    new_df_2.SelectColumns(xgb_model_2.get_booster().feature_names)
    df.dataframe.loc[:,"vlq_s_bdt_binary"] = xgb_model_2.predict_proba(new_df_2.dataframe)[:,1]
    del new_df_2

    xgb_model_3 = pickle.load(open("BDTs/binary_{}_{}_interference.pkl".format(args.channel,args.year), "rb"))
    new_df_3 = Dataframe()
    new_df_3.dataframe = df.dataframe.copy(deep=False)
    new_df_3.SelectColumns(xgb_model_3.get_booster().feature_names)
    df.dataframe.loc[:,"vlq_i_bdt_binary"] = xgb_model_3.predict_proba(new_df_3.dataframe)[:,1]
    del new_df_3

    df.WriteToRoot(args.output_location+'/'+args.filename.replace(".root","_"+str(k)+".root"))
    
    del df, small_tree
    if int(args.offset)!=-1: break
  k += 1
print "Finished processing"
