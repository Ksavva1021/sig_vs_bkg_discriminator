import uproot
import os
import json
import pandas
import pickle
import argparse
import numpy as np
import xgboost as xgb
from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.reweighter import reweighter

# python scripts/add_ff_to_trees.py --input_location=/vols/cms/gu18/Offline/output/4tau/2018_1907 --output_location="./" --filename=TauB_mmtt_2018.root --channel=mmtt --year=2018 --splitting=100000 --offset=0

parser = argparse.ArgumentParser()
parser.add_argument('--input_location',help= 'Name of input location (not including file name)', default='/vols/cms/gu18/Offline/output/4tau/2018_1907')
parser.add_argument('--output_location',help= 'Name of output location (not including file name)', default='./')
parser.add_argument('--filename',help= 'Name of file', default='TauB_tttt_2018.root')
parser.add_argument('--channel',help= 'Name of channel', default='tttt')
parser.add_argument('--year',help= 'Name of year', default='2018')
parser.add_argument('--splitting',help= 'Number of events per task', default='100000')
parser.add_argument('--offset',help= 'Offset of job', default='0')
args = parser.parse_args()

tree = uproot.open(args.input_location+'/'+args.filename, localsource=uproot.FileSource.defaults)["ntuple"]
models_to_add = {
                 "wt_ff_None_vvvloose_raw_3"   :"ff_mmtt_None_vvvloose_raw_ff_3.pkl",
                 "wt_ff_None_vvvloose_raw_4"   :"ff_mmtt_None_vvvloose_raw_ff_4.pkl",
                 "wt_ff_None_vvvloose_raw_34"  :"ff_mmtt_None_vvvloose_raw_ff_34.pkl",
                 "wt_ff_None_vvvloose_corr_3"  :"ff_mmtt_None_vvvloose_correction_3.pkl",
                 "wt_ff_None_vvvloose_corr_4"  :"ff_mmtt_None_vvvloose_correction_4.pkl",
                 "wt_ff_None_vvvloose_alt_3"   :"ff_mmtt_None_vvvloose_alternative_ff_3.pkl",
                 "wt_ff_None_vvvloose_alt_4"   :"ff_mmtt_None_vvvloose_alternative_ff_4.pkl"
                 }


k = 0
for small_tree in tree.iterate(entrysteps=int(args.splitting)):
  if k == int(args.offset) or int(args.offset)==-1:
    df = Dataframe()
    df.dataframe = pandas.DataFrame.from_dict(small_tree)

    for key, val in models_to_add.items():
      xgb_model = pickle.load(open("BDTs/{}".format(val), "rb"))       
      new_df = Dataframe()
      features = list(xgb_model.column_names)
      features.remove("yr_2016")
      features.remove("yr_2017")
      features.remove("yr_2018")
      new_df.dataframe = df.dataframe.copy(deep=False)
      new_df.SelectColumns(features)
      new_df.dataframe.loc[:,"yr_2016"] = (args.year=="2016")
      new_df.dataframe.loc[:,"yr_2017"] = (args.year=="2017")
      new_df.dataframe.loc[:,"yr_2018"] = (args.year=="2018")
      new_df.dataframe = new_df.dataframe.reindex(list(xgb_model.column_names), axis=1)
 
      df.dataframe.loc[:,key] = xgb_model.predict_reweights(new_df.dataframe) 
      del new_df

    df.WriteToRoot(args.output_location+'/'+args.filename.replace(".root","_"+str(k)+".root"))
    del df, small_tree
    if int(args.offset)!=-1: break
  k += 1
print "Finished processing"
