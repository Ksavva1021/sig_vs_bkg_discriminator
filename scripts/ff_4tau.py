from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawROCCurve, DrawBDTScoreDistributions, DrawFeatureImportance, DrawVarDistribution
import argparse
import pickle as pkl
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mttt')
parser.add_argument('--year', help= 'Year to train BDT for', default='2018')
parser.add_argument('--load', help= 'Load dataframe from file',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
args = parser.parse_args()

variables = [
             "pt_1", "pt_2","pt_3","pt_4"
#             "fabs(dphi_12)","fabs(dphi_13)","fabs(dphi_14)","fabs(dphi_23)","fabs(dphi_24)","fabs(dphi_34)",
#             "fabs(dR_12)","fabs(dR_13)","fabs(dR_14)","fabs(dR_23)","fabs(dR_24)","fabs(dR_34)",
#             "mt_1","mt_2","mt_3","mt_4",
#             "mt_lep_12","mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34",
#             "mvis_12","mvis_13","mvis_14","mvis_23","mvis_24","mvis_34",
#             "q_1","q_2","q_3","q_4",
#             "pt_tt_12","pt_tt_13","pt_tt_14","pt_tt_23","pt_tt_24","pt_tt_34",
#             "n_jets","n_bjets",
             ]

bkg_df = Dataframe()
bkg_df.LoadRootFilesFromJson("json_selection/ff_data_fail.json",variables,quiet=(args.verbosity<2))
print bkg_df.dataframe
