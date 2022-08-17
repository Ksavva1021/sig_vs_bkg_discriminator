from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawClosurePlots, DrawReweightPlots, DrawColzReweightPlots, DrawConfusionMatrix
from UserCode.sig_vs_bkg_discriminator import reweighter
from collections import OrderedDict
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import json
import xgboost as xgb
import itertools

#python scripts/ff_4tau_venn.py --channel=mmtt --load_dataframes --collect_scan_batch

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mttt')
parser.add_argument('--pass_wp',help= 'Pass WP for fake factors', default='vvvloose')
parser.add_argument('--fail_wp',help= 'Channel to train BDT for', default=None)
parser.add_argument('--load_dataframes', help= 'Load dataframe from file',  action='store_true')
parser.add_argument('--load_models', help= 'Load model from file',  action='store_true')
parser.add_argument('--load_hyperparameters', help= 'Load hyperparameters from file',  action='store_true')
parser.add_argument('--scan', help= 'Do hyperparameter scan',  action='store_true')
parser.add_argument('--scan_batch', help= 'Do hyperparameter scan on the batch',  action='store_true')
parser.add_argument('--collect_scan_batch', help= 'Collect hyperparameter scan on the batch',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
args = parser.parse_args()


############# Variables needed #################

years = ["2018"]

fitting_variables = [
             "pt_1", "pt_2","pt_3","pt_4",
             "fabs(dphi_12)","fabs(dphi_13)","fabs(dphi_14)","fabs(dphi_23)","fabs(dphi_24)","fabs(dphi_34)",
             "fabs(dR_12)","fabs(dR_13)","fabs(dR_14)","fabs(dR_23)","fabs(dR_24)","fabs(dR_34)",
             "mt_1","mt_2","mt_3","mt_4",
             "mt_lep_12","mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34",
             "mvis_12","mvis_13","mvis_14","mvis_23","mvis_24","mvis_34",
             "pt_tt_12","pt_tt_13","pt_tt_14","pt_tt_23","pt_tt_24","pt_tt_34",
             "n_jets","n_bjets",
             ]

scoring_variables = fitting_variables + ["yr_2016","yr_2017","yr_2018"]

reweight_plot_variables = [
                           ["pt_1",(50,0,250)],
                           ["pt_2",(50,0,250)],
                           ["pt_3",(50,0,250)],
                           ["pt_4",(50,0,250)],
                           ]

closure_plot_variables = [
                          ["mvis_12",(60,0,300)],
                          ["mvis_13",(60,0,300)],
                          ["mvis_14",(60,0,300)],
                          ["mvis_23",(60,0,300)],
                          ["mvis_24",(60,0,300)],
                          ["mvis_34",(60,0,300)],
                          ["pt_1",(50,0,250)],
                          ["pt_2",(50,0,250)],
                          ["pt_3",(50,0,250)],
                          ["pt_4",(50,0,250)],
                          ]

param_grid = {
              "n_estimators":[100,200,300,400],
              "max_depth":[3,4,5,6],
              "learning_rate":[0.06,0.08,0.1,0.12],
              "min_samples_leaf": [300,400,500,600],
              }


############### Functions #######################


def PrintDatasetSummary(name,dataset):
  print name
  print dataset.head(10)
  print "Total Length = {}, Total Sum of Weights = {}".format(len(dataset),dataset.loc[:,"weights"].sum())
  print "Average Weights = {}".format(dataset.loc[:,"weights"].sum()/len(dataset))
  print ""

def AddColumnsToDataframe(df, columns, vals):
  for ind, i in enumerate(columns): 
    df.loc[:,i] = val[ind]
  return df.reindex(sorted(df.columns), axis=1)

def MakeSelFromList(lst):
  if "" in lst: lst.remove("")
  lst.sort()
  sel = ""
  for ind, l in enumerate(lst):
    sel += l
    if ind != len(lst)-1:
      sel += " && "
  return sel 


############# Do Reweighting ####################

print "<<<< Doing fake factor reweighting >>>>"

### setup sideband and signal regions ###

SIDEBAND = "(q_sum!=0)"
SIGNAL = "(q_sum==0)"

### setup pass and fail ###

if args.fail_wp == None:
  FAIL = "(deepTauVsJets_iso_X>=0 && deepTauVsJets_{}_X==0)".format(args.pass_wp)
else:
  FAIL = "(deepTauVsJets_{}_X && deepTauVsJets_{}_X==0)".format(args.fail_wp,args.pass_wp)

PASS = "(deepTauVsJets_{}_X==1)".format(args.pass_wp)


### Get all combination you need to loop through ###

lst_n = []
for ind,i in enumerate(args.channel):
  if i == "t":
    lst_n.append(ind+1)

total_keys = []
for ind,i in enumerate(lst_n):
  for k in list(itertools.combinations(lst_n, ind+1)):
    total_keys.append(k)

### Begin load and training loop ###

datasets_created = []

for t_ff in total_keys:

  print "<< Running for {} >>".format(t_ff)

  ### get selection strings ###
  f_i = ""
  p_i = ""
  for ind_t, t in enumerate(t_ff):
    f_i += FAIL.replace("X",str(t))
    p_i += PASS.replace("X",str(t))
    if ind_t != len(t_ff)-1: 
      f_i += " && "
      p_i += " && "

  o_ff = tuple(set(lst_n) - set(t_ff))
  f_o = ""
  p_o = ""
  for ind_o, o in enumerate(o_ff):
    f_o += FAIL.replace("X",str(o))
    p_o += PASS.replace("X",str(o))
    if ind_o != len(o_ff)-1:
      f_o += " && "
      p_o += " && " 

  sel = OrderedDict()

  sel["Raw F_{F}"] = {
                      "pass":MakeSelFromList([SIDEBAND,p_i,p_o]),
                      "fail":MakeSelFromList([SIDEBAND,f_i,p_o])
                      }
  
  if len(t_ff) != args.channel.count("t"):
    sel["Alternative F_{F}"] = {
                                "pass":MakeSelFromList([SIDEBAND,p_i,f_o]),
                                "fail":MakeSelFromList([SIDEBAND,f_i,f_o])
                                }
    sel["Correction"] = {
                         "pass":MakeSelFromList([SIGNAL,p_i,f_o]),
                         "fail":MakeSelFromList([SIGNAL,f_i,f_o])
                         }


  ### begin loop through different reweighting regions ###

  for rwt_key, rwt_val in sel.iteritems():
  
    ### get dataframes ##

    print "<< Getting dataframes >>"

    pf_df = {}

    for pf_key, pf_val in rwt_val.items(): 

      dataset_name = "{}_{}".format(args.channel,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_"))
  
      if (not args.load_dataframes) and (dataset_name not in datasets_created):

        print "<< Making {} {} dataframe >>".format(pf_key,rwt_key)

        replace = {"SELECTION":pf_val}

        pf_df[pf_key] = Dataframe()
        pf_df[pf_key].LoadRootFilesFromJson("json_selection/ff_4tau_venn/ff_data_{}.json".format(args.channel),fitting_variables,quiet=(args.verbosity<2),replace=replace)
        pf_df[pf_key].TrainTestSplit()
        if args.verbosity > 0: PrintDatasetSummary("{} {} dataframe".format(pf_key,rwt_key),pf_df[pf_key].dataframe)
  
        pf_df[pf_key].dataframe.to_pickle("dataframes/{}_dataframe.pkl".format(dataset_name))
        datasets_created.append(dataset_name)

      else:

        print "<< Loading {} {} dataframe >>".format(pf_key,rwt_key)

        pf_df[pf_key] = Dataframe()
        pf_df[pf_key].dataframe = pd.read_pickle("dataframes/{}_dataframe.pkl".format(dataset_name))
        if args.verbosity > 0: PrintDatasetSummary("{} {} dataframe".format(pf_key,rwt_key),pf_df[pf_key].dataframe) 


    ### split datasets ###

    train_fail, wt_train_fail, test_fail, wt_test_fail = pf_df["fail"].SplitTrainTestXWts()
    train_pass, wt_train_pass, test_pass, wt_test_pass = pf_df["pass"].SplitTrainTestXWts()

    ### load in alternative FF for correction ###

    alt_model_name = "{}_{}_{}_{}_{}".format(args.channel,str(args.fail_wp),args.pass_wp,"alternative_ff",str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ",""))

    if "Correction" in rwt_key:
      rwter_alt = pkl.load(open("BDTs/ff_{}.pkl".format(alt_model_name), "rb"))
      wt_train_fail = np.multiply(wt_train_fail,rwter_alt.predict_reweights(train_fail))
      wt_test_fail = np.multiply(wt_test_fail,rwter_alt.predict_reweights(test_fail))

    ### run training ###

    model_name = "{}_{}_{}_{}_{}".format(args.channel,str(args.fail_wp),args.pass_wp,rwt_key.lower().replace(" ","_").replace("_{","").replace("}",""),str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ",""))

    if not args.load_models:
    
      print "<< Running reweighting training >>"
      
      rwter = reweighter.reweighter()
      
      if args.scan:

        rwter.grid_search(train_fail, train_pass, wt_train_fail ,wt_train_pass, test_fail, test_pass, wt_test_fail ,wt_test_pass, param_grid=param_grid, scoring_variables=scoring_variables)
        with open('hyperparameters/ff_hp_{}.json'.format(model_name), 'w') as outfile: json.dump(rwter.dump_hyperparameters(), outfile)

      elif args.scan_batch:

        rwter.grid_search_batch(model_name,train_fail, train_pass, wt_train_fail ,wt_train_pass, test_fail, test_pass, wt_test_fail ,wt_test_pass, param_grid=param_grid, scoring_variables=scoring_variables)
 
      elif args.collect_scan_batch:

        rwter.collect_grid_search_batch(model_name)
        with open('hyperparameters/ff_hp_{}.json'.format(model_name), 'w') as outfile: json.dump(rwter.dump_hyperparameters(), outfile)
        rwter.norm_and_fit(train_fail, train_pass, wt_train_fail ,wt_train_pass)

      else:

        if args.load_hyperparameters:
          with open('hyperparameters/ff_hp_{}.json'.format(model_name)) as json_file: params = json.load(json_file)
          rwter.set_params(params)
        rwter.norm_and_fit(train_fail, train_pass, wt_train_fail ,wt_train_pass)
    
      if not args.scan_batch: pkl.dump(rwter,open("BDTs/ff_{}.pkl".format(model_name), "wb"))
    
    else:
    
      print "<< Loading raw ff training >>"
    
      rwter = pkl.load(open("BDTs/ff_{}.pkl".format(model_name), "rb"))

    ### drawing reweights ###
    
    if not args.scan_batch:

      print "<< Producting reweight plots >>"
   
      reweights_train = pd.Series(rwter.predict_reweights(train_fail),name="reweights")
      reweights_test = pd.Series(rwter.predict_reweights(test_fail),name="reweights")
      
      for yr in years:
        for var in reweight_plot_variables:
          var_name = var[0]
          
          var_train = train_fail.loc[:,var_name]
          DrawReweightPlots(var_train, reweights_train, var_name, rwt_key, plot_name="reweight_plot_train_{}_{}".format(var_name,model_name), title_left=args.channel, title_right=yr)
          DrawColzReweightPlots(var_train, reweights_train, wt_train_fail.copy(), var_name, rwt_key, plot_name="reweight_colz_plot_train_{}_{}".format(var_name,model_name), title_left=args.channel, title_right=yr)
      
          var_test = test_fail.loc[:,var_name]
          DrawReweightPlots(var_test, reweights_test, var_name, rwt_key, plot_name="reweight_plot_test_{}_{}".format(var_name,model_name), title_left=args.channel, title_right=yr)
          DrawColzReweightPlots(var_test, reweights_test, wt_test_fail.copy(), var_name, rwt_key, plot_name="reweight_colz_plot_test_{}_{}".format(var_name,model_name), title_left=args.channel, title_right=yr)
     
          var_all = pd.concat([var_train,var_test], ignore_index=True, sort=False)
          reweights_all = pd.concat([reweights_train,reweights_test], ignore_index=True, sort=False)
          wt_all = pd.concat([wt_train_fail,wt_test_fail], ignore_index=True, sort=False)
          DrawReweightPlots(var_all, reweights_all, var_name, rwt_key, plot_name="reweight_plot_all_{}_{}".format(var_name,model_name), title_left=args.channel, title_right=yr)
          DrawColzReweightPlots(var_all, reweights_all, wt_all, var_name, rwt_key, plot_name="reweight_colz_plot_all_{}_{}".format(var_name,model_name), title_left=args.channel, title_right=yr)
 
      ### closure plots in individual regions ###
      
      print "<< Producting closure plots in individual regions>>"
   
      xwt_train_fail = pd.concat([train_fail,np.multiply(wt_train_fail,reweights_train)],axis=1)
      xwt_train_pass = pd.concat([train_pass,wt_train_pass],axis=1)
      xwt_test_fail = pd.concat([test_fail,np.multiply(wt_test_fail,reweights_test)],axis=1)
      xwt_test_pass = pd.concat([test_pass,wt_test_pass],axis=1)
      
      for yr in years:
        for var in closure_plot_variables:
          var_name = var[0]
      
          cut_fail_test = xwt_test_fail[(xwt_test_fail.loc[:,"yr_"+yr] == 1)]
          cut_pass_test = xwt_test_pass[(xwt_test_pass.loc[:,"yr_"+yr] == 1)]
          cut_fail_test = cut_fail_test.loc[:,["weights",var[0]]]
          cut_pass_test = cut_pass_test.loc[:,["weights",var[0]]]
          print "KS test {}:".format(var_name),rwter.KS(cut_fail_test, cut_pass_test, cut_fail_test.loc[:,"weights"], cut_pass_test.loc[:,"weights"], columns=[var[0]])[0]
          DrawClosurePlots(cut_fail_test, cut_pass_test, "Reweight x fail", "pass", var[0], var[1], plot_name="closure_plot_test_{}_{}".format(var_name,model_name), title_left=args.channel, title_right=yr)
      
          cut_fail_train = xwt_train_fail[(xwt_train_fail.loc[:,"yr_"+yr] == 1)]
          cut_pass_train = xwt_train_pass[(xwt_train_pass.loc[:,"yr_"+yr] == 1)]
          cut_fail_train = cut_fail_train.loc[:,["weights",var[0]]]
          cut_pass_train = cut_pass_train.loc[:,["weights",var[0]]]
          print "KS train {}:".format(var_name),rwter.KS(cut_fail_train, cut_pass_train, cut_fail_train.loc[:,"weights"], cut_pass_train.loc[:,"weights"], columns=[var[0]])[0]
          DrawClosurePlots(cut_fail_train, cut_pass_train, "Reweight x fail", "pass", var[0], var[1], plot_name="closure_plot_train_{}_{}".format(var_name,model_name), title_left=args.channel, title_right=yr)

     
          cut_fail_all = pd.concat([cut_fail_train,cut_fail_test], ignore_index=True, sort=False)
          cut_pass_all = pd.concat([cut_pass_train,cut_pass_test], ignore_index=True, sort=False)
          print "KS all {}:".format(var_name),rwter.KS(cut_fail_all, cut_pass_all, cut_fail_all.loc[:,"weights"], cut_pass_all.loc[:,"weights"], columns=[var[0]])[0]
          DrawClosurePlots(cut_fail_all, cut_pass_all, "Reweight x fail", "pass", var[0], var[1], plot_name="closure_plot_all_{}_{}".format(var_name,model_name), title_left=args.channel, title_right=yr)

### closure plots using the combined weights ###
