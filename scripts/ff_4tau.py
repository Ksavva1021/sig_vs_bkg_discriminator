from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawClosurePlots, DrawReweightPlots, DrawConfusionMatrix
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from UserCode.sig_vs_bkg_discriminator import reweighter

#python scripts/ff_4tau.py --channel=mmtt

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mttt')
parser.add_argument('--pass_wp',help= 'Pass WP for fake factors', default='vvvloose')
parser.add_argument('--fail_wp',help= 'Channel to train BDT for', default=None)
parser.add_argument('--load_dataframes', help= 'Load dataframe from file',  action='store_true')
parser.add_argument('--load_model_multiclass', help= 'Load model from file for multiclass',  action='store_true')
parser.add_argument('--load_model_raw_ff', help= 'Load model from file for raw',  action='store_true')
parser.add_argument('--load_model_alt_ff', help= 'Load model from file for raw',  action='store_true')
parser.add_argument('--load_model_correction', help= 'Load model from file for correction',  action='store_true')
parser.add_argument('--load_hyperparameters_multiclass', help= 'Load hyperparameters from file for raw',  action='store_true')
parser.add_argument('--load_hyperparameters_raw_ff', help= 'Load hyperparameters from file for raw',  action='store_true')
parser.add_argument('--load_hyperparameters_alt_ff', help= 'Load hyperparameters from file for raw',  action='store_true')
parser.add_argument('--load_hyperparameters_correction', help= 'Load hyperparameters from file for correction',  action='store_true')
parser.add_argument('--scan_multiclass', help= 'Do hyperparameter scan for raw',  action='store_true')
parser.add_argument('--scan_raw_ff', help= 'Do hyperparameter scan for raw',  action='store_true')
parser.add_argument('--scan_alt_ff', help= 'Do hyperparameter scan for raw',  action='store_true')
parser.add_argument('--scan_correction', help= 'Do hyperparameter scan for correction',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
args = parser.parse_args()


############# Variables needed #################

years = ["2018"]

y_multiclass = "n_jetfakes"

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

scoring_variables = fitting_variables + ["yr_2016","yr_2017","yr_2018","pred_n_jetfakes"]

reweight_plot_variables = [
                           ["pt_1",(50,0,250)],
#                           ["pt_2",(40,0,200)],
#                           ["pt_3",(30,0,150)],
#                           ["pt_4",(30,0,150)],
                           ]

closure_plot_variables = [
                          ["mvis_12",(60,0,300)],
#                          ["mvis_13",(60,0,300)],
#                          ["mvis_14",(60,0,300)],
#                          ["mvis_23",(60,0,300)],
#                          ["mvis_24",(60,0,300)],
#                          ["mvis_34",(60,0,300)],
                          ]

############### Functions #######################


def PrintDatasetSummary(name,dataset):
  print name
  print dataset.head(10)
  print "Total Length = {}, Total Sum of Weights = {}".format(len(dataset),dataset.loc[:,"weights"].sum())
  print ""

def AddColumnsToDataframe(df, columns, vals):
  for ind, i in enumerate(columns): 
    df.loc[:,i] = val[ind]
  return df.reindex(sorted(df.columns), axis=1)


############# Do Multiclass #####################

print "<<<< Doing multiclass number of jet fakes classification >>>>"

### Get dataframes ###

category_values = range(0,args.channel.count("t")+1)

add_multiclass_variables = ["boolean(q_sum)"]
for ind,p in enumerate(args.channel):
  if p == "t":
    add_multiclass_variables.append("deepTauVsJets_{}_{}".format(args.pass_wp,ind+1))

if not args.load_dataframes:

  print "<< Making dataframes >>"

  bkg_df = Dataframe()
  bkg_df.LoadRootFilesFromJson("json_selection/ff_4tau/ff_mc_{}.json".format(args.channel),fitting_variables+[y_multiclass]+add_multiclass_variables,quiet=(args.verbosity<2))
  bkg_df.TrainTestSplit()
  bkg_df.NormaliseWeights(train_frac=1.0,test_frac=1.0)
  #bkg_df.NormaliseWeightsInCategory(y_multiclass, category_values,train_frac=1.0,test_frac=1.0)
  if args.verbosity > 0: PrintDatasetSummary("Background dataframe",bkg_df.dataframe)

  bkg_df.dataframe.to_pickle("dataframes/ff_bkg_{}_dataframe.pkl".format(args.channel))

else:

  print "<< Loading in dataframes >>"

  bkg_df = Dataframe()
  bkg_df.dataframe = pd.read_pickle("dataframes/ff_bkg_{}_dataframe.pkl".format(args.channel))
  if args.verbosity > 0: PrintDatasetSummary("Background dataframe",bkg_df.dataframe)

### split weights and test and train ###

X_train_bkg_df, y_train_bkg_df, wt_train_bkg_df, X_test_bkg_df, y_test_bkg_df, wt_test_bkg_df = bkg_df.SplitTrainTestXyWts(y=y_multiclass)


### do multiclass training ###

if not args.load_model_multiclass:

  print "<< Running multiclass training >>"

  multiclass_model = xgb.XGBClassifier()

  if args.scan_multiclass:
    param_grid = {
                  "n_estimators":[200,250],
                  "max_depth":[3],
                  "learning_rate":[0.06,0.08],
                  "min_samples_leaf": [500],
                  }
    #dr_reweighter.grid_search(train_fail_dr, train_pass_dr, wt_train_fail_dr ,wt_train_pass_dr, test_fail_dr, test_pass_dr, wt_test_fail_dr ,wt_test_pass_dr, param_grid=param_grid, scoring_variables=scoring_variables)
    # set this up
    with open('hyperparameters/ff_multiclass_hp_{}.json'.format(args.channel), 'w') as outfile: json.dump(multiclass_model.dump_hyperparameters(), outfile)

  else:
    if args.load_hyperparameters_multiclass:
      with open('hyperparameters/ff_multiclass_hp_{}.json'.format(args.channel)) as json_file: params = json.load(json_file)
      multiclass_mode.set_params(params)

    multiclass_model.fit(X_train_bkg_df, y_train_bkg_df, sample_weight=wt_train_bkg_df)

  pkl.dump(multiclass_model,open("BDTs/ff_multiclass_{}.pkl".format(args.channel), "wb"))

else:

  print "<< Loading multiclass training >>"

  multiclass_model = pkl.load(open("BDTs/ff_multiclass_{}.pkl".format(args.channel), "rb"))

### score the multiclass ###

print "<< Scoring multiclass classifier >>"

train_preds = multiclass_model.predict(X_train_bkg_df)
test_preds = multiclass_model.predict(X_test_bkg_df)
train_probs = multiclass_model.predict_proba(X_train_bkg_df)
DrawConfusionMatrix(y_train_bkg_df,train_preds,wt_train_bkg_df,category_values,output="confusion_matrix_train_{}".format(args.channel))
DrawConfusionMatrix(y_test_bkg_df,test_preds,wt_test_bkg_df,category_values,output="confusion_matrix_test_{}".format(args.channel))



############# Do Reweighting ####################

print "<<<< Doing fake factor reweighting >>>>"

first_loop = True

xwt_train_fail_dr = {}
xwt_test_fail_dr = {}
for ind, obj in enumerate(args.channel):
  if obj == "t":

    obj_num = ind + 1
  
    ### Get dataframes ###
    
    if not args.load_dataframes:
    
      print "<< Making dataframes >>"
    
      replace = {}
      if args.fail_wp == None:
        replace["FAIL_1==1"] = "(deepTauVsJets_iso_1>=0)"
        replace["FAIL_2==1"] = "(deepTauVsJets_iso_2>=0)"
        replace["FAIL_3==1"] = "(deepTauVsJets_iso_3>=0)"
        replace["FAIL_4==1"] = "(deepTauVsJets_iso_4>=0)"
      else:
        replace["FAIL"] = "deepTauVsJets_{}".format(args.fail_wp)
      replace["PASS"] = "deepTauVsJets_{}".format(args.pass_wp)
    
      fail_dr = Dataframe()
      fail_dr.LoadRootFilesFromJson("json_selection/ff_4tau/ff_data_{}.json".format(args.channel),fitting_variables+add_multiclass_variables,quiet=(args.verbosity<2),in_extra_name="dr_fail_{}".format(obj_num),replace=replace)
      fail_dr.TrainTestSplit()
      if args.verbosity > 0: PrintDatasetSummary("Fail dr {} dataframe".format(obj_num),fail_dr.dataframe)
      
      if first_loop: # pass_df will be the same no matter what ratio is calculated
        pass_dr = Dataframe()
        pass_dr.LoadRootFilesFromJson("json_selection/ff_4tau/ff_data_{}.json".format(args.channel),fitting_variables+add_multiclass_variables,quiet=(args.verbosity<2),in_extra_name="dr_pass".format(obj_num),replace=replace)
        pass_dr.TrainTestSplit()
        if args.verbosity > 0: PrintDatasetSummary("Pass dr dataframe",pass_dr.dataframe)
    
      fail_dr_alt = Dataframe()
      fail_dr_alt.LoadRootFilesFromJson("json_selection/ff_4tau/ff_data_{}.json".format(args.channel),fitting_variables+add_multiclass_variables,quiet=(args.verbosity<2),in_extra_name="dr_alt_fail_{}".format(obj_num),replace=replace)
      fail_dr_alt.TrainTestSplit()
      if args.verbosity > 0: PrintDatasetSummary("Fail alternative {} dr dataframe".format(obj_num),fail_dr_alt.dataframe)
    
      pass_dr_alt = Dataframe()
      pass_dr_alt.LoadRootFilesFromJson("json_selection/ff_4tau/ff_data_{}.json".format(args.channel),fitting_variables+add_multiclass_variables,quiet=(args.verbosity<2),in_extra_name="dr_alt_pass_{}".format(obj_num),replace=replace)
      pass_dr_alt.TrainTestSplit()
      if args.verbosity > 0: PrintDatasetSummary("Pass alternative {} dr dataframe".format(obj_num),pass_dr_alt.dataframe)
    
      fail_cr = Dataframe()
      fail_cr.LoadRootFilesFromJson("json_selection/ff_4tau/ff_data_{}.json".format(args.channel),fitting_variables+add_multiclass_variables,quiet=(args.verbosity<2),in_extra_name="cr_fail_{}".format(obj_num),replace=replace)
      fail_cr.TrainTestSplit()
      if args.verbosity > 0: PrintDatasetSummary("Fail cr {} dataframe".format(obj_num),fail_cr.dataframe)
      
      pass_cr = Dataframe()
      pass_cr.LoadRootFilesFromJson("json_selection/ff_4tau/ff_data_{}.json".format(args.channel),fitting_variables+add_multiclass_variables,quiet=(args.verbosity<2),in_extra_name="cr_pass_{}".format(obj_num),replace=replace)
      pass_cr.TrainTestSplit()
      if args.verbosity > 0: PrintDatasetSummary("Pass cr {} dataframe".format(obj_num),pass_cr.dataframe)
    
    
      fail_dr.dataframe.to_pickle("dataframes/ff_dr_fail_{}_{}_dataframe.pkl".format(obj_num,args.channel))
      pass_dr.dataframe.to_pickle("dataframes/ff_dr_pass_{}_dataframe.pkl".format(args.channel))
      fail_dr_alt.dataframe.to_pickle("dataframes/ff_dr_alt_fail_{}_{}_dataframe.pkl".format(obj_num,args.channel))
      pass_dr_alt.dataframe.to_pickle("dataframes/ff_dr_alt_pass_{}_{}_dataframe.pkl".format(obj_num,args.channel))
      fail_cr.dataframe.to_pickle("dataframes/ff_cr_fail_{}_{}_dataframe.pkl".format(obj_num,args.channel))
      pass_cr.dataframe.to_pickle("dataframes/ff_cr_pass_{}_{}_dataframe.pkl".format(obj_num,args.channel))
    
    else:
    
      print "<< Loading in dataframes >>"
    
      fail_dr = Dataframe()
      fail_dr.dataframe = pd.read_pickle("dataframes/ff_dr_fail_{}_{}_dataframe.pkl".format(obj_num,args.channel))
      if args.verbosity > 0: PrintDatasetSummary("Fail dr dataframe",fail_dr.dataframe)
    
      pass_dr = Dataframe()
      pass_dr.dataframe = pd.read_pickle("dataframes/ff_dr_pass_{}_dataframe.pkl".format(args.channel))
      if args.verbosity > 0: PrintDatasetSummary("Pass dr dataframe",pass_dr.dataframe)
    
      fail_dr_alt = Dataframe()
      fail_dr_alt.dataframe = pd.read_pickle("dataframes/ff_dr_alt_fail_{}_{}_dataframe.pkl".format(obj_num,args.channel))
      if args.verbosity > 0: PrintDatasetSummary("Fail alternative dr dataframe",fail_dr_alt.dataframe)
    
      pass_dr_alt = Dataframe()
      pass_dr_alt.dataframe = pd.read_pickle("dataframes/ff_dr_alt_pass_{}_{}_dataframe.pkl".format(obj_num,args.channel))
      if args.verbosity > 0: PrintDatasetSummary("Pass alternative dr dataframe",pass_dr_alt.dataframe)
    
      fail_cr = Dataframe()
      fail_cr.dataframe = pd.read_pickle("dataframes/ff_cr_fail_{}_{}_dataframe.pkl".format(obj_num,args.channel))
      if args.verbosity > 0: PrintDatasetSummary("Fail cr dataframe",fail_cr.dataframe)
    
      pass_cr = Dataframe()
      pass_cr.dataframe = pd.read_pickle("dataframes/ff_cr_pass_{}_{}_dataframe.pkl".format(obj_num,args.channel))
      if args.verbosity > 0: PrintDatasetSummary("Pass cr dataframe",pass_cr.dataframe)

    ### split weights and test and train ###
    
    train_fail_dr, wt_train_fail_dr, test_fail_dr, wt_test_fail_dr = fail_dr.SplitTrainTestXWts()
    train_pass_dr, wt_train_pass_dr, test_pass_dr, wt_test_pass_dr = pass_dr.SplitTrainTestXWts()
    train_fail_dr_alt, wt_train_fail_dr_alt, test_fail_dr_alt, wt_test_fail_dr_alt = fail_dr_alt.SplitTrainTestXWts()
    train_pass_dr_alt, wt_train_pass_dr_alt, test_pass_dr_alt, wt_test_pass_dr_alt = pass_dr_alt.SplitTrainTestXWts()
    train_fail_cr, wt_train_fail_cr, test_fail_cr, wt_test_fail_cr = fail_cr.SplitTrainTestXWts()
    train_pass_cr, wt_train_pass_cr, test_pass_cr, wt_test_pass_cr = pass_cr.SplitTrainTestXWts()

    ### add multiclass score to reweights ###

    #train_fail_dr_multiclass_score = multiclass_model.predict_proba(train_fail_dr)
    #train_pass_dr_multiclass_score = multiclass_model.predict_proba(train_pass_dr)
    #train_fail_dr_alt_multiclass_score = multiclass_model.predict_proba(train_fail_dr_alt)
    #train_pass_dr_alt_multiclass_score = multiclass_model.predict_proba(train_pass_dr_alt)
    #train_fail_cr_multiclass_score = multiclass_model.predict_proba(train_fail_cr)
    #train_pass_cr_multiclass_score = multiclass_model.predict_proba(train_pass_cr)
    #test_fail_dr_multiclass_score = multiclass_model.predict_proba(test_fail_dr)
    #test_pass_dr_multiclass_score = multiclass_model.predict_proba(test_pass_dr)
    #test_fail_dr_alt_multiclass_score = multiclass_model.predict_proba(test_fail_dr_alt)
    #test_pass_dr_alt_multiclass_score = multiclass_model.predict_proba(test_pass_dr_alt)
    #test_fail_cr_multiclass_score = multiclass_model.predict_proba(test_fail_cr)
    #test_pass_cr_multiclass_score = multiclass_model.predict_proba(test_pass_cr)

    #for ind, val in enumerate(category_values):
    #  train_fail_dr.loc[:,str(val)+"_fakes_score"] = train_fail_dr_multiclass_score[:,[ind]]
    #  train_pass_dr.loc[:,str(val)+"_fakes_score"] = train_pass_dr_multiclass_score[:,[ind]]
    #  train_fail_dr_alt.loc[:,str(val)+"_fakes_score"] = train_fail_dr_alt_multiclass_score[:,[ind]]
    #  train_pass_dr_alt.loc[:,str(val)+"_fakes_score"] = train_pass_dr_alt_multiclass_score[:,[ind]]
    #  train_fail_cr.loc[:,str(val)+"_fakes_score"] = train_fail_cr_multiclass_score[:,[ind]]
    #  train_pass_cr.loc[:,str(val)+"_fakes_score"] = train_pass_cr_multiclass_score[:,[ind]]
    #  test_fail_dr.loc[:,str(val)+"_fakes_score"] = test_fail_dr_multiclass_score[:,[ind]]
    #  test_pass_dr.loc[:,str(val)+"_fakes_score"] = test_pass_dr_multiclass_score[:,[ind]]
    #  test_fail_dr_alt.loc[:,str(val)+"_fakes_score"] = test_fail_dr_alt_multiclass_score[:,[ind]]
    #  test_pass_dr_alt.loc[:,str(val)+"_fakes_score"] = test_pass_dr_alt_multiclass_score[:,[ind]]
    #  test_fail_cr.loc[:,str(val)+"_fakes_score"] = test_fail_cr_multiclass_score[:,[ind]]
    #  test_pass_cr.loc[:,str(val)+"_fakes_score"] = test_pass_cr_multiclass_score[:,[ind]]

    train_fail_dr.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(train_fail_dr)
    train_pass_dr.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(train_pass_dr)
    train_fail_dr_alt.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(train_fail_dr_alt)
    train_pass_dr_alt.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(train_pass_dr_alt)
    train_fail_cr.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(train_fail_cr)
    train_pass_cr.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(train_pass_cr)
    test_fail_dr.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(test_fail_dr)
    test_pass_dr.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(test_pass_dr)
    test_fail_dr_alt.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(test_fail_dr_alt)
    test_pass_dr_alt.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(test_pass_dr_alt)
    test_fail_cr.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(test_fail_cr)
    test_pass_cr.loc[:,"pred_n_jetfakes"] = multiclass_model.predict(test_pass_cr)

    train_fail_dr = train_fail_dr.drop(add_multiclass_variables,axis=1)
    train_pass_dr = train_pass_dr.drop(add_multiclass_variables,axis=1)
    train_fail_dr_alt = train_fail_dr_alt.drop(add_multiclass_variables,axis=1)
    train_pass_dr_alt = train_pass_dr_alt.drop(add_multiclass_variables,axis=1)
    train_fail_cr = train_fail_cr.drop(add_multiclass_variables,axis=1)
    train_pass_cr = train_pass_cr.drop(add_multiclass_variables,axis=1)
    test_fail_dr = test_fail_dr.drop(add_multiclass_variables,axis=1)
    test_pass_dr = test_pass_dr.drop(add_multiclass_variables,axis=1)
    test_fail_dr_alt = test_fail_dr_alt.drop(add_multiclass_variables,axis=1)
    test_pass_dr_alt = test_pass_dr_alt.drop(add_multiclass_variables,axis=1)
    test_fail_cr = test_fail_cr.drop(add_multiclass_variables,axis=1)
    test_pass_cr = test_pass_cr.drop(add_multiclass_variables,axis=1)
 
    ### raw training ###
    
    if not args.load_model_raw_ff:
    
      print "<< Running raw ff training >>"
      
      dr_reweighter = reweighter.reweighter()
      
      if args.scan_raw_ff:
        param_grid = {
                      "n_estimators":[200,250],
                      "max_depth":[3],
                      "learning_rate":[0.06,0.08],
                      "min_samples_leaf": [500],
                      }
        dr_reweighter.grid_search(train_fail_dr, train_pass_dr, wt_train_fail_dr ,wt_train_pass_dr, test_fail_dr, test_pass_dr, wt_test_fail_dr ,wt_test_pass_dr, param_grid=param_grid, scoring_variables=scoring_variables)
        with open('hyperparameters/ff_dr_hp_{}_{}.json'.format(obj_num,args.channel), 'w') as outfile: json.dump(dr_reweighter.dump_hyperparameters(), outfile)
    
      else:
        if args.load_hyperparameters_raw_ff:
          with open('hyperparameters/ff_dr_hp_{}_{}.json'.format(obj_num,args.channel)) as json_file: params = json.load(json_file)
          dr_reweighter.set_params(params)
        dr_reweighter.norm_and_fit(train_fail_dr, train_pass_dr, wt_train_fail_dr ,wt_train_pass_dr)
    
      pkl.dump(dr_reweighter,open("BDTs/ff_dr_{}_{}.pkl".format(obj_num,args.channel), "wb"))
    
    else:
    
      print "<< Loading raw ff training >>"
    
      dr_reweighter = pkl.load(open("BDTs/ff_dr_{}_{}.pkl".format(obj_num,args.channel), "rb"))
    
    
    ### alt training ##
    
    if not args.load_model_alt_ff:
    
      print "<< Running alternative ff training >>"
    
      dr_alt_reweighter = reweighter.reweighter()
    
      if args.scan_alt_ff:
        param_grid = {
                      "n_estimators":[200,250],
                      "max_depth":[3],
                      "learning_rate":[0.06,0.08],
                      "min_samples_leaf": [500],
                      }
        dr_alt_reweighter.grid_search(train_fail_dr_alt, train_pass_dr_alt, wt_train_fail_dr_alt ,wt_train_pass_dr_alt, test_fail_dr_alt, test_pass_dr_alt, wt_test_fail_dr_alt ,wt_test_pass_dr_alt, param_grid=param_grid, scoring_variables=scoring_variables)
        with open('hyperparameters/ff_dr_alt_hp_{}_{}.json'.format(obj_num,args.channel), 'w') as outfile: json.dump(dr_alt_reweighter.dump_hyperparameters(), outfile)
    
      else:
        if args.load_hyperparameters_alt_ff:
          with open('hyperparameters/ff_dr_alt_hp_{}_{}.json'.format(obj_num,args.channel)) as json_file: params = json.load(json_file)
          dr_alt_reweighter.set_params(params)
        dr_alt_reweighter.norm_and_fit(train_fail_dr_alt, train_pass_dr_alt, wt_train_fail_dr_alt ,wt_train_pass_dr_alt)
    
      pkl.dump(dr_alt_reweighter,open("BDTs/ff_dr_alt_{}_{}.pkl".format(obj_num,args.channel), "wb"))
    
    else:
    
      print "<< Loading alternative ff training >>"
    
      dr_alt_reweighter = pkl.load(open("BDTs/ff_dr_alt_{}_{}.pkl".format(obj_num,args.channel), "rb"))
    
    
    ### correction training ##
    
    ff_train = dr_alt_reweighter.predict_reweights(train_fail_cr)
    ff_test = dr_alt_reweighter.predict_reweights(test_fail_cr)
    
    
    if not args.load_model_correction:
    
      print "<< Running correction training >>"
    
      cr_reweighter = reweighter.reweighter()
      
      if args.scan_correction:
        param_grid = {
                      "n_estimators":[200,250],
                      "max_depth":[3],
                      "learning_rate":[0.06,0.08],
                      "min_samples_leaf": [500],
                      }
        cr_reweighter.grid_search(train_fail_cr, train_pass_cr, np.multiply(wt_train_fail_cr,ff_train) ,wt_train_pass_cr, test_fail_cr, test_pass_cr, np.multiply(wt_test_fail_cr,ff_test) ,wt_test_pass_cr, param_grid=param_grid, scoring_variables=scoring_variables)
        with open('hyperparameters/ff_cr_hp_{}_{}.json'.format(obj_num,args.channel), 'w') as outfile: json.dump(cr_reweighter.dump_hyperparameters(), outfile)
    
      else:
        if args.load_hyperparameters_correction:
          with open('hyperparameters/ff_cr_hp_{}_{}.json'.format(obj_num,args.channel)) as json_file: params = json.load(json_file)
          cr_reweighter.set_params(params)
        cr_reweighter.norm_and_fit(train_fail_cr, train_pass_cr, np.multiply(wt_train_fail_cr,ff_train) ,wt_train_pass_cr)
    
      pkl.dump(cr_reweighter,open("BDTs/ff_cr_{}_{}.pkl".format(obj_num,args.channel), "wb"))
    
    else:
    
      print "<< Loading correction training >>"
    
      cr_reweighter = pkl.load(open("BDTs/ff_cr_{}_{}.pkl".format(obj_num,args.channel), "rb"))
    
    
    ### drawing reweights ###
    
    print "<< Producting reweight plots >>"
    
    reweights_train_dr = pd.Series(dr_reweighter.predict_reweights(train_fail_dr),name="reweights")
    reweights_test_dr = pd.Series(dr_reweighter.predict_reweights(test_fail_dr),name="reweights")
    reweights_train_dr_alt = pd.Series(dr_alt_reweighter.predict_reweights(train_fail_dr_alt),name="reweights")
    reweights_test_dr_alt = pd.Series(dr_alt_reweighter.predict_reweights(test_fail_dr_alt),name="reweights")
    reweights_train_cr = pd.Series(cr_reweighter.predict_reweights(train_fail_cr),name="reweights")
    reweights_test_cr = pd.Series(cr_reweighter.predict_reweights(test_fail_cr),name="reweights")
    
    for yr in years:
      for var in reweight_plot_variables:
        var_name = var[0]
        
        var_train_dr = train_fail_dr.loc[:,var_name]
        DrawReweightPlots(var_train_dr, reweights_train_dr, var_name, "F_{F}", plot_name="reweight_plot_train_dr_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        var_test_dr = test_fail_dr.loc[:,var_name]
        DrawReweightPlots(var_test_dr, reweights_test_dr, var_name, "F_{F}", plot_name="reweight_plot_test_dr_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        var_train_dr_alt = train_fail_dr_alt.loc[:,var_name]
        DrawReweightPlots(var_train_dr_alt, reweights_train_dr_alt, var_name, "F_{F}", plot_name="reweight_plot_train_dr_alt_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        var_test_dr_alt = test_fail_dr_alt.loc[:,var_name]
        DrawReweightPlots(var_test_dr_alt, reweights_test_dr_alt, var_name, "F_{F}", plot_name="reweight_plot_test_dr_alt_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        var_train_cr = train_fail_cr.loc[:,var_name]
        DrawReweightPlots(var_train_cr, reweights_train_cr, var_name, "Correction", plot_name="reweight_plot_train_cr_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        var_test_cr = test_fail_cr.loc[:,var_name]
        DrawReweightPlots(var_test_cr, reweights_test_cr, var_name, "Correction", plot_name="reweight_plot_test_cr_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
    ### closure plots in individual regions###
    
    print "<< Producting closure plots in individual regions>>"
   
    # keep the fail dataset for later closure plots
    xwt_train_fail_dr[obj_num] = pd.concat([train_fail_dr,np.multiply(dr_reweighter.predict_reweights(train_fail_dr),wt_train_fail_dr)],axis=1)
    xwt_train_pass_dr = pd.concat([train_pass_dr,wt_train_pass_dr],axis=1)
    xwt_test_fail_dr[obj_num] = pd.concat([test_fail_dr,np.multiply(dr_reweighter.predict_reweights(test_fail_dr),wt_test_fail_dr)],axis=1)
    xwt_test_pass_dr = pd.concat([test_pass_dr,wt_test_pass_dr],axis=1)
    
    xwt_train_fail_dr_alt = pd.concat([train_fail_dr_alt,np.multiply(dr_alt_reweighter.predict_reweights(train_fail_dr_alt),wt_train_fail_dr_alt)],axis=1)
    xwt_train_pass_dr_alt = pd.concat([train_pass_dr_alt,wt_train_pass_dr_alt],axis=1)
    xwt_test_fail_dr_alt = pd.concat([test_fail_dr_alt,np.multiply(dr_alt_reweighter.predict_reweights(test_fail_dr_alt),wt_test_fail_dr_alt)],axis=1)
    xwt_test_pass_dr_alt = pd.concat([test_pass_dr_alt,wt_test_pass_dr_alt],axis=1)
    
    xwt_train_fail_cr_no_corr = pd.concat([train_fail_cr,np.multiply(ff_train,wt_train_fail_cr)],axis=1)
    xwt_test_fail_cr_no_corr = pd.concat([test_fail_cr,np.multiply(ff_test,wt_test_fail_cr)],axis=1)
    
    xwt_train_fail_cr = pd.concat([train_fail_cr,np.multiply(ff_train,np.multiply(cr_reweighter.predict_reweights(train_fail_cr),wt_train_fail_cr))],axis=1)
    xwt_train_pass_cr = pd.concat([train_pass_cr,wt_train_pass_cr],axis=1)
    xwt_test_fail_cr = pd.concat([test_fail_cr,np.multiply(ff_test,np.multiply(cr_reweighter.predict_reweights(test_fail_cr),wt_test_fail_cr))],axis=1)
    xwt_test_pass_cr = pd.concat([test_pass_cr,wt_test_pass_cr],axis=1)
    
    
    for yr in years:
      for var in closure_plot_variables:
        var_name = var[0]
    
        cut_fail_dr_test = xwt_test_fail_dr[obj_num][(xwt_test_fail_dr[obj_num].loc[:,"yr_"+yr] == 1)]
        cut_pass_dr_test = xwt_test_pass_dr[(xwt_test_pass_dr.loc[:,"yr_"+yr] == 1)]
        cut_fail_dr_test = cut_fail_dr_test.loc[:,["weights",var[0]]]
        cut_pass_dr_test = cut_pass_dr_test.loc[:,["weights",var[0]]]
        DrawClosurePlots(cut_pass_dr_test, cut_fail_dr_test, "pass", "FF x fail", var[0], var[1], plot_name="closure_plot_test_dr_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        cut_fail_dr_train = xwt_train_fail_dr[obj_num][(xwt_train_fail_dr[obj_num].loc[:,"yr_"+yr] == 1)]
        cut_pass_dr_train = xwt_train_pass_dr[(xwt_train_pass_dr.loc[:,"yr_"+yr] == 1)]
        cut_fail_dr_train = cut_fail_dr_train.loc[:,["weights",var[0]]]
        cut_pass_dr_train = cut_pass_dr_train.loc[:,["weights",var[0]]]
        DrawClosurePlots(cut_pass_dr_train, cut_fail_dr_train, "pass", "FF x fail", var[0], var[1], plot_name="closure_plot_train_dr_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        cut_fail_dr_alt_test = xwt_test_fail_dr_alt[(xwt_test_fail_dr_alt.loc[:,"yr_"+yr] == 1)]
        cut_pass_dr_alt_test = xwt_test_pass_dr_alt[(xwt_test_pass_dr_alt.loc[:,"yr_"+yr] == 1)]
        cut_fail_dr_alt_test = cut_fail_dr_alt_test.loc[:,["weights",var[0]]]
        cut_pass_dr_alt_test = cut_pass_dr_alt_test.loc[:,["weights",var[0]]]
        DrawClosurePlots(cut_pass_dr_alt_test, cut_fail_dr_alt_test, "pass", "FF x fail", var[0], var[1], plot_name="closure_plot_test_dr_alt_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        cut_fail_dr_alt_train = xwt_train_fail_dr_alt[(xwt_train_fail_dr_alt.loc[:,"yr_"+yr] == 1)]
        cut_pass_dr_alt_train = xwt_train_pass_dr_alt[(xwt_train_pass_dr_alt.loc[:,"yr_"+yr] == 1)]
        cut_fail_dr_alt_train = cut_fail_dr_alt_train.loc[:,["weights",var[0]]]
        cut_pass_dr_alt_train = cut_pass_dr_alt_train.loc[:,["weights",var[0]]]
        DrawClosurePlots(cut_pass_dr_alt_train, cut_fail_dr_alt_train, "pass", "FF x fail", var[0], var[1], plot_name="closure_plot_train_dr_alt_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        cut_fail_cr_test = xwt_test_fail_cr[(xwt_test_fail_cr.loc[:,"yr_"+yr] == 1)]
        cut_pass_cr_test = xwt_test_pass_cr[(xwt_test_pass_cr.loc[:,"yr_"+yr] == 1)]
        cut_fail_cr_test = cut_fail_cr_test.loc[:,["weights",var[0]]]
        cut_pass_cr_test = cut_pass_cr_test.loc[:,["weights",var[0]]]
        DrawClosurePlots(cut_pass_cr_test, cut_fail_cr_test, "pass", "FF x corr x fail", var[0], var[1], plot_name="closure_plot_test_cr_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        cut_fail_cr_train = xwt_train_fail_cr[(xwt_train_fail_cr.loc[:,"yr_"+yr] == 1)]
        cut_pass_cr_train = xwt_train_pass_cr[(xwt_train_pass_cr.loc[:,"yr_"+yr] == 1)]
        cut_fail_cr_train = cut_fail_cr_train.loc[:,["weights",var[0]]]
        cut_pass_cr_train = cut_pass_cr_train.loc[:,["weights",var[0]]]
        DrawClosurePlots(cut_pass_cr_train, cut_fail_cr_train, "pass", "FF x corr x fail", var[0], var[1], plot_name="closure_plot_train_cr_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        cut_fail_cr_no_corr_test = xwt_test_fail_cr_no_corr[(xwt_test_fail_cr_no_corr.loc[:,"yr_"+yr] == 1)]
        cut_pass_cr_test = xwt_test_pass_cr[(xwt_test_pass_cr.loc[:,"yr_"+yr] == 1)]
        cut_fail_cr_no_corr_test = cut_fail_cr_no_corr_test.loc[:,["weights",var[0]]]
        cut_pass_cr_test = cut_pass_cr_test.loc[:,["weights",var[0]]]
        DrawClosurePlots(cut_pass_cr_test, cut_fail_cr_no_corr_test, "pass", "FF x fail", var[0], var[1], plot_name="closure_plot_test_cr_no_corr_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)
    
        cut_fail_cr_no_corr_train = xwt_train_fail_cr_no_corr[(xwt_train_fail_cr_no_corr.loc[:,"yr_"+yr] == 1)]
        cut_pass_cr_train = xwt_train_pass_cr[(xwt_train_pass_cr.loc[:,"yr_"+yr] == 1)]
        cut_fail_cr_no_corr_train = cut_fail_cr_no_corr_train.loc[:,["weights",var[0]]]
        cut_pass_cr_train = cut_pass_cr_train.loc[:,["weights",var[0]]]
        DrawClosurePlots(cut_pass_cr_train, cut_fail_cr_no_corr_train, "pass", "FF x fail", var[0], var[1], plot_name="closure_plot_train_cr_no_corr_{}_{}_{}_{}".format(var_name,obj_num,args.channel,yr), title_left=args.channel, title_right=yr)

    first_loop = False


### closure plots in individual regions ###

print "<< Producting closure plots in combined region>>"

for key, val in xwt_test_fail_dr.items():
  for i in range(1,args.channel.count("t")+1):
    if i == 1:
      frac_test = val.loc[:,str(i)+"_fakes_score"]
    else:
      frac_test += ((1./i)*val.loc[:,str(i)+"_fakes_score"])
  xwt_test_fail_dr[key].loc[:,"weights"].multiply(frac_test)

for key, val in xwt_train_fail_dr.items():
  for i in range(1,args.channel.count("t")+1):
    if i == 1:
      frac_train = val.loc[:,str(i)+"_fakes_score"]
    else:
      frac_train += ((1./i)*val.loc[:,str(i)+"_fakes_score"])
  xwt_train_fail_dr[key].loc[:,"weights"].multiply(frac_train)

for yr in years:
  for var in closure_plot_variables:
    var_name = var[0]
    cut_pass_dr_test = xwt_test_pass_dr[(xwt_test_pass_dr.loc[:,"yr_"+yr] == 1)]
    cut_pass_dr_test = cut_pass_dr_test.loc[:,["weights",var[0]]]
    cut_pass_dr_train = xwt_train_pass_dr[(xwt_train_pass_dr.loc[:,"yr_"+yr] == 1)]
    cut_pass_dr_train = cut_pass_dr_train.loc[:,["weights",var[0]]]


    make_df = True
    for ind,p in enumerate(args.channel):
      if p == "t":
        cut_fail_dr_test = xwt_test_fail_dr[obj_num][(xwt_test_fail_dr[obj_num].loc[:,"yr_"+yr] == 1)]
        cut_fail_dr_test = cut_fail_dr_test.loc[:,["weights",var[0]]]

        cut_fail_dr_train = xwt_train_fail_dr[obj_num][(xwt_train_fail_dr[obj_num].loc[:,"yr_"+yr] == 1)]
        cut_fail_dr_train = cut_fail_dr_train.loc[:,["weights",var[0]]]

        if make_df:
          total_cut_fail_dr_test = cut_fail_dr_test.copy()
          total_cut_fail_dr_train = cut_fail_dr_train.copy()
          make_df = False
        else:
          total_cut_fail_dr_test = pd.concat([total_cut_fail_dr_test,cut_fail_dr_test],ignore_index=True, sort=False)
          total_cut_fail_dr_train = pd.concat([total_cut_fail_dr_train,cut_fail_dr_train],ignore_index=True, sort=False)

    DrawClosurePlots(cut_pass_dr_test, total_cut_fail_dr_test, "pass", "FF x fail", var[0], var[1], plot_name="closure_plot_test_dr_{}_fractioned_{}_{}".format(var_name,args.channel,yr), title_left=args.channel, title_right=yr)
    DrawClosurePlots(cut_pass_dr_train, total_cut_fail_dr_train, "pass", "FF x fail", var[0], var[1], plot_name="closure_plot_train_dr_{}_fractioned_{}_{}".format(var_name,args.channel,yr), title_left=args.channel, title_right=yr)


