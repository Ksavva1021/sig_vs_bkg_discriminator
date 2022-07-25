from UserCode.sig_vs_bkg_discriminator.Dataframe import Dataframe
from UserCode.sig_vs_bkg_discriminator.plotting import DrawClosurePlots 
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import json
from UserCode.sig_vs_bkg_discriminator import reweighter

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mttt')
parser.add_argument('--load_dataframes', help= 'Load dataframe from file',  action='store_true')
parser.add_argument('--load_model_raw', help= 'Load model from file for raw',  action='store_true')
parser.add_argument('--load_model_correction', help= 'Load model from file for correction',  action='store_true')
parser.add_argument('--load_hyperparameters_raw', help= 'Load hyperparameters from file for raw',  action='store_true')
parser.add_argument('--load_hyperparameters_correction', help= 'Load hyperparameters from file for correction',  action='store_true')
parser.add_argument('--scan_raw', help= 'Do hyperparameter scan for raw',  action='store_true')
parser.add_argument('--scan_correction', help= 'Do hyperparameter scan for correction',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
args = parser.parse_args()

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


scoring_variables = [
             "pt_1", "pt_2","pt_3","pt_4",
             "fabs(dphi_12)","fabs(dphi_13)","fabs(dphi_14)","fabs(dphi_23)","fabs(dphi_24)","fabs(dphi_34)",
             "fabs(dR_12)","fabs(dR_13)","fabs(dR_14)","fabs(dR_23)","fabs(dR_24)","fabs(dR_34)",
             "mt_1","mt_2","mt_3","mt_4",
             "mt_lep_12","mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34",
             "mvis_12","mvis_13","mvis_14","mvis_23","mvis_24","mvis_34",
             "pt_tt_12","pt_tt_13","pt_tt_14","pt_tt_23","pt_tt_24","pt_tt_34",
             "n_jets","n_bjets",
             "yr_2016","yr_2017","yr_2018"
             ] 

closure_variables = [
                     ["mvis_12",(60,0,300)],
                     ["mvis_13",(60,0,300)],
                     ["mvis_14",(60,0,300)],
                     ["mvis_23",(60,0,300)],
                     ["mvis_24",(60,0,300)],
                     ["mvis_34",(60,0,300)],
                     ["pt_1",(50,0,250)],
                     ["pt_2",(40,0,200)],
                     ["pt_3",(30,0,150)],
                     ["pt_4",(30,0,150)],
                     ]

############### Functions #######################


def PrintDatasetSummary(name,dataset):
  print name
  print dataset.head(10)
  print "Total Length = {}, Total Sum of Weights = {}".format(len(dataset),dataset.loc[:,"weights"].sum())
  print ""


#################################################

ch = args.channel

# get dataframes
if not args.load_dataframes:

  print "<< Making dataframes >>"

  fail_dr = Dataframe()
  fail_dr.LoadRootFilesFromJson("json_selection/ff_%(ch)s.json" % vars(),fitting_variables,quiet=(args.verbosity<2),in_extra_name="dr_fail")
  fail_dr.TrainTestSplit()
  if args.verbosity > 0: PrintDatasetSummary("Fail dr dataframe",fail_dr.dataframe)
  
  pass_dr = Dataframe()
  pass_dr.LoadRootFilesFromJson("json_selection/ff_%(ch)s.json" % vars(),fitting_variables,quiet=(args.verbosity<2),in_extra_name="dr_pass")
  pass_dr.TrainTestSplit()
  if args.verbosity > 0: PrintDatasetSummary("Pass dr dataframe",pass_dr.dataframe)

  fail_cr = Dataframe()
  fail_cr.LoadRootFilesFromJson("json_selection/ff_%(ch)s.json" % vars(),fitting_variables,quiet=(args.verbosity<2),in_extra_name="cr_fail")
  fail_cr.TrainTestSplit()
  if args.verbosity > 0: PrintDatasetSummary("Fail cr dataframe",fail_cr.dataframe)
  
  pass_cr = Dataframe()
  pass_cr.LoadRootFilesFromJson("json_selection/ff_%(ch)s.json" % vars(),fitting_variables,quiet=(args.verbosity<2),in_extra_name="cr_pass")
  pass_cr.TrainTestSplit()
  if args.verbosity > 0: PrintDatasetSummary("Pass cr dataframe",pass_cr.dataframe)


  fail_dr.dataframe.to_pickle("dataframes/ff_dr_fail_%(ch)s_dataframe.pkl" % vars())
  pass_dr.dataframe.to_pickle("dataframes/ff_dr_pass_%(ch)s_dataframe.pkl" % vars())
  fail_cr.dataframe.to_pickle("dataframes/ff_cr_fail_%(ch)s_dataframe.pkl" % vars())
  pass_cr.dataframe.to_pickle("dataframes/ff_cr_pass_%(ch)s_dataframe.pkl" % vars())

else:

  print "<< Loading in dataframes >>"

  fail_dr = Dataframe()
  fail_dr.dataframe = pd.read_pickle("dataframes/ff_dr_fail_%(ch)s_dataframe.pkl" % vars())
  if args.verbosity > 0: PrintDatasetSummary("Fail dr dataframe",fail_dr.dataframe)

  pass_dr = Dataframe()
  pass_dr.dataframe = pd.read_pickle("dataframes/ff_dr_pass_%(ch)s_dataframe.pkl" % vars())
  if args.verbosity > 0: PrintDatasetSummary("Pass dr dataframe",pass_dr.dataframe)

  fail_cr = Dataframe()
  fail_cr.dataframe = pd.read_pickle("dataframes/ff_cr_fail_%(ch)s_dataframe.pkl" % vars())
  if args.verbosity > 0: PrintDatasetSummary("Fail cr dataframe",fail_cr.dataframe)

  pass_cr = Dataframe()
  pass_cr.dataframe = pd.read_pickle("dataframes/ff_cr_pass_%(ch)s_dataframe.pkl" % vars())
  if args.verbosity > 0: PrintDatasetSummary("Pass cr dataframe",pass_cr.dataframe)



train_fail_dr, wt_train_fail_dr, test_fail_dr, wt_test_fail_dr = fail_dr.SplitTrainTestXWts()
train_pass_dr, wt_train_pass_dr, test_pass_dr, wt_test_pass_dr = pass_dr.SplitTrainTestXWts()
train_fail_cr, wt_train_fail_cr, test_fail_cr, wt_test_fail_cr = fail_cr.SplitTrainTestXWts()
train_pass_cr, wt_train_pass_cr, test_pass_cr, wt_test_pass_cr = pass_cr.SplitTrainTestXWts()


# raw training
if not args.load_model_raw:

  print "<< Running raw training >>"
  
  dr_reweighter = reweighter.reweighter()
  
  if args.scan_raw:
    param_grid = {
                  "n_estimators":[200,250],
                  "max_depth":[3],
                  "learning_rate":[0.06,0.08],
                  "min_samples_leaf": [500],
                  }
    dr_reweighter.grid_search(train_fail_dr, train_pass_dr, wt_train_fail_dr ,wt_train_pass_dr, test_fail_dr, test_pass_dr, wt_test_fail_dr ,wt_test_pass_dr, param_grid=param_grid, scoring_variables=scoring_variables)
    with open('hyperparameters/ff_dr_hp_%(ch)s.json' % vars(), 'w') as outfile: json.dump(dr_reweighter.dump_hyperparameters(), outfile)

  else:
    if args.load_hyperparameters_raw:
      with open('hyperparameters/ff_dr_hp_%(ch)s.json' % vars()) as json_file: params = json.load(json_file)
      dr_reweighter.set_params(params)
    dr_reweighter.norm_and_fit(train_fail_dr, train_pass_dr, wt_train_fail_dr ,wt_train_pass_dr)

  pkl.dump(dr_reweighter,open("BDTs/ff_dr_%(ch)s.pkl" % vars(), "wb"))

else:

  dr_reweighter = pkl.load(open("BDTs/ff_dr_%(ch)s.pkl" % vars(), "rb"))


# correction training
ff_train = dr_reweighter.predict_reweights(train_fail_cr)
ff_test = dr_reweighter.predict_reweights(test_fail_cr)


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
    with open('hyperparameters/ff_cr_hp_%(ch)s.json' % vars(), 'w') as outfile: json.dump(cr_reweighter.dump_hyperparameters(), outfile)

  else:
    if args.load_hyperparameters_correction:
      with open('hyperparameters/ff_cr_hp_%(ch)s.json' % vars()) as json_file: params = json.load(json_file)
      cr_reweighter.set_params(params)
    cr_reweighter.norm_and_fit(train_fail_cr, train_pass_cr, np.multiply(wt_train_fail_cr,ff_train) ,wt_train_pass_cr)

  pkl.dump(cr_reweighter,open("BDTs/ff_cr_%(ch)s.pkl" % vars(), "wb"))

else:

  cr_reweighter = pkl.load(open("BDTs/ff_cr_%(ch)s.pkl" % vars(), "rb"))


# Would be nice to add plots to show size of reweighting, maybe binned in pT or something?

# closure plots
print "<< Producting closure plots >>"

xwt_train_fail_dr = pd.concat([train_fail_dr,np.multiply(dr_reweighter.predict_reweights(train_fail_dr),wt_train_fail_dr)],axis=1)
xwt_train_pass_dr = pd.concat([train_pass_dr,wt_train_pass_dr],axis=1)
xwt_test_fail_dr = pd.concat([test_fail_dr,np.multiply(dr_reweighter.predict_reweights(test_fail_dr),wt_test_fail_dr)],axis=1)
xwt_test_pass_dr = pd.concat([test_pass_dr,wt_test_pass_dr],axis=1)

xwt_train_fail_cr = pd.concat([train_fail_cr,np.multiply(ff_train,np.multiply(cr_reweighter.predict_reweights(train_fail_cr),wt_train_fail_cr))],axis=1)
xwt_train_pass_cr = pd.concat([train_pass_cr,wt_train_pass_cr],axis=1)
xwt_test_fail_cr = pd.concat([test_fail_cr,np.multiply(ff_test,np.multiply(cr_reweighter.predict_reweights(test_fail_cr),wt_test_fail_cr))],axis=1)
xwt_test_pass_cr = pd.concat([test_pass_cr,wt_test_pass_cr],axis=1)


for yr in ["2018"]:
  for var in closure_variables:
    var_name = var[0]

    cut_fail_dr_test = xwt_test_fail_dr[(xwt_test_fail_dr.loc[:,"yr_"+yr] == 1)]
    cut_pass_dr_test = xwt_test_pass_dr[(xwt_test_pass_dr.loc[:,"yr_"+yr] == 1)]
    cut_fail_dr_test = cut_fail_dr_test.loc[:,["weights",var[0]]]
    cut_pass_dr_test = cut_pass_dr_test.loc[:,["weights",var[0]]]
    DrawClosurePlots(cut_pass_dr_test, cut_fail_dr_test, "pass", "FF x fail", var[0], var[1], plot_name="closure_plot_test_dr_%(var_name)s_%(ch)s_%(yr)s" % vars(), title_left=ch, title_right=yr)

    cut_fail_dr_train = xwt_train_fail_dr[(xwt_train_fail_dr.loc[:,"yr_"+yr] == 1)]
    cut_pass_dr_train = xwt_train_pass_dr[(xwt_train_pass_dr.loc[:,"yr_"+yr] == 1)]
    cut_fail_dr_train = cut_fail_dr_train.loc[:,["weights",var[0]]]
    cut_pass_dr_train = cut_pass_dr_train.loc[:,["weights",var[0]]]
    DrawClosurePlots(cut_pass_dr_train, cut_fail_dr_train, "pass", "FF x fail", var[0], var[1], plot_name="closure_plot_train_dr_%(var_name)s_%(ch)s_%(yr)s" % vars(), title_left=ch, title_right=yr)

    cut_fail_cr_test = xwt_test_fail_cr[(xwt_test_fail_cr.loc[:,"yr_"+yr] == 1)]
    cut_pass_cr_test = xwt_test_pass_cr[(xwt_test_pass_cr.loc[:,"yr_"+yr] == 1)]
    cut_fail_cr_test = cut_fail_cr_test.loc[:,["weights",var[0]]]
    cut_pass_cr_test = cut_pass_cr_test.loc[:,["weights",var[0]]]
    DrawClosurePlots(cut_pass_cr_test, cut_fail_cr_test, "pass", "FF x corr x fail", var[0], var[1], plot_name="closure_plot_test_cr_%(var_name)s_%(ch)s_%(yr)s" % vars(), title_left=ch, title_right=yr)

    cut_fail_cr_train = xwt_train_fail_cr[(xwt_train_fail_cr.loc[:,"yr_"+yr] == 1)]
    cut_pass_cr_train = xwt_train_pass_cr[(xwt_train_pass_cr.loc[:,"yr_"+yr] == 1)]
    cut_fail_cr_train = cut_fail_cr_train.loc[:,["weights",var[0]]]
    cut_pass_cr_train = cut_pass_cr_train.loc[:,["weights",var[0]]]
    DrawClosurePlots(cut_pass_cr_train, cut_fail_cr_train, "pass", "FF x corr x fail", var[0], var[1], plot_name="closure_plot_train_cr_%(var_name)s_%(ch)s_%(yr)s" % vars(), title_left=ch, title_right=yr)

