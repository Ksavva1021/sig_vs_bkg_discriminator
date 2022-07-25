from hep_ml.reweight import GBReweighter
from hep_ml.metrics_utils import ks_2samp_weighted
import copy
import itertools
import numpy as np

class reweighter(GBReweighter):
  
  def __init__(self,
               n_estimators=40,
               learning_rate=0.2,
               max_depth=3,
               min_samples_leaf=200,
               loss_regularization=5.,
               gb_args=None):

    self.learning_rate = learning_rate
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.min_samples_leaf = min_samples_leaf
    self.gb_args = gb_args
    self.loss_regularization = loss_regularization
    self.normalization = 1.0

  def norm_and_fit(self, original, target, original_weight, target_weight):
    self.normalization = target_weight.sum()/original_weight.sum()
    original_weight = self.normalization*original_weight
    self.fit(original, target, original_weight=original_weight, target_weight=target_weight)
    total_original_weights = np.multiply(self.predict_reweights(original,add_norm=False),original_weight)
    self.normalization = self.normalization*target_weight.sum()/total_original_weights.sum() 

  def predict_reweights(self, original, add_norm=True):
    wts = self.predict_weights(original) 
    if add_norm:
      return self.normalization*wts
    else:
      return wts

  def dump_hyperparameters(self):
    hp = copy.deepcopy(self.gb_args)
    hp["learning_rate"] = self.learning_rate
    hp["n_estimators"] = self.n_estimators
    hp["max_depth"] = self.max_depth
    hp["min_samples_leaf"] = self.min_samples_leaf
    hp["loss_regularization"] = self.loss_regularization
    return hp

  def draw_distributions(self, original, target, original_weights, target_weights, columns=['pt_1']):
    plt.figure(figsize=[15, 12])
    for id, column in enumerate(columns, 1):
      xlim = np.percentile(np.hstack([target[column]]), [0.01, 99.99])
      plt.subplot(3, 3, id)
      plt.hist(original[column], weights=original_weights, range=xlim, bins=50, alpha=0.8, color="red", label="original")
      plt.hist(target[column], weights=target_weights, range=xlim, bins=50, alpha=0.8, color="blue", label="target")
      plt.title(column)
      plt.legend()
    plt.show()
  
  def KS(self, original, target, original_weights, target_weights, columns=['pt_1']):
    ks_dict = {}
    ks_total = 0
    for id, column in enumerate(columns, 1):
      ks_dict[column] = round(ks_2samp_weighted(original[column], target[column], weights1=original_weights, weights2=target_weights),6)
      ks_total += ks_dict[column]
    return round(ks_total,6), ks_dict

  def set_params(self,val):
    if "learning_rate" in val.keys():
      self.learning_rate = val["learning_rate"]
      del val["learning_rate"]
    if "n_estimators" in val.keys():
      self.n_estimators = val["n_estimators"]
      del val["n_estimators"]
    if "max_depth" in val.keys():
      self.max_depth = val["max_depth"]
      del val["max_depth"]
    if "min_samples_leaf" in val.keys():
      self.min_samples_leaf = val["min_samples_leaf"]
      del val["min_samples_leaf"]
    if "loss_regularization" in val.keys():
      self.loss_regularization = val["loss_regularization"]
      del val["loss_regularization"]
    self.gb_args = copy.deepcopy(val)

  def grid_search(self, original_train, target_train, original_train_weight, target_train_weight, original_test, target_test, original_test_weight, target_test_weight, param_grid={}, scoring_variables=["pt_1"]):
    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    lowest_KS = 9999
    for ind, val in enumerate(permutations_dicts):
      unchanged_val = copy.deepcopy(val)
      self.set_params(val)
      self.norm_and_fit(original_train, target_train, original_train_weight, target_train_weight)
      test_reweights = self.predict_reweights(original_test)
      score = self.KS(original_test, target_test, np.multiply(original_test_weight,test_reweights), target_test_weight, columns=scoring_variables)
      print "Parameter grid:", unchanged_val
      print "KS score:", score
      print ""
      if score[0] < lowest_KS:
        lowest_KS = score[0]*1.0
        best_model = copy.deepcopy(self)
        best_hp = copy.deepcopy(unchanged_val)
    print "Best hyperparameters:", best_hp
    print "Lowest KS score:", lowest_KS
    self = best_model 
    


