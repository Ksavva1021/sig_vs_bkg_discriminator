import matplotlib
import matplotlib.pyplot as plt
import pylab as plt
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from xgboost import plot_importance
import numpy as np
import seaborn as sns
def DrawROCCurve(act,pred,wt,output="roc_curve"):

  fpr, tpr, threshold = metrics.roc_curve(act, pred,sample_weight=wt)
  roc_auc = roc_auc_score(act, pred, sample_weight=wt)
  
  plt.title('ROC')
  plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
  print "AUC Score:", roc_auc
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.draw()
  plt.savefig("plots/"+output+".pdf")
  print "plots/"+output+".pdf created"
  plt.close()


def DrawDistributions(signal_dataframe,bkg_dataframe,variable,xrange,nbins,channel="channel",location="plots/Distributions/"):
    x_lower_bound = xrange[0]
    x_upper_bound = xrange[1]
    plt.figure()
    sns.distplot(signal_dataframe['{}'.format(variable)],  kde=True, hist_kws={"color": "k","range": [x_lower_bound,x_upper_bound]},kde_kws={"color": "k","clip":[x_lower_bound,x_upper_bound]}, label='Signal',bins = nbins)
    sns.distplot(bkg_dataframe['{}'.format(variable)],  kde=True,hist_kws={"color": "g","range": [x_lower_bound,x_upper_bound]},kde_kws={"color": "g","clip":[x_lower_bound,x_upper_bound]}, label='bkg',bins = nbins)
    plt.legend(prop={'size': 12})
    plt.title('{}_{}'.format(channel,variable))
    plt.xlabel('{}'.format(variable))
    plt.ylabel('Density')
    plt.xlim(x_lower_bound,x_upper_bound)
    plt.savefig("{}{}_{}.png".format(location,channel,variable))


def DrawMultipleROCCurves(act,pred,wt,output="roc_curve",name="ROC"):

  for key,val in act.iteritems():
    fpr, tpr, threshold = metrics.roc_curve(act[key], pred[key], sample_weight=wt[key])
    roc_auc = roc_auc_score(act[key], pred[key], sample_weight=wt[key])
    #print key,"AUC Score:", roc_auc
    plt.plot(fpr, tpr, label = r"{}".format(key) +', AUC = %0.4f' % roc_auc)
  plt.title(name)
  plt.legend(loc = 'lower right',fontsize='x-small')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.draw()
  plt.savefig("plots/"+output+".pdf")
  print "plots/"+output+".pdf created"
  plt.close()

def DrawMultipleROCCurvesOnOnePage(act_dict,pred_dict,wt_dict,output="combined_roc_curves",len_x=3,len_y=3):
  plt.rcParams.update({'font.size': 3})
  fig, axs = plt.subplots(len_x,len_y)
  plt.subplots_adjust(left=0.05, 
                      bottom=0.05, 
                      right=0.95, 
                      top=0.95, 
                      wspace=0.2, 
                      hspace=0.3)
  i = 0
  j = 0
  for name, act in act_dict.iteritems():
    ind = 0
    max_roc_auc = 0
    max_roc_auc_ind = 0
    for key,val in act.iteritems():
      fpr, tpr, threshold = metrics.roc_curve(act_dict[name][key], pred_dict[name][key], sample_weight=wt_dict[name][key])
      roc_auc = roc_auc_score(act_dict[name][key], pred_dict[name][key], sample_weight=wt_dict[name][key])
      axs[i,j].plot(fpr, tpr, label = r"${}$".format(key) +', AUC = %0.4f' % roc_auc)
      if roc_auc > max_roc_auc: 
        max_roc_auc = 1*roc_auc
        max_roc_auc_ind = 1*ind
      ind += 1
    axs[i,j].set_title(r"{}".format(name))
    leg = axs[i,j].legend(loc = 'lower right')
    axs[i,j].plot([0, 1], [0, 1],'r--')
    axs[i,j].set_xlim([0, 1])
    axs[i,j].set_ylim([0, 1])
    axs[i,j].set_ylabel('True Positive Rate')
    axs[i,j].set_xlabel('False Positive Rate')
    for ind_leg, text in enumerate(leg.get_texts()):
      if max_roc_auc_ind == ind_leg:
        plt.setp(text, color = 'r')
    i += 1
    if i % len_x == 0:
      i = 0
      j += 1
  fig.savefig("plots/"+output+".pdf")
  print "plots/"+output+".pdf created"

def DrawBDTScoreDistributions(pred_dict,output="bdt_score"):

  for key, val in pred_dict.items():
    _, bins, _ = plt.hist(val["preds"], weights=val["weights"] ,bins=100, histtype='step', label=key)

  plt.xlabel("BDT Score")
  plt.xlim(bins[0], bins[-1])
  plt.legend(loc='best')
  plt.draw()
  plt.savefig("plots/"+output+".pdf")
  plt.close()
  
def DrawVarDistribution(df,n_classes,variable,xlim_low,xlim_high,output="var_dist",bin_edges=100):
  if n_classes == 2:
    c0 = df.loc[(df.loc[:,"y"]==0)]
    c1 = df.loc[(df.loc[:,"y"]==1)]
    
    wt0 = c0.loc[:,"weights"]
    wt1 = c1.loc[:,"weights"]
    
    val0 = c0.loc[:,variable]
    val1 = c1.loc[:,variable]
 
    dict_ = {"cat1":{"preds":val0,"weights":wt0},"cat2":{"preds":val1,"weights":wt1}}

  else:
    c0 = df.loc[(df.loc[:,"y"]==0)]
    c1 = df.loc[(df.loc[:,"y"]==1)]
    c2 = df.loc[(df.loc[:,"y"]==2)]
    
    wt0 = c0.loc[:,"weights"]
    wt1 = c1.loc[:,"weights"]
    wt2 = c2.loc[:,"weights"]
    
    val0 = c0.loc[:,variable]
    val1 = c1.loc[:,variable]
    val2 = c2.loc[:,variable]
    
    dict_ = {"cat1":{"preds":val0,"weights":wt0},"cat2":{"preds":val1,"weights":wt1},"cat3":{"preds":val2,"weights":wt2}}
    
  for key, val in dict_.items():
    _, bins, _ = plt.hist(val["preds"], weights=val["weights"] ,bins=bin_edges, histtype='step', label=key)
    
  plt.xlabel(variable)
  plt.xlim(xlim_low,xlim_high)
  plt.legend(loc='best')
  plt.draw()
  plt.savefig("plots/"+output+".pdf")
  plt.close()

def DrawFeatureImportance(model,imp_type,output="feature_importance"):
  ax = plot_importance(model, importance_type = imp_type, xlabel = imp_type)
  ax.tick_params(axis='y', labelsize=5)
  ax.figure.savefig("plots/"+output+".pdf")
  

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def DrawConfusionMatrix(y_test,preds,wt_test,label,output="confusion_matrix"):
  cm = confusion_matrix(y_test,preds,sample_weight=wt_test)
  new_cm = []
  for ind, i in enumerate(cm):
    norm = sum(i)
    new_cm.append([])
    for j in i: new_cm[ind].append(j/norm)

  data_array = np.array(new_cm)
  fig, ax1 = plt.subplots()
  im, cbar = heatmap(data_array, label, label, ax=ax1, cmap="Blues")
  texts = annotate_heatmap(im, valfmt="{x:.3f}")

  plt.ylabel('True Label')
  plt.xlabel('Predicted Label')
  fig.tight_layout()
  plt.savefig("plots/"+output+".pdf")

