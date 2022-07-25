import ROOT
import uproot
import numpy as np
import pandas as pd
import re
import math
import json
from prettytable import PrettyTable
from textwrap import wrap
from sklearn.model_selection import train_test_split

class Dataframe:

  def __init__(self):
    self.root_files = []
    self.tree_names = []
    self.root_selection = {}
    self.python_selection = {}
    self.variables_for_selection = []
    self.columns = []
    self.modified_columns = []
    self.variables_for_modified = []
    self.scale_column = {}
    self.file_location = ''
    self.file_ext = ''
    self.dataframe = None
    self.func_dict = {"fabs":"abs(x)","cos":"math.cos(x)","sin":"math.sin(x)", "cosh":"math.cosh(x)", "sinh":"math.sinh(x)", "ln":"math.log(x)"}

  def __AllSplitStringsSteps__(self,selection):
    #print "-----------------------------------------------------------------------------------------------"
    #print selection
    split_strings = self.__GetSplitStrings__(selection)
    self.__AddVariablesForSelection__(split_strings)
    new_split_strings = self.__ConvetSplitStrings__(split_strings)
    #print ''.join(new_split_strings)
    #print "-----------------------------------------------------------------------------------------------"
    return new_split_strings

  def __AddVariablesForSelection__(self,split_strings):
    # Get the variables needed for selection
    for i in split_strings:
      if i[0].isalpha() and i not in self.func_dict.keys() and i not in self.variables_for_selection:
        self.variables_for_selection.append(i)

  def __ConvetSplitStrings__(self,split_strings):
    # Change operator names to python
    for ind, val in enumerate(split_strings):
      if val in self.variables_for_selection:
        split_strings[ind] = '(df.loc[:,"{}"]'.format(val)
      elif val in ["*","/"] and (split_strings[ind-1].isdigit() or split_strings[ind+1].isdigit()):
        continue
      else:
        split_strings[ind]= val.replace("&&",")&").replace("||",")|").replace("*",").multiply").replace("/",").divide")

    split_strings.append(")")

    # Split up all deliminators
    new_split_strings = []
    for ind, val in enumerate(split_strings):
      if not (val[0].isdigit() or "df.loc" in val or val in self.func_dict.keys()):
      #if not (val[0].isdigit() or val[0].isalpha()):
        new_split_strings += list(val)
      else:
        new_split_strings.append(val)

    # Change function names to lambda functions
    functions_in_string = False
    for key in self.func_dict.keys():
      if key in new_split_strings:
        functions_in_string = True
    while functions_in_string:
      found_func = None
      for ind, val in enumerate(new_split_strings):
        if val in self.func_dict.keys() and found_func == None:
          found_func = val
          num_of_func = ind
          bracket_count = 0
        if found_func != None and ind != num_of_func:
          bracket_count = bracket_count + len(re.findall("\(",val)) - len(re.findall("\)",val))
          if bracket_count == 0:
            new_split_strings[ind] = ").apply(lambda x: {}))".format(self.func_dict[found_func])
            new_split_strings[num_of_func] = "("
            found_func = None
            continue

      functions_in_string = False
      for key in self.func_dict.keys():
        if key in new_split_strings:
          functions_in_string = True

    return new_split_strings

  def __GetSplitStrings__(self,selection):
    selection = selection.replace(" ","")

    # Put the selection into a list of useful parts
    split_strings = []
    prev_alpha = False
    for ch in selection:
      if not prev_alpha:
        if ch.isalpha() or ch.isdigit() or ch == ".":
          split_strings.append(ch)
          prev_alpha = True
        else:
          if len(split_strings) == 0:
            split_strings.append(ch)
          else:
            split_strings[len(split_strings)-1] += ch
      else:
        if not (ch.isalpha() or ch.isdigit() or ch in ["_","."]) :
          split_strings.append(ch)
          prev_alpha = False
        else:
          split_strings[len(split_strings)-1] += ch

    return split_strings

  def AddRootFiles(self,root_files,tree_name="ntuple"):

    if isinstance(root_files, str) or isinstance(root_files, unicode):
      root_files = [root_files]
    elif not isinstance(root_files, list):
      raise TypeError
    
    for f in root_files:
     if not f in self.root_files:
       self.root_files.append(f)
       self.tree_names.append(tree_name)
       self.root_selection[f] = "(1)"
       self.python_selection[f] = "(1)"
     else:
       print "ERROR: file name already exists"

  def PrintRootFiles(self):
    print "Root files and tree names currently being used:"
    tab = PrettyTable(["Root Files","Tree Names","Selection"])
    VAL_WRAP_WIDTH = 150
    for ind, val in enumerate(self.root_files): 
      wrapped_value_lines = wrap(self.root_selection[val] or '', VAL_WRAP_WIDTH) or ['']
      tab.add_row([val, self.tree_names[ind], wrapped_value_lines[0]])
      for subseq in wrapped_value_lines[1:]:
        tab.add_row(['', '',subseq])
    print tab

  def AddRootSelection(self,filenames,selection,OR=False,extra_name=None):
    if isinstance(filenames, str):
      filenames = [filenames]
    elif not isinstance(filenames, list):
      raise TypeError
    for filename in filenames:
      if extra_name != None: 
        if filename in self.root_files:
          index = self.root_files.index(filename) 
          self.root_files.pop(index)
          tree_name = self.tree_names[index]        
          self.tree_names.pop(index)
        else:
          for ind, i in enumerate(self.root_files):
            if filename in i: tree_name = self.tree_names[ind]
        self.root_selection["{} ({})".format(filename,extra_name)] = self.root_selection[filename]
        filename = "{} ({})".format(filename,extra_name) 
        self.root_files.append(filename)
        self.tree_names.append(tree_name)

      if filename in self.root_files:
        if self.root_selection[filename] != "(1)":
          if not OR:
            self.root_selection[filename] = "(({})&&({}))".format(self.root_selection[filename],selection)
          else:
            self.root_selection[filename] = "(({})||({}))".format(self.root_selection[filename],selection)
        else:
          self.root_selection[filename] = selection
        self.python_selection[filename] = "df.loc[({})]".format(''.join(self.__AllSplitStringsSteps__(self.root_selection[filename])))
      else:
        print("ERROR: Filename not found. Selection not added.")

  def AddBaselineRootSelection(self,selection):
    for key, val in self.root_selection.items():
      if val != "(1)":
        self.root_selection[key] = "(({})&&({}))".format(val,selection)
      else:
        self.root_selection[key] = selection
      self.python_selection[key] = "df.loc[({})]".format(''.join(self.__AllSplitStringsSteps__(self.root_selection[key])))


  def AddColumns(self,variables):
    if isinstance(variables, str) or isinstance(variables, unicode):
      variables = [variables]
    elif not isinstance(variables, list):
      raise TypeError

    for var in variables:
      self.scale_column[var] = {}
      modified = False
      delim = '\(|\)|'
      for ch in var:
        if not (ch.isdigit() or ch.isalpha() or ch in ["_"]): 
          modified = True
          delim += "\{}|".format(ch)
      if not modified:   
        self.columns.append(var)
      else:
        self.modified_columns.append(var)
        self.variables_for_modified += filter(None,re.split(delim, var))

    for var in self.variables_for_modified:
      if var in self.func_dict.keys(): self.variables_for_modified.remove(var)

  def ScaleColumn(self,files,column,scale,extra_name=None):
    if isinstance(files, str) or isinstance(files, unicode):
      files = [files]
    elif not isinstance(files, list):
      raise TypeError

    for f in files:
       if extra_name != None: f = "{} ({})".format(f,extra_name)
       if isinstance(scale, str) or isinstance(scale, unicode): self.__AddVariablesForSelection__(self.__GetSplitStrings__(scale))
       self.scale_column[column][f] = scale

  def GetDataframe(self):
    get_variables = set(self.columns+self.variables_for_selection+self.variables_for_modified)
    remove_list = list(set(get_variables)-set(self.columns))
    for ind, f in enumerate(self.root_files):
      
      # Get dataframe from root file
      if self.file_location[-1] == "/":
        tree = uproot.open(self.file_location+f.split(" (")[0]+self.file_ext)[self.tree_names[ind]]
        
      else:
        tree = uproot.open(self.file_location+"/"+f.split(" (")[0]+self.file_ext)[self.tree_names[ind]]      
      
      batches = 500
      events_per_batch = tree.numentries / batches
      remainder = tree.numentries // batches
      start = 0
      temp_df = pd.DataFrame()
      for i in range(batches):
        end = (i+1) * events_per_batch 
        df = 1*tree.pandas.df(get_variables,entrystart=start,entrystop=end)
        # Cut dataframe
        if not self.root_selection[f] == "(1)":
           df = eval(self.python_selection[f])
        temp_df = pd.concat([temp_df,df], ignore_index=True, sort=False)      
        start = end
      if (remainder != 0):
        df = 1*tree.pandas.df(get_variables,entrystart=(tree.numentries-remainder),entrystop=tree.numentries)
        if not self.root_selection[f] == "(1)":
           df = eval(self.python_selection[f])
        temp_df = pd.concat([temp_df,df], ignore_index=True, sort=False) 
      
      df = temp_df.copy(deep=True)
        
      # Calculate modified variables
      for mod_var in self.modified_columns:
        df.loc[:,mod_var] = eval(''.join(self.__AllSplitStringsSteps__(mod_var)))

      # Scale relevant column
      for key, val in self.scale_column.items():
        if f in val.keys():
          if isinstance(val[f], str) or isinstance(val[f], unicode):
            df.loc[:,key] =  eval("".join(self.__AllSplitStringsSteps__(val[f])))
          else:
            df.loc[:,key] = float(val[f])*df.loc[:,key]
      
      # Drop unneeded columns
      df = df.drop(remove_list,axis=1)

      # Combine dataframes
      if ind == 0:
        total_df = df.copy(deep=True)
      else:
        total_df = pd.concat([total_df,df], ignore_index=True, sort=False)

    self.dataframe = total_df

  def LoadRootFilesFromJson(self,json_file,variables,specific_file=None,quiet=False):
    with open(json_file) as jf:
      data = json.load(jf)

    with open(data["params_file"]) as pf:
      params = json.load(pf)

    self.file_location = data["file_location"]
    self.file_ext = data["file_ext"]

    self.AddColumns([data["weights"]]+variables)

    for en, opt in data["add_sel"].items():
      for f in opt["files"]:
        if specific_file == None or specific_file == f:
          self.AddRootFiles([f],tree_name="ntuple") 
          self.AddRootSelection([f],opt["sel"],extra_name=en)
          self.ScaleColumn([f],data["weights"],opt["weight"],extra_name=en)
          if f[-1] not in ["A","B","C","D","E","F","G","H"]:
            self.ScaleColumn([f],data["weights"],data["lumi"]*params[f]['xs']/params[f]['evt'],extra_name=en)
    self.AddBaselineRootSelection(data["baseline_sel"])  
    if not quiet: self.PrintRootFiles()

    self.GetDataframe()

    self.dataframe = self.dataframe.rename(columns={data["weights"]:'weights'})

  def NormaliseWeights(self,column="weights",total_scale=1000000,train_frac=None,test_frac=None):
    if train_frac == None and test_frac == None:
      self.dataframe.loc[:,column] = total_scale*self.dataframe.loc[:,column]/self.dataframe.loc[:,column].sum()
    else:
      train = self.dataframe.loc[(self.dataframe.loc[:,'train']==1)].copy(deep=True)
      test = self.dataframe.loc[(self.dataframe.loc[:,'train']==0)].copy(deep=True)
      train.loc[:,column] = train_frac*total_scale*train.loc[:,column]/train.loc[:,column].sum()
      test.loc[:,column] = test_frac*total_scale*test.loc[:,column]/test.loc[:,column].sum()
      self.dataframe =  pd.concat([train,test],ignore_index=True, sort=False)

  def TrainTestSplit(self,column="train",testsize=0.5,seed=42):
    train, test = train_test_split(self.dataframe,test_size=testsize, random_state=seed)
    train.loc[:,column] = 1
    test.loc[:,column] = 0
    self.dataframe =  pd.concat([train,test],ignore_index=True, sort=False)

  def SelectColumns(self,columns):
    self.AddColumns(columns)

    # Calculate modified variables
    for mod_var in self.modified_columns:
      self.dataframe.loc[:,mod_var] = eval(''.join(self.__AllSplitStringsSteps__(mod_var)).replace("df.","self.dataframe."))

    self.dataframe = self.dataframe.loc[:,columns]

  def DropModifiedVariables(self):
    self.dataframe = self.dataframe.drop(self.modified_columns,axis=1)

  def Copy(self):
    import copy
    return copy.deepcopy(self)

  def WriteToRoot(self,path,key='ntuple'):

    from collections import Counter
    from numpy.lib.recfunctions import append_fields
    from pandas import DataFrame, RangeIndex
    from root_numpy import root2array, list_trees
    from root_numpy import list_branches
    from root_numpy.extern.six import string_types

    column_name_counts = Counter(self.dataframe.columns)
    if max(column_name_counts.values()) > 1:
        raise ValueError('DataFrame contains duplicated column names: ' +
                         ' '.join({k for k, v in column_name_counts.items() if v > 1}))

    from root_numpy import array2tree
    # We don't want to modify the user's DataFrame here, so we make a shallow copy
    df_ = self.dataframe.copy(deep=False)

    # Convert categorical columns into something root_numpy can serialise
    for col in df_.select_dtypes(['category']).columns:
        name_components = ['__rpCaT', col, str(df_[col].cat.ordered)]
        name_components.extend(df_[col].cat.categories)
        if ['*' not in c for c in name_components]:
            sep = '*'
        else:
            raise ValueError('Unable to find suitable separator for columns')
        df_[col] = df_[col].cat.codes
        df_.rename(index=str, columns={col: sep.join(name_components)}, inplace=True)

    arr = df_.to_records(index=False)

    root_file = ROOT.TFile.Open(path, "recreate")
    if not root_file:
        raise IOError("cannot open file {0}".format(path))
    if not root_file.IsWritable():
        raise IOError("file {0} is not writable".format(path))

    # Navigate to the requested directory
    open_dirs = [root_file]
    for dir_name in key.split('/')[:-1]:
        current_dir = open_dirs[-1].Get(dir_name)
        if not current_dir:
            current_dir = open_dirs[-1].mkdir(dir_name)
        current_dir.cd()
        open_dirs.append(current_dir)

    # The key is now just the top component
    key = key.split('/')[-1]

    # If a tree with that name exists, we want to update it
    tree = open_dirs[-1].Get(key)
    if not tree:
        tree = None
    tree = array2tree(arr, name=key, tree=tree)
    tree.Write(key, ROOT.TFile.kOverwrite)
    root_file.Close()



