import matplotlib.pyplot as plt
import pandas as pd
import uproot
import numpy as np
import time
import sys
import warnings
import ROOT
from ROOT import RDataFrame
warnings.filterwarnings("ignore")

# conversion to df using uproot is faster than using RDataframe


def RDF_to_pandas(key:str, inputRootFiles:str, columns:list=None, cut=None):
    """
    :param key: Tree name of RootFiles
    :param inputRootFiles: {path}/files.root
    :param columns: Variables of interest
    :param cut: root-like cut to be applied.
    :return: Pandas dataframe
    """
    if cut: df = ROOT.RDataFrame(key, inputRootFiles).Filter(cut)
    else: df = ROOT.RDataFrame(key, inputRootFiles)

    if columns: df = df.AsNumpy(columns=columns)
    else: df = df.AsNumpy()

    return pd.DataFrame(df)

ti = time.time()
df_RData = RDF_to_pandas('b', '/home/jordan/Documents/Analisis_thesis/nZ4430/ntuples_mcSignal/ntuple_chB_JpsiToLep_ee_0.root')
tf = time.time()
print(tf-ti)

