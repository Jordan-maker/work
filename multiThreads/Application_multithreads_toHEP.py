import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import threading; nThreads = 8
from copy import copy as cp
import ROOT
from ROOT import RDataFrame
import sys

sys.path.append('/home/jordan/Documents/Analisis_thesis/invisible_search')
from functions import RDF_to_pandas
import warnings
warnings.simplefilter("ignore")

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text',usetex=True)
plt.rcParams.update({
        'axes.grid': False,
        'grid.linestyle': '-',
        'grid.alpha': 0.2,
        'lines.markersize': 4.0,
        'xtick.minor.visible': False,
        'xtick.direction': 'in',
        'xtick.major.size': 3.8,
        'xtick.minor.size': 1.8,
        'xtick.top': False,
        'ytick.minor.visible': False,
        'ytick.direction': 'in',
        'ytick.major.size': 3.8,
        'ytick.minor.size': 1.8,
        'ytick.right': False,
    })

#--------------------------#
Dmeson = 'D'            # {'D', 'Dstar_pi0', 'Dstar_gamma'}
tag = 'HadronicTag'     #{'HadronicTag', 'SemilepTag'}
#--------------------------#

# Loading datasets
varLoad = ['E_ECL', 'nKLMClusters']

###### CUTS TO APPLY TO DATAFRAMES ##########

# event-based Cuts
baseCut =f'nPi0 == 0 && nGoodTracks == 4'
baseCut+=f' && track_tag_genMotherPDG != 130' # Veto K_L0

# General cuts
D_Mass_Cut = 'D_tag_M > 1.80 && D_tag_M < 1.90'
B_Mass_Cut = 'B_tag_M > 5.27 && B_tag_M < 5.29'
B_Mass_Cut_partial = 'B_tag_M < 5.29'
B_tag_CMS_mcP = 'B_tag_CMS_mcP > 0.2 && B_tag_CMS_mcP < 0.45'
B_tag_lab_mcP = 'B_tag_mcP > 1.0'
#photonsEnergyCMS = 'genTotalPhotonsEnergyOfEvent < 5.28' # To reduce B0 -> gamma gamma

#
optCut_htag = f'{baseCut} && {D_Mass_Cut} && {B_tag_CMS_mcP} && {B_tag_lab_mcP} && {B_Mass_Cut}'
optCut_stag = f'{baseCut} && {D_Mass_Cut} && {B_tag_CMS_mcP} && {B_tag_lab_mcP} && {B_Mass_Cut_partial}'

# perfect PID
partial_matchCut = 'abs(K_sig_mcPDG) == 321 && abs(pi_tag_mcPDG) == 211 && abs(K_tag_mcPDG) == 321'
partial_matchCut_HadronicTag = f'{partial_matchCut} && abs(track_tag_mcPDG) == 211'
partial_matchCut_SemilepTag  = f'{partial_matchCut} && (abs(track_tag_mcPDG)==11 || abs(track_tag_mcPDG)==13)'

# Matching Cuts
matchCut_HadronicTag = f'{partial_matchCut_HadronicTag} && abs(K_sig_genMotherPDG) == 521'
matchCut_SemilepTag  = f'{partial_matchCut_SemilepTag}  && abs(K_sig_genMotherPDG) == 521'

#------ PATHS -------#
path_base='/home/jordan/Documents/Analisis_thesis/invisible_search/neutron_study'

path_ntuples  = f'{path_base}/ntuples'
path_outPlots = f'{path_base}/FOM/plots'

if tag == 'HadronicTag':
    df_Knunu_HadronicTag = RDF_to_pandas('b', f'{path_ntuples}/ntuple_Knunu_HadronicTag_{Dmeson}.root', varLoad, cut=f'{matchCut_HadronicTag} && {optCut_htag}')
    df_Knn_HadronicTag = RDF_to_pandas('b', f'{path_ntuples}/ntuple_Knn_HadronicTag_{Dmeson}.root', varLoad, cut=f'{matchCut_HadronicTag} && {optCut_htag}')

    Signal = cp(df_Knunu_HadronicTag)
    Background = cp(df_Knn_HadronicTag)

else:
    df_Knunu_SemilepTag  = RDF_to_pandas('b', f'{path_ntuples}/ntuple_Knunu_SemilepTag_{Dmeson}.root',  varLoad, cut=f'{matchCut_SemilepTag} && {optCut_stag}')
    df_Knn_SemilepTag  = RDF_to_pandas('b', f'{path_ntuples}/ntuple_Knn_SemilepTag_{Dmeson}.root',  varLoad, cut=f'{matchCut_SemilepTag} && {optCut_stag}')

    Signal = cp(df_Knunu_SemilepTag)
    Background = cp(df_Knn_SemilepTag)

#print(len(df_Knunu_HadronicTag), len(df_Knunu_SemilepTag))
#print(len(df_Knn_HadronicTag), len(df_Knn_SemilepTag))
#sys.exit()

############################# FILL POSIBLE VALUES ###############################

class features:

    def __init__(self, name:str, hack:str, name_latex:str=None, step:float=0.01, df=Signal):

        self.name = name
        self.hack = hack
        self.nickname = f'{name}_{hack}'
        self.step = step
        self.num_decimals = abs(int(math.log10(step)))

        if name_latex is None:
            self.name_latex = self.nickname
        else:
            self.name_latex = '$' + name_latex + '\_' + hack + '$'

        self.mean_value = df.eval(self.name).mean()
        self.max_value  = df.eval(self.name).max()
        self.min_value  = df.eval(self.name).min()

        #-------------------------------#
        if hack == 'up':

            self.sign = '<='

            N_divisions_up = int(round(abs(self.max_value - self.min_value)/step, 0))
            self.values = [round(self.max_value-j*step, self.num_decimals) for j in range(N_divisions_up)]
            self.cut = 100   #max(self.values) # initial value for the cut

        if hack == 'low':

            self.sign = '>='

            N_divisions_low = int(round(abs(self.max_value - self.min_value)/step, 0))
            self.values = [round(self.min_value+j*step, self.num_decimals) for j in range(N_divisions_low)]
            self.cut = -100   #min(self.values) # initial value for the cut

        length = len(self.values)
        self.subvalues = [self.values[i*length//nThreads: (i+1)*length//nThreads] for i in range(nThreads)]

#--------------------------------------------------------------------------#

nKLMClusters_up = features(name='nKLMClusters', hack='up', name_latex="nKLMClusters", step=1.0)
E_ECL_up = features(name='E_ECL', hack='up', name_latex="E_{ECL}", step=0.01)

variables_to_FOM = [nKLMClusters_up, E_ECL_up]

###########################################################################

def FOM(S, B):
    FOM = S/math.sqrt(S+B)
    return FOM

#------------------------- INITIAL VALUES -----------------------------#

initial_S = Signal.eval(varLoad[0]).count()
initial_B = Background.eval(varLoad[0]).count()

best_variable_by_step = ['PreSel.']
best_cut_by_step = ['None']
best_FOM_by_step = [FOM(S=initial_S, B=initial_B)]

#----------------------------------------------------------------------#

def bestCut(variable:classmethod, add_cut:str=None):

    fom = {}
    def parallelizing(k):

        for value in variable.subvalues[k]:

            ind_cut = f'{variable.name} {variable.sign} {value}'

            if add_cut is None:
                cut = ind_cut
            else:
                cut = f'{ind_cut} and {add_cut}'

            S = Signal.query(cut).eval(variable.name).count()
            B = Background.query(cut).eval(variable.name).count()
            f = FOM(S, B)
            fom[f] = value
            print(f"cut: {cut}, FOM: {f}")

        return fom

    threads = []
    for k in range(len(variable.subvalues)):
        t = threading.Thread(target=parallelizing, args=(k,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    #-----------------#
    max_FOM = max(fom)
    best_cut = fom[max_FOM]
    print(f"\nFom_Max = {max_FOM}, best_cut = {best_cut}\n")
    return max_FOM, best_cut, fom

###############################################################

def joiningCut(listOfVariables:list):

    joined_cut = ''
    for variable in listOfVariables:

        joined_cut += f'{variable.name} {variable.sign} {variable.cut}'

        if variable != listOfVariables[-1]:
            joined_cut += ' and '

    return joined_cut


def definingOrder(listOfVariables:list, add_cut=None, step=0):

    info_list = []
    for actual_variable in listOfVariables:

        if step > 0 and len(listOfVariables)>1:
            add_cut = joiningCut(listOfVariables=[var for var in listOfVariables if var != actual_variable])

        fom, cut, _ = bestCut(actual_variable, add_cut=add_cut)
        info_list.append((fom, actual_variable.name, actual_variable.nickname, cut))

    best_variable = max(info_list)
    eval(best_variable[2]).cut = best_variable[3]
    print("\n\n", best_variable, "\n\n")
    return best_variable

#-------------------------------------------------------------------------------------------#

def run(listOfVariables:list, fixed:str, step:int=0):

    best_variable = definingOrder(listOfVariables=listOfVariables, step=step)
    fixed_value = f'{best_variable[1]} {eval(best_variable[2]).sign} {best_variable[3]}'

    best_variable_by_step.append(best_variable[2])
    best_cut_by_step.append(fixed_value)
    best_FOM_by_step.append(best_variable[0])

    if step > 0 and fixed:
        fixed_value += ' and ' + fixed

    return fixed_value

#############################################################################################

def plot_FOM(var_list, FOM_list, name_output_image:str='FOM', include_preselection=False): # Both entries must have the same length

    if include_preselection == True:

        var_list = [var_list[0]] + [eval(var).name_latex for var in var_list[1:]]
        FOM_list = [FOM_list[0]] + [FOM for FOM in FOM_list[1:]]

    else:
        var_list = [eval(var).name_latex for var in var_list[1:]]
        FOM_list = [FOM for FOM in FOM_list[1:]]

    fig, h = plt.subplots(figsize=(3.6, 3.6))

    h.set_ylabel("FOM", fontsize=14)

    plt.xticks(rotation=90, fontsize=9.5)
    plt.yticks(fontsize=9.5)

    plt.subplots_adjust(left=0.14, bottom=0.35, right=0.98, top=0.945)

    plt.grid(True, color='gray', ls='-', lw=0.3, alpha=0.12)
    h.plot(var_list, FOM_list, color='r', marker='o', markersize=3.2, linestyle='-', linewidth=0.6, alpha=0.6)

    plt.savefig(f'{path_outPlots}/{name_output_image}_{tag}.png', dpi=300)
    #plt.show()

#############################################################################################
#---------#
    nbins = 60
    binning = lambda Range, nbins: [round(Range[0] + (Range[1] - Range[0]) * i / nbins, 2) for i in range(nbins + 1)]

def plot_variable(variable:str, cut=None, xlabel:str='', xline1:float=None, xline2:float=None):

    if cut:
        df_signal = Signal.query(cut).eval(variable)
        df_background = Background.query(cut).eval(variable)
    else:
        df_signal = Signal.eval(variable)
        df_background = Background.eval(variable)

    #---------#
    if variable == 'E_ECL':          Binning = binning(Range=(0, 8), nbins=nbins)
    elif variable == 'nKLMClusters': Binning =binning(Range=(0, 12), nbins=nbins)

    label_signal = "$K^{+}\\nu\\bar{\\nu}$"
    label_background = "$K^{+}n^{0}\\bar{n}^{0}$"

    #----#
    plt.subplots(figsize=(4.7, 3.5))

    plt.hist(df_signal,     bins=Binning, label=label_signal, color='tab:blue', histtype='step')
    plt.hist(df_background, bins=Binning, label=label_background, color='tab:orange', histtype='step')

    plt.xlabel(xlabel, fontsize=14.5)
    plt.ylabel(f'Candidates', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if xline1: plt.axvline(x=xline1, color='darkred', lw=0.7)
    if xline2: plt.axvline(x=xline2, color='darkred', lw=0.7)

    plt.legend(loc='best', fontsize=10.5)

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.98, top=0.98)
    plt.savefig(f'{path_outPlots}/{variable}_{tag}.png', dpi=400)
    #plt.show()

#############################################################################################

def plot_variable_scanning(variable:str, fom:dict, best_cut:float, xlabel:str=''):

    plt.subplots(figsize=(4.0, 3.2))

    plt.scatter(fom.values(), fom.keys(), s=2, edgecolors='tab:red')

    plt.xlabel(xlabel, fontsize=14.5)
    plt.ylabel('FOM($S/\sqrt{S+B}$)', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title(tag, fontsize=16)

    if best_cut: plt.axvline(x=best_cut, color='darkred', lw=0.7)

    #plt.legend(loc='best', fontsize=10.5)

    plt.subplots_adjust(left=0.145, bottom=0.15, right=0.98, top=0.92)
    plt.savefig(f'{path_outPlots}/FOM_scanning/{variable}_{tag}.png', dpi=400)
    #plt.show()


#############################################################################################
# Efficiency calculation

def efficiency_optCuts(cut):

    Signal_after = Signal.query(cut)
    Background_after = Background.query(cut)

    eff_Signal = len(Signal_after)/len(Signal)
    eff_Background = len(Background_after)/len(Background)

    return eff_Signal, eff_Background


####################################

def main(listOfVariables:list, plot_fom=True, plot_corr_M=True, save_info=True):

    #----------------------------------------------------------#
    fixed_cut = run(listOfVariables=listOfVariables, fixed='', step=0)
    for step in range(1, 100):

        fixed_cut = run(listOfVariables=listOfVariables, fixed=fixed_cut, step=step)

        if best_FOM_by_step[step] == best_FOM_by_step[step-1]:
            break

    del best_variable_by_step[-1]; del best_cut_by_step[-1]; del best_FOM_by_step[-1]

    opt_cuts = joiningCut(listOfVariables=listOfVariables)

    #--------------------#
    if save_info == True:

        file = open(f"summary_FOM_{tag}.txt", "w")
        file.write("\nbest_variable_by_step = " + str(best_variable_by_step))
        file.write("\nbest_cut_by_step = " + str(best_cut_by_step))
        file.write("\nbest_FOM_by_step = " + str(best_FOM_by_step))
        file.write("\nopt_cuts = " + str(opt_cuts))
        file.close()
        print("summary.txt was generated succesfully!")
    #--------------#

    if plot_fom == True:
        plot_FOM(best_variable_by_step, best_FOM_by_step, include_preselection=True, name_output_image='FOM')

####################################

if __name__ == '__main__':
    
    main(listOfVariables=variables_to_FOM, plot_fom=True, save_info=True)

    # HadronicTag
    plot_variable(variable='E_ECL', xlabel='$E_{ECL}$', xline1=4.0)
    plot_variable(variable='nKLMClusters', cut='E_ECL < 4.0', xlabel='nKLMClusters', xline1=5.2)

    # SemilepTag
    #plot_variable(variable='nKLMClusters', xlabel='nKLMClusters', xline1=4.2)
    #plot_variable(variable='E_ECL', cut='nKLMClusters <= 4.0', xlabel='$E_{ECL}$', xline1=4.13)

    #______________________________#
    # efficiency of optimized cuts:

    # HadronicTag
    eff_Signal, eff_Background = efficiency_optCuts(cut='E_ECL <= 4.0 and nKLMClusters <= 5.0')
    # results = (0.9246, 0.5643)

    # SemilepTag
    #eff_Signal, eff_Background = efficiency_optCuts(cut='nKLMClusters <= 4.0 and E_ECL <= 4.13')
    # results = (0.9264, 0.5389)

    #print(eff_Signal, eff_Background)

    #______________________________#
    # Plots FOM scanning:

    # -- HadronicTag --#
    tag = 'HadronicTag'

    # 1st step:
    _, best_cut, fom = bestCut(variable=E_ECL_up, add_cut=None)
    plot_variable_scanning(variable='E_ECL', fom=fom, best_cut=best_cut, xlabel='$E_{ECL}$ [GeV]')

    # 2nd step:
    _, best_cut, fom = bestCut(variable=nKLMClusters_up, add_cut='E_ECL <= 4.0')
    plot_variable_scanning(variable='nKLMClusters', fom=fom, best_cut=best_cut, xlabel='nKLMClusters')


    #-- SemilepTag --#
    #tag = 'SemilepTag'

    # 1st step:
    #_, best_cut, fom = bestCut(variable=nKLMClusters_up, add_cut=None)
    #plot_variable_scanning(variable='nKLMClusters', fom=fom, best_cut=best_cut, xlabel='nKLMClusters')

    # 2nd step:
    #_, best_cut, fom = bestCut(variable=E_ECL_up, add_cut='nKLMClusters <= 4.0')
    #plot_variable_scanning(variable='E_ECL', fom=fom, best_cut=best_cut, xlabel='$E_{ECL}$ [GeV]')






