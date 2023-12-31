import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import sys
import warnings
import ROOT
from ROOT import RDataFrame
from functions import RDF_to_pandas

warnings.filterwarnings("ignore")

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams.update({
    'axes.grid': False,
    'grid.linestyle': '-',
    'grid.alpha': 0.2,
    'lines.markersize': 4.0,
    'xtick.minor.visible': True,
    'xtick.direction': 'in',
    'xtick.major.size': 3.8,
    'xtick.minor.size': 1.8,
    'xtick.top': False,
    'ytick.minor.visible': True,
    'ytick.direction': 'in',
    'ytick.major.size': 3.8,
    'ytick.minor.size': 1.8,
    'ytick.right': False,
})
#--------------------------#

Particles = ['pi_sig', 'K_tag', 'pi_tag', 'track_tag', 'D_tag', 'B_sig', 'B_tag']
varsToLoad = ['mcPDG', 'genMotherPDG', 'genMotherPDG_1']

columns = [f'{particle}_{var}' for particle in Particles for var in varsToLoad]

varsToPlot = ['D_tag_M', 'B_tag_M'] + ['nTracks_', 'nGammas', 'nPi0', 'nEta', 'nEtaPrime']

varsToPlot += ['foxWolframR2', 'thrust', 'cosToThrustOfEvent', 'thrustAxisCosTheta']
varsToPlot += ['genMissingMass2OfEvent', 'genMissingEnergyOfEventCMS', 'genMissingMomentumOfEventCMS', 'genTotalPhotonsEnergyOfEvent', 'genVisibleEnergyOfEventCMS']
varsToPlot += ['Mbc', 'deltaE', 'B_sig_mcP', 'B_sig_CMS_mcP']
varsToPlot += ['B_tag_M', 'B_tag_mcP', 'B_tag_CMS_mcP', 'D_tag_M', 'D_tag_mcP', 'D_tag_CMS_mcP']
varsToPlot += ['pi_sig_mcP', 'pi_sig_CMS_mcP', 'K_tag_mcP', 'K_tag_CMS_mcP', 'pi_tag_mcP', 'pi_tag_CMS_mcP', 'track_tag_mcP', 'track_tag_CMS_mcP']

columns += varsToPlot
#------------#

nPi0Cut = 'nPi0 == 0'
nEtaCut = 'nEta == 0'
nEtaPrimeCut = 'nEtaPrime == 0'
nTracksCut = 'nTracks_ == 4'

baseCut=f'{nPi0Cut} && {nEtaCut} && {nEtaPrimeCut} && {nTracksCut}'

# Cuts from FOM
D_Mass_Cut = 'D_tag_M > 1.80 && D_tag_M < 1.90'
B_Mass_Cut = 'B_tag_M > 5.27 && B_tag_M < 5.29'
B_Mass_Cut_partial = 'B_tag_M < 5.29'
B_tag_mcP_CMS = 'B_tag_CMS_mcP > 0.2 && B_tag_CMS_mcP < 0.46'


finalCut_htag = f'{baseCut} && {D_Mass_Cut} && {B_tag_mcP_CMS} && {B_Mass_Cut}'
finalCut_stag = f'{baseCut} && {D_Mass_Cut} && {B_tag_mcP_CMS} && {B_Mass_Cut_partial}'

# Matching Cuts

matchCut_HadronicTag = 'abs(pi_sig_mcPDG) == 211 && abs(pi_tag_mcPDG) == 211 && abs(track_tag_mcPDG) == 211 && abs(K_tag_mcPDG) == 321 && ' \
                       'abs(D_tag_mcPDG) == 421 && abs(B_sig_mcPDG) == 521 && abs(B_tag_mcPDG) == 521 && ' \
                       'abs(pi_sig_genMotherPDG) == 521 && abs(track_tag_genMotherPDG) == 521 && ' \
                       'track_tag_genMotherPDG == D_tag_genMotherPDG && ' \
                       'abs(K_tag_genMotherPDG) == 421 && K_tag_genMotherPDG == pi_tag_genMotherPDG && pi_tag_mcPDG*track_tag_mcPDG < 0 && ' \
                       'abs(B_sig_genMotherPDG) == 300553 && B_sig_genMotherPDG == B_tag_genMotherPDG && B_sig_mcPDG*B_tag_mcPDG < 0'
matchCut_SemilepTag  = 'abs(pi_sig_mcPDG) == 211 && abs(pi_tag_mcPDG) == 211 && (abs(track_tag_mcPDG)==11 || abs(track_tag_mcPDG)==13) && ' \
                      'abs(K_tag_mcPDG) == 321 && abs(D_tag_mcPDG) == 421 && abs(B_sig_mcPDG) == 521 && abs(B_tag_mcPDG) == 521 && ' \
                      'abs(pi_sig_genMotherPDG) == 521 && abs(track_tag_genMotherPDG) == 521 && ' \
                      'track_tag_genMotherPDG == D_tag_genMotherPDG && ' \
                      'abs(K_tag_genMotherPDG) == 421 && K_tag_genMotherPDG == pi_tag_genMotherPDG && pi_tag_mcPDG*track_tag_mcPDG > 0 && ' \
                      'abs(B_sig_genMotherPDG) == 300553 && B_sig_genMotherPDG == B_tag_genMotherPDG && B_sig_mcPDG*B_tag_mcPDG < 0'

# perfect PID

partial_matchCut = 'abs(pi_sig_mcPDG) == 211 && abs(pi_tag_mcPDG) == 211 && abs(K_tag_mcPDG) == 321'
partial_matchCut_HadronicTag = f'{partial_matchCut} && abs(track_tag_mcPDG) == 211'
partial_matchCut_SemilepTag  = f'{partial_matchCut} && (abs(track_tag_mcPDG)==11 || abs(track_tag_mcPDG)==13)'

# -------------#
path_MC = '/home/jordan/Documents/Analisis_thesis/invisible_search/merged_nTracks4/merged/merged'  # Generated by me
path_MC_Grid = '/home/jordan/Documents/Analisis_thesis/invisible_search/merged_1ab_v2'             # from GRID samples
path_outPlots = '/home/jordan/Documents/Analisis_thesis/invisible_search/plots_variables'

# ntuple_signal_<>.root can be changed by ntuple_pinunu_<>.root

df_signal_HadronicTag = RDF_to_pandas(key='b', inputRootFiles=f'{path_MC}/ntuple_pinunu_HadronicTag_0.root', columns=columns, cut=f'{matchCut_HadronicTag} && {finalCut_htag}')  # 10K generated events
df_signal_SemilepTag  = RDF_to_pandas(key='b', inputRootFiles=f'{path_MC}/ntuple_pinunu_SemilepTag_0.root',  columns=columns, cut=f'{matchCut_SemilepTag}  && {finalCut_stag}')   # 10K generated events

df_charged_HadronicTag = RDF_to_pandas('b', f'{path_MC_Grid}/ntuple_charged_htag_1ab.root', columns, cut=f'{partial_matchCut_HadronicTag} && {finalCut_htag}')
df_charged_SemilepTag  = RDF_to_pandas('b', f'{path_MC_Grid}/ntuple_charged_stag_1ab.root', columns, cut=f'{partial_matchCut_SemilepTag}  && {finalCut_stag}')

df_mixed_HadronicTag = RDF_to_pandas('b', f'{path_MC_Grid}/ntuple_mixed_htag_1ab.root', columns, cut=f'{partial_matchCut_HadronicTag} && {finalCut_htag}')
df_mixed_SemilepTag  = RDF_to_pandas('b', f'{path_MC_Grid}/ntuple_mixed_stag_1ab.root', columns, cut=f'{partial_matchCut_SemilepTag}  && {finalCut_stag}')

df_qqbar_HadronicTag = RDF_to_pandas('b', f'{path_MC_Grid}/ntuple_qqbar_htag_1ab.root', columns, cut=f'{partial_matchCut_HadronicTag} && {finalCut_htag}')
df_qqbar_SemilepTag  = RDF_to_pandas('b', f'{path_MC_Grid}/ntuple_qqbar_stag_1ab.root', columns, cut=f'{partial_matchCut_SemilepTag}  && {finalCut_stag}')

df_taupair_HadronicTag = RDF_to_pandas('b', f'{path_MC_Grid}/ntuple_taupair_htag_1ab.root', columns, cut=f'{partial_matchCut_HadronicTag} && {finalCut_htag}')
df_taupair_SemilepTag  = RDF_to_pandas('b', f'{path_MC_Grid}/ntuple_taupair_stag_1ab.root', columns, cut=f'{partial_matchCut_SemilepTag}  && {finalCut_stag}')

#-------------#
names  = ['signal'] + ['charged', 'mixed', 'qqbar', 'taupair']
labels = ['signal'] + ['$B^{+}B^{-}$', '$B^{0}\\bar{B^{0}}$', '$q\\bar{q}$', '$\\tau\\bar{\\tau}$']
colors = ['tab:red'] + ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
zorder = [5, 3, 4, 1, 2]

# Plots
nbins = 60
binning = lambda Range, nbins: [round(Range[0] + (Range[1] - Range[0]) * i / nbins, 2) for i in range(nbins + 1)]

#----------------------------------------#
D_tag_M_Binning = binning((1.5, 2.0),  nbins)
nGammas_Binning = binning((0, 15), nbins)
nTracks__Binning = binning((3, 5), nbins)
nPi0_Binning = binning((0, 15), nbins)
nEta_Binning = binning((0, 5), nbins)
nEtaPrime_Binning = binning((0, 5), nbins)

cosToThrustOfEvent_Binning = binning((-1, 1), nbins)
thrustAxisCosTheta_Binning = 60

foxWolframR2_Binning = binning((0, 1), nbins)
thrust_Binning = binning((0.55, 1.0), nbins)
genMissingMass2OfEvent_Binning = binning((0, 60), nbins)
genMissingEnergyOfEventCMS_Binning = binning((0, 9), nbins)
genMissingMomentumOfEventCMS_Binning = binning((0, 5), nbins)
genTotalPhotonsEnergyOfEvent_Binning = binning((0, 4), nbins)
genVisibleEnergyOfEventCMS_Binning = binning((0, 11), nbins)

Mbc_Binning = binning((0, 5.4), nbins)
deltaE_Binning = binning((-3, 3), nbins)

B_sig_mcP_Binning = binning((0, 4), nbins)
B_sig_CMS_mcP_Binning = binning((0, 3), nbins)
B_tag_mcP_Binning = binning((0, 3), nbins)
B_tag_CMS_mcP_Binning = binning((0, 1.0), nbins)
B_tag_M_Binning = binning((1, 6), nbins)
D_tag_mcP_Binning = binning((0, 4), nbins)
D_tag_CMS_mcP_Binning = binning((0, 3), nbins)
pi_sig_mcP_Binning = binning((0, 7), nbins)
pi_sig_CMS_mcP_Binning = binning((0, 5.5), nbins)
pi_tag_mcP_Binning = binning((0, 7), nbins)
pi_tag_CMS_mcP_Binning = binning((0, 5.5), nbins)
K_tag_mcP_Binning = binning((0, 7), nbins)
K_tag_CMS_mcP_Binning = binning((0, 5.5), nbins)
track_tag_mcP_Binning = binning((0, 5), nbins)
track_tag_CMS_mcP_Binning = binning((0, 5), nbins)

#------------#

def plotting(variable: str, Binning: list, nameVar: str, xlabel: str = "", ylabel: str = "", locLegend: str = 'best', LogScale=True):

    for tag in ['HadronicTag', 'SemilepTag']:

        plt.subplots(figsize=(4.3, 3.5))

        for name, label, color, z in zip(names, labels, colors, zorder):

            if name == 'signal':
                df = eval(f'df_{name}_{tag}').eval(variable); histtype = 'stepfilled'; alpha = 0.5
            else:
                df = eval(f'df_{name}_{tag}').eval(variable); histtype = 'step';       alpha = None

            plt.hist(df, bins=Binning, histtype=histtype, alpha=alpha, color=color, zorder=z, label=label)

        if LogScale:
            plt.yscale('log')

        plt.title(tag, fontsize=16)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.legend(loc=locLegend)
        plt.subplots_adjust(left=0.14, bottom=0.15, right=0.98, top=0.9)
        plt.savefig(f'{path_outPlots}/{nameVar}_{tag}.png', dpi=400)


#-----------#
def main():
    for var in varsToPlot:
        plotting(variable=var, Binning=eval(f'{var}_Binning'), nameVar=var, xlabel=var.replace("_", "\_"), ylabel="Candidates", locLegend="best")

if __name__ == '__main__':
    main()


