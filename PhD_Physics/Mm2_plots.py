import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm, colors

from functions import *

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text',usetex=True)
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

mass_invisible = float(sys.argv[1])

#-------------------------#
Particles = ['pi_sig', 'K_tag', 'pi_tag', 'track_tag']
momentaCMS = ['CMS_mcPX', 'CMS_mcPY', 'CMS_mcPZ', 'CMS_mcE']

columns = [f'{particle}_{p}' for particle in Particles for p in momentaCMS]
columns+= ['track_tag_mcPDG']

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

#-------------------#
path = '/home/jordan/Documents/Analisis_thesis/invisible_search/merged_nTracks4/merged/merged/signals'

#df_HadronicTag = RDF_to_pandas('b', f'{path}/ntuple_m{mass_invisible}_HadronicTag_0.root', columns=columns, cut=matchCut_HadronicTag)

df_SemilepTag_e  = RDF_to_pandas('b', f'{path}/ntuple_m{mass_invisible}_SemilepTag_0.root',  columns=columns, cut=f'{matchCut_SemilepTag} && abs(track_tag_mcPDG)==11')
df_SemilepTag_mu = RDF_to_pandas('b', f'{path}/ntuple_m{mass_invisible}_SemilepTag_0.root',  columns=columns, cut=f'{matchCut_SemilepTag} && abs(track_tag_mcPDG)==13')
df_SemilepTag = pd.concat([df_SemilepTag_e, df_SemilepTag_mu], ignore_index=True)

#------------------------------#

nbins=60

if mass_invisible==0.0:   Range=(-5, 5)       ; RMmin=(-5, 2);       RMmax=(-2, 5)
elif mass_invisible==1.0: Range=(-4, 6)       ; RMmin=(-4, 2);       RMmax=(-1, 5)
elif mass_invisible==2.0: Range=(0, 8)        ; RMmin=(0, 5);        RMmax=(2, 8)
elif mass_invisible==3.0: Range=(6, 12)       ; RMmin=(5.5, 10);     RMmax=(7.5, 12)
elif mass_invisible==4.0: Range=(14,18)       ; RMmin=(14, 16.5);    RMmax=(15, 18)
elif mass_invisible==5.0: Range=(24.5, 25.5)  ; RMmin=(24.5, 25.2);  RMmax=(24.8, 25.4)

binning = lambda Range, nbins: [round(Range[0] + (Range[1] - Range[0])*i/nbins, 2) for i in range(nbins + 1)]
Binning=binning(Range, nbins)


def plotting(Mmax2, Mmin2, mass_invisible):

    fig, ax = plt.subplots(figsize=(3.9, 3.0))

    plt.hist(Mmax2, bins=Binning, label='M$^{2}_{max}$', histtype='step')
    plt.hist(Mmin2, bins=Binning, label='M$^{2}_{min}$', histtype='stepfilled', alpha=0.3)

    plt.ylabel(f"Events/({round(Binning[1]-Binning[0], 3)})", fontsize=12)

    plt.text(0.16, 0.68, 'm$_{\\alpha}=$'+f'${mass_invisible}$', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=17)
    plt.text(0.19, 0.60, f'No.Events={len(Mmin2)}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=8)

    plt.legend(loc='upper left', fontsize=12)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.98, top=0.98)
    plt.savefig(f'/home/jordan/Documents/Analisis_thesis/invisible_search/MminMmax/masses_invisible/m{mass_invisible}.png', dpi=400)
    #plt.show()


def plot2D(Mmax2, Mmin2, mass_invisible):

    fig, ax = plt.subplots(figsize=(4.35, 3.0))

    # define color map
    cmap = cm.get_cmap("Oranges")

    x = plt.hist2d(Mmin2, Mmax2, bins=[100, 100], range=[RMmin, RMmax], cmap=cmap)

    # need to normalize because color maps are defined in [0, 1]
    norm = colors.Normalize(0, int(x[0].max() + 1))

    # plot colorbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Events per pixel')

    #fig.patch.set_facecolor('xkcd:white')

    plt.text(0.19, 0.92, 'm$_{\\alpha}=$' + f'${mass_invisible}$', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=17)
    plt.text(0.19, 0.82, f'No.Events={len(Mmin2)}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=8)

    plt.text(0.19, 0.77, f'CorrCoef={round(np.corrcoef(Mmax2, Mmin2)[0,1], 3)}',
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=8)

    plt.xlabel("M"+"$_{min}^{2}$", fontsize=14)
    plt.ylabel("M"+"$_{max}^{2}$", fontsize=14)

    plt.subplots_adjust(left=0.13, bottom=0.16, right=0.97, top=0.97)
    plt.savefig(f'/home/jordan/Documents/Analisis_thesis/invisible_search/MminMmax/masses_invisible/2D_m{mass_invisible}.png', dpi=400)
    #plt.show()


def main():

    df_SemilepTag['Mmax2'], df_SemilepTag['Mmin2'], eff = Calculation_Mm2(df=df_SemilepTag,
                                                                          nick_sig_fsp=['pi_sig'],
                                                                          nick_tag_fsp=['pi_tag', 'K_tag', 'track_tag'],
                                                                          X_mass=B_mass)

    Mmax2 = df_SemilepTag['Mmax2'].dropna()
    Mmin2 = df_SemilepTag['Mmin2'].dropna()
    # ------------------------------#
    plotting(Mmax2, Mmin2, mass_invisible=mass_invisible)
    plot2D(Mmax2, Mmin2, mass_invisible=mass_invisible)


if __name__ == '__main__':
    main()

