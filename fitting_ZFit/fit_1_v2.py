import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import zfit
from zfit import z

import root_pandas as rp
import glob as glob
import numpy as np
import math
import warnings
warnings.simplefilter("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import mplhep

from inv_mass import *


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
mpl.rc('hatch', linewidth=0.3)


sample = rp.read_root(glob.glob('/home/jordan/Documentos/exotic_magister/rootfiles_nX3872/complete/ntuple_*.root'))
cut = "foxWolframR2 < 0.4 and nGoodTracks > 4 and B_Mbc > 5.27 and abs(B_deltaE) < 0.02 and 3.07 < Jpsi_M < 3.117 and Jpsi_p_CMS < 2.0"
mX = inv_mass(particles=["Jpsi", "pi1", "pi2"], mass_PDG=[3.09690, 0.13957061, 0.13957061], df=sample, cut=cut)


def plot_hist(df, nbins:int=40):

    fig, h = plt.subplots(figsize=(5, 3.8))

    mplhep.histplot(np.histogram(df, bins=nbins), yerr=True, color='black', histtype='errorbar',
                    markersize=3.6, capsize=2, markeredgewidth=0.9, zorder=1, elinewidth=0.9)

    plt.hist(df, color='black', bins=nbins, histtype="stepfilled", alpha=0.1)

    bin_size = (df.max() - df.min())/nbins
    bin_size = round(bin_size, 3)

    plt.ylabel(f"Events/({bin_size} GeV/c$^{2}$)", fontsize=15)
    plt.xlabel("$m(J/\psi\pi^{-}\pi^{+})$ [GeV/c$^{2}$]", fontsize=15)

    plt.subplots_adjust(left=0.12, bottom=0.14, right=0.975, top=0.95)
    plt.savefig('/home/jordan/Documentos/exotic_magister/Macros/nX3872_crossCheck/mX.png', dpi=300)


#plot_hist(df=mX)

obs = zfit.Space(obs='mX', limits=(3.65, 3.73))

# Gaussian1 components
mu1 = zfit.Parameter("mu1", 3.6857, 3.67, 3.70, step_size=0.001, floating=False)
sigma1 = zfit.Parameter("sigma1", 0.00396, 0.001, 0.01, floating=True)
gauss1 = zfit.pdf.Gauss(mu=mu1, sigma=sigma1, obs=obs)

# Gaussian2 components
mu2 = zfit.Parameter("mu2", 3.6857, 3.67, 3.70, step_size=0.001, floating=False)
sigma2 = zfit.Parameter("sigma2", 0.001558, 0.001, 0.005, floating=True)
gauss2 = zfit.pdf.Gauss(mu=mu2, sigma=sigma2, obs=obs)

## Gaussian3 components
#mu3 = zfit.Parameter("mu3", 3.6857, 3.67, 3.70, step_size=0.001, floating=False)
#sigma3 = zfit.Parameter("sigma3", 0.010580, 0.01, 0.1, floating=True)
#gauss3 = zfit.pdf.Gauss(mu=mu3, sigma=sigma3, obs=obs)

# 1st degree Chebyshev components
coef1 = zfit.Parameter("coef1", 1.0, -10, 10, floating=True)
chebyschev = zfit.pdf.Chebyshev(coeffs=[coef1], obs=obs)

# Model extended

c1 = zfit.Parameter("c1", 0.3, 0, 1)
c2 = zfit.Parameter("c2", 0.3, 0, 1)
#c3 = zfit.Parameter("c3", 0.3, 0, 1)
model = zfit.pdf.SumPDF([gauss1, gauss2, chebyschev], [c1, c2])

yield_ = zfit.Parameter('yield_model', 200, 0, 1000, step_size=1, floating=True)
model_ext = model.create_extended(yield_)

# data
data = zfit.Data.from_numpy(obs=obs, array=mX.to_numpy())


def minimizing(model, data):

    nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
    minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
    return minimizer.minimize(nll)

def plot_figure(obs, model, pdf1, pdf2, pdf3, data, yield_, x_axe:tuple=(3.65, 3.73)):

    fig, h = plt.subplots(figsize=(5, 3.8))

    x = np.linspace(*x_axe, 1000)
    y_model = model.pdf(x).numpy()
    y_pdf1 = (pdf1.pdf(x)*c1).numpy()
    y_pdf2 = (pdf2.pdf(x)*c2).numpy()
    #y_pdf3 = (pdf2.pdf(x)*c3).numpy()
    y_pdf3 = (pdf3.pdf(x)*(1-c1-c2)).numpy()
    data_np = data[:, 0].numpy()

    nbins = int( (x[-1]-x[0])/0.002 )
    #nbins=40
    scaling = (yield_/nbins)*obs.area()

    # plot the data
    plt.hist(data_np, color='black', bins=nbins, histtype="stepfilled", alpha=0.1)

    plt.plot(x, y_model*scaling, label="Model", linewidth=0.9)
    plt.plot(x, y_pdf1*scaling, label="Gauss1", linewidth=0.9)
    plt.plot(x, y_pdf2*scaling, label="Gauss2", linewidth=0.9)
    #plt.plot(x, y_pdf3*scaling, label="Gauss3", linewidth=0.9)
    plt.plot(x, y_pdf3*scaling, label="Chebyshev 1st", linewidth=0.9)

    mplhep.histplot(np.histogram(data_np, bins=nbins), yerr=True, color='black', histtype='errorbar',
                                 markersize=3.6, capsize=2, markeredgewidth=0.9, zorder=1, elinewidth=0.9)

    plt.ylabel("Events/(2 MeV/$c^{2}$)", fontsize=13)
    plt.xlabel("$m(J/\psi\pi^{-}\pi^{+})$", fontsize=14)

    plt.subplots_adjust(left=0.11, bottom=0.15, right=0.97, top=0.94)
    plt.legend()

    plt.savefig('/home/jordan/Documentos/exotic_magister/Macros/nX3872_crossCheck/mPsi2S_2Gauss.png', dpi=300)
    #plt.show()
####################################

result = minimizing(model_ext, data)
print(result)
result.errors()

plot_figure(obs=obs, model=model, pdf1=gauss1, pdf2=gauss2, pdf3=chebyschev, data=data, yield_=result.params[yield_]['value'])
