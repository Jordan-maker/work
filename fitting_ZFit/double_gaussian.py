import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import root_pandas as rp
import glob as glob
import numpy as np
import zfit
from zfit import z
from pdf_argus import Argus
import mplhep


plt.rcParams["font.serif"] = "cmr10"


dataframe = rp.read_root(glob.glob('/home/jordan/Documentos/exotic_magister/root_files_BG/MC13a_BGx1_100fb/complete/ntuple_*.root'), columns= ['foxWolframR2'])
df = dataframe.foxWolframR2

#-----------------------------------------#

obs = zfit.Space(obs='deltaE', limits=(df.min(), df.max()))

# parameters

mu_1 = zfit.Parameter("mu1", 0.2, 0, 0.4, step_size=0.001, floating=True)
mu_2 = zfit.Parameter("mu2", 0.4, 0, 0.5, step_size=0.001, floating=True)

sigma_1 = zfit.Parameter("sigma1", 0.1, 0.01, 0.4, floating=True)
sigma_2 = zfit.Parameter("sigma2", 0.1, 0.01, 0.4, floating=True)

frac = zfit.Parameter('frac', 0.3, 0, 1)

# model building, pdf creation
gauss1 = zfit.pdf.Gauss(mu=mu_1, sigma=sigma_1, obs=obs)
gauss2 = zfit.pdf.Gauss(mu=mu_2, sigma=sigma_2, obs=obs)

model = zfit.pdf.SumPDF([gauss1, gauss2], frac)

yield_ = zfit.Parameter('yield_model', 10000, 0, 1000000, step_size=10, floating=True)
model_ext = model.create_extended(yield_)

# data
data = zfit.Data.from_numpy(obs=obs, array=df.to_numpy())


def minimizing(model, data):

    nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
    minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
    return minimizer.minimize(nll)


def plot_figure(obs, model, PDF_1, PDF_2, data, yield_, x_axe:tuple=(0, 1.0), nbins:int=100):

    plt.figure()

    x = np.linspace(*x_axe, 1000)
    y_model = model.pdf(x).numpy()  # rerun now after the fitting
    y_PDF_1 = (PDF_1.pdf(x)*frac).numpy()
    y_PDF_2 = (PDF_2.pdf(x)*(1-frac)).numpy()
    data_np = data[:, 0].numpy()

    scaling = (yield_/nbins)*obs.area()

    # plot the data
    plt.hist(data_np, color='black', bins=nbins, histtype="stepfilled", alpha=0.1)

    plt.plot(x, y_model*scaling, label="Model")
    plt.plot(x, y_PDF_1*scaling, label="Gauss1")
    plt.plot(x, y_PDF_2*scaling, label="Gauss2")
    mplhep.histplot(np.histogram(data_np, bins=nbins), yerr=True, color='black', histtype='errorbar',
                markersize=5, capsize=2.5, markeredgewidth=1.5, zorder=1, elinewidth=1.5)
    plt.ylabel("Counts")
    plt.xlabel("Physical observable")
    plt.legend()

    plt.show()


def main():

    result = minimizing(model_ext, data)
    # result.hesse(method='minuit_hesse')   # error estimator of Minuit
    result.errors()                         # zfit internal error estimator

    plot_figure(obs=obs, model=model, PDF_1=gauss1, PDF_2=gauss2, data=data, yield_=result.params[yield_]['value'])

if __name__ == '__main__':
    main()
    pass







