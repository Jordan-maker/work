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



obs = zfit.Space('obs1', (-5, 10))

size_normal = 10000
data_normal_np = np.random.normal(size=size_normal, scale=2)

data_normal = zfit.Data.from_numpy(obs=obs, array=data_normal_np)


mu = zfit.Parameter('mu', 1, -3, 3, step_size=0.2)
sigma_num = zfit.Parameter('sigma42', 1, 0.1, 10, floating=False)

gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma_num)

integral = gauss.integrate(limits=(-1, 3))


def plot_model(model, data, scale=1, plot_data=True):  # we will use scale later on

    nbins = 50

    lower, upper = data.data_range.limit1d
    x = tf.linspace(lower, upper, num=1000)  # np.linspace also works
    y = model.pdf(x) * size_normal / nbins * data.data_range.area()
    y *= scale
    plt.plot(x, y)
    data_plot = zfit.run(z.unstack_x(data))  # we could also use the `to_pandas` method
    if plot_data:
        plt.hist(data_plot, bins=nbins)
        plt.show()


#plot_model(gauss, data_normal)

####################################################################################################

mass_obs = zfit.Space('mass', (0, 1000))

# Signal component

mu_sig = zfit.Parameter('mu_sig', 400, 100, 600)
sigma_sig = zfit.Parameter('sigma_sig', 50, 1, 100)
alpha_sig = zfit.Parameter('alpha_sig', 200, 100, 400, floating=False)  # won't be used in the fit
n_sig = zfit.Parameter('n sig', 4, 0.1, 30, floating=False)
signal = zfit.pdf.CrystalBall(obs=mass_obs, mu=mu_sig, sigma=sigma_sig, alpha=alpha_sig, n=n_sig)

# combinatorial background

lam = zfit.Parameter('lambda', -0.01, -0.05, -0.001)
comb_bkg = zfit.pdf.Exponential(obs=mass_obs, lam=lam)

part_reco_data = np.random.normal(loc=200, scale=150, size=700)
part_reco_data = zfit.Data.from_numpy(obs=mass_obs, array=part_reco_data)  # we don't need to do this but now we're sure it's inside the limits

part_reco = zfit.pdf.GaussianKDE1DimV1(obs=mass_obs, data=part_reco_data, bandwidth='adaptive')

# Composing models

sig_frac = zfit.Parameter('sig_frac', 0.3, 0, 1)
comb_bkg_frac = zfit.Parameter('comb_bkg_frac', 0.25, 0, 1)
model = zfit.pdf.SumPDF([signal, comb_bkg, part_reco], [sig_frac, comb_bkg_frac])


with zfit.param.set_values([mu_sig, sigma_sig, sig_frac, comb_bkg_frac, lam], [370, 34, 0.18, 0.15, -0.006]):
    data = model.sample(n=10000)

#plot_model(model, data)


def plot_comp_model(model, data):
    for mod, frac in zip(model.pdfs, model.params.values()):
        plot_model(mod, data, scale=frac, plot_data=False)
    plot_model(model, data)

#plot_comp_model(model, data)

yield_model = zfit.Parameter('yield_model', 10000, 0, 20000, step_size=10)
model_ext = model.create_extended(yield_model)


# Loss function

nll = zfit.loss.ExtendedUnbinnedNLL(model_ext, data)
minimizer = zfit.minimize.Minuit(use_minuit_grad=True)



values = z.unstack_x(data)
obs_right_tail = zfit.Space('mass', (700, 1000))
data_tail = zfit.Data.from_tensor(obs=obs_right_tail, tensor=values)
with comb_bkg.set_norm_range(obs_right_tail):
    nll_tail = zfit.loss.UnbinnedNLL(comb_bkg, data_tail)
    minimizer.minimize(nll_tail)


lam.floating = False

result = minimizer.minimize(nll)

plot_comp_model(model_ext, data)



