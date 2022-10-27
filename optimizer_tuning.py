import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from tqdm import tqdm

from bayes_opt import BayesianOptimization, UtilityFunction
import warnings
warnings.filterwarnings("ignore")

class Venus:
    # note: always set the func paramter
    def __init__(
        self,
        inj_limits=[97, 110],
        mid_limits=[97, 110],
        ext_limits=[116, 128],
        beam_range=[0.50, 1.00],
        jitter=0, 
        func = (lambda X: -1)
    ):
        """The limits on the magnetic solenoids currents and the beam range (ouput).
        A random jitter can be added also (fraction of 1.)."""
        self.inj_limits = inj_limits
        self.mid_limits = mid_limits
        self.ext_limits = ext_limits
        self.beam_range = beam_range
        self.currents = np.zeros(3)
        self.jitter = jitter
        self.rng = np.random.default_rng(42)
        self.func = func

    def set_mag_currents(self, inj, mid, ext):
        """Set the magnetic currents on the coils."""
        for v, lim in zip([inj, mid, ext], [self.inj_limits, self.mid_limits, self.ext_limits]):
            if v < lim[0] or v > lim[1]:
                raise ValueError("Setting outside limits")
        self.currents = np.array([inj, mid, ext])

    def _rescale_inputs(self, inputs):
        """input to himmelblau4 must be in [-6, 6]."""
        return (
            (c - l[0]) * 12.0 / (l[1] - l[0]) - 6.0
            for c, l in zip(inputs, [self.inj_limits, self.mid_limits, self.ext_limits])
        )

    def _rescale_output(self, output):
        """simple square returns values betwen 0 and 27 for w, x, y, z in [-6, 6]."""
        return (
            (1. - (output / 27.0) + self.rng.normal(0.0, self.jitter)) *
            (self.beam_range[1] - self.beam_range[0]) + self.beam_range[0]
        )
    
    def get_noise_level(self):
        # return std of the noise
        noise = self.jitter*(self.beam_range[1] - self.beam_range[0])
        return noise

    def get_beam_current(self):
        """Read the current value of the beam current"""
        return self.func(self.currents)

    @staticmethod
    def _simple_square(w, x, y):
        """A not so funky 3 dimensional parameter space with a single minima."""
        return (
            (w - 3.)**2 + (x - 3.)**2 + (y - 3.)**2
        )

# plotting
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']=100
RANDSTATE = 42

sns.set(rc = {'figure.figsize':(13,8)})

def load_model(target):
    gpr_model, avg_list, std_list = pickle.load(open(target,"rb"))
    normalizer = lambda X: (X - avg_list )/ std_list
    func = lambda currents: gpr_model.predict(normalizer(currents.reshape(1, -1)))[0]
    return func
    
func1 = load_model("Models/gpr_exp1.dump") 

pbounds = {"A": [97, 110], "B": [97, 110], "C": [116, 128]}
venus = Venus(jitter=0.15, func=func1)
def bbf(A, B, C):
    venus.set_mag_currents(A, B, C)
    v = venus.get_beam_current()
    return v

random_state = 0

# for each kappa, alpha combo, try n times
n = 10
kappa_list = []
alpha_list = []
random_state += 1
optimizer = BayesianOptimization(f = bbf,
                                     pbounds = pbounds, verbose = 0,
                                     random_state = random_state)
optimizer.maximize(init_points = 5, n_iter = 30, kappa=kappa, alpha=0.15)
best = optimizer.max["target"]