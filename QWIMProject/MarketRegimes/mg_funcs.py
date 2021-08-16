#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.mixture as mix
import datetime as dt
import scipy.stats as scs


import warnings; warnings.simplefilter('ignore')
import requests, zipfile, io
from urllib.request import urlopen
import pandas as pd

import matplotlib as mp
from matplotlib import cm
from matplotlib.dates import YearLocator, MonthLocator

import missingno as msno
from tqdm import tqdm
import matplotlib.font_manager
import seaborn as sns


#%%
class Mkt_regime:
    def __init__(self,dfa):
        ticker_ = dfa.columns[0]
        dfb = np.log(dfa[ticker_]/dfa[ticker_].shift(1)).dropna()
        dfa = pd.concat([dfa, dfb], axis = 1)
        dfa.columns = [ticker_, 'sret']
        #plt.plot(dfa[ticker])
        #print(dfa.head())
        self.X = dfa.dropna()
        self.ticker = ticker_
        self.states_df = pd.DataFrame()
        self.n_component_ = 0
        self.hid_states_ = None

    def hidden_states(self,n_=3):
        model = mix.GaussianMixture(n_components = n_,
                                covariance_type ="full",
                            n_init = 100,
                            random_state=7).fit(self.X)
                            
        # Predict the optimal sequence of internal hidden state
        hidden_states_ = model.predict(self.X)
        print("Means and vars of each hidden state")
        for i in range(model.n_components):
            print("{}th hidden state".format(i))
            print("mean = ", model.means_[i]) # list len 2
            print("var = ", np.diag(model.covariances_[i])) # list len 2
            print()

        states_ = (pd.DataFrame(hidden_states_, columns=['states'], index=self.X.index)
                .join(self.X, how="inner")
                .assign(mkt_cret=self.X.sret.cumsum())
                .reset_index(drop=False)
                .rename(columns={'index':'Date','{}'.format(self.ticker):'Lst_price'}))
        states_['Ticker'] = [self.ticker]*len(self.X)
        print(states_.head())
        self.states_df = states_
        self.n_component_ = n_
        self.hid_states_ = hidden_states_
        return self.states_df

    def hidden_states_plot(self,n_=3):
        if self.n_component_!=n_:
            States_df = self.hidden_states(n_)
            hidden_states_ = self.hid_states_
        else:
            hidden_states_ = self.hid_states_
        col = 'sret'
        sns.set(font_scale=1.25)
        style_kwds = {'xtick.major.size':3, 'ytick.major.size':3,
                    'font.family':u'courier prime code', 'legend.frameon':True}
        sns.set_style('white', style_kwds)

        fig, axs = plt.subplots(n_, sharex=True, sharey=True, figsize=(12,9))
        colors = cm.rainbow(np.linspace(0, 1, n_))

        for i, (ax, color) in enumerate(zip(axs, colors)):
            # Use fancy indexing to plot data in each state.
            mask = hidden_states_ == i
            ax.plot_date(self.X.index.values[mask],
                        self.X[col].values[mask],
                        ".-", c=color)
            ax.set_title("{}th hidden state".format(i), fontsize = 16, fontweight="demi")
            
        #plt.gcf().autofmt_xdate()
        plt.tight_layout()
        fig.savefig('Hidden Markov (Mixture) {} Model_Regime Subplots.png'.format(self.ticker),dpi=400)
        return 0

    def histor_regime(self,mkt,n_=3):
        if self.states_df.empty:
            states = self.hidden_states(n_)
        else:
            states = self.states_df

        style_kwds = {'xtick.major.size':3, 'ytick.major.size':3,
                        'font.family':u'courier prime code', 'legend.frameon':True}
        sns.set(font_scale=1.5)
        sns.set_style('white', style_kwds)
        order = list(range(self.n_component_))
        fg = sns.FacetGrid(data=states, hue = 'states', hue_order=order, aspect=1.31, size=12)
        fg.map(plt.scatter, 'Date', mkt, alpha=0.8).add_legend()
        sns.despine(offset=10)
        fg.fig.suptitle('Historical '+self.ticker+' Regimes', fontsize=24, fontweight='demi')
        fg.savefig('Historical {} Regimes.png'.format(self.ticker),dpi=400)
        return 0

def mul_regime_plot(df_lst,type):
    dftotal = pd.DataFrame()
    for df in df_lst:
        dftotal = dftotal.append(df)
    n_component = len(dftotal.states.unique())
    style_kwds = {'xtick.major.size':3, 'ytick.major.size':3,
                        'font.family':u'courier prime code', 'legend.frameon':True}
    sns.set(font_scale=1.5)
    sns.set_style('white', style_kwds)
    order = list(range(n_component))
    fg = sns.FacetGrid(data=dftotal, col = 'Ticker', hue = 'states', hue_order=order, aspect=1.31, size=12)
    fg.map(plt.scatter, 'Date', 'Lst_price', alpha=0.8).add_legend()
    sns.despine(offset=10)
    #fg.fig.suptitle('Historical '+type+' Regimes', fontsize=24, fontweight='demi')
    fg.savefig('Historical {} Regimes'.format(type),dpi=400)
    return 0