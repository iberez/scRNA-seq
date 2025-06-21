# Isaac Berez
# 17.01.23
import sys
from scipy.io import mmread
import os
import glob
import pandas as pd
import numpy as np
#from pandas_ods_reader import read_ods
from copy import deepcopy
import pprint
import json
import re
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import harmonypy as hm
from matplotlib.cm import ScalarMappable
from datetime import date
import mpld3
import hvplot.pandas
import holoviews as hv
from holoviews import opts
import panel as pn
import bokeh
from bokeh.resources import INLINE
from adjustText import adjust_text
from scipy.stats import mannwhitneyu, false_discovery_control, wilcoxon
import matplotlib as mpl
import scipy.stats as st

import dimorph_processing as dp
import cell_comparison as cc
import sex_stats as ss
import run_sex_stats as rss

today = str(date.today())

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def isolate_expression(ct_name,df_norm, meta_data_df):
    ct_df = df_norm.iloc[:, (np.where(meta_data_df.loc['full_name'] == ct_name)[0])]
    ct_md_df = meta_data_df.iloc[:, (np.where(meta_data_df.loc['full_name'] == ct_name)[0])]
    #groups = pd.unique(meta_data_df.loc['Group'])
    groups = np.array(['Naïve-M', 'Naïve-F', 'Breeder-M', 'Breeder-F'], dtype=object)
    n_m_expr = ct_df.iloc[:,np.where(ct_md_df.loc['Group'] == groups[0])[0]]
    n_f_expr = ct_df.iloc[:,np.where(ct_md_df.loc['Group'] == groups[1])[0]]
    b_m_expr = ct_df.iloc[:,np.where(ct_md_df.loc['Group'] == groups[2])[0]]
    b_f_expr = ct_df.iloc[:,np.where(ct_md_df.loc['Group'] == groups[3])[0]]
    return n_m_expr,n_f_expr,b_m_expr,b_f_expr

def plot_violin(gene,ax,ct_name,n_m_expr,n_f_expr,b_m_expr,b_f_expr, groups, outfolder, savefig = False):

    #fig,ax = plt.subplots(figsize=(3,3)) #un comment if using for single ct
    v = ax.violinplot([n_m_expr.loc[gene],n_f_expr.loc[gene],b_m_expr.loc[gene],b_f_expr.loc[gene]],showmeans=True, showextrema=False)

    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(groups, rotation = 45)
    
    #if ax == axes[0]:
        #ax.set_ylabel(gene + ' Expr')

    v['cmeans'].set_color('black')
    #sns.stripplot(data=[b_m_expr.loc[gene],b_f_expr.loc[gene],n_f_expr.loc[gene],n_m_expr.loc[gene]],x = groups,jitter=jitter, alpha=0.5)

    for pc in v["bodies"]:
        pc.set_facecolor("none")
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
        pc.set_alpha(1)

    colors = ['orange','blue','green','red']
    # Plot scatter points for each group with jitter
    for i, group_data in enumerate([n_m_expr.loc[gene],n_f_expr.loc[gene],b_m_expr.loc[gene],b_f_expr.loc[gene]]):
        # Generate random jitter between -0.1 and 0.1
        jitter = np.random.uniform(-0.1, 0.1, size=len(group_data))
        # Compute x-positions as group index (0, 1, 2) plus jitter
        x_positions = i + 1 + jitter
        #print (x_positions)
        # Plot points with matching color, transparency, and small size
        ax.scatter(x_positions, group_data, color = colors[i],alpha=0.3, s=10, edgecolors='none')
    

    ax.set_title(ct_name)

    # Remove top and right spines (the "box")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if savefig:
        savefolder = outfolder + gene + '/'
        os.makedirs(savefolder, exist_ok=True)
        print (savefolder)
        plt.savefig(savefolder + gene + '_' + ct_name + '.pdf',bbox_inches="tight")
    #comment out plt.show() when calling in a loop, uncomment when using for single ct 
    #plt.show()
    return None