from scipy.io import mmread
import os
import glob
import pandas as pd
import numpy as np
from pandas_ods_reader import read_ods
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
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering, leaves_list
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

import dimorph_processing as dp

def compute_sex_stats(meta_data_df):
    '''takes metadata and computes percentage of 'Breeder-F', 'Breeder-M', 'Naïve-F', 'Naïve-M' for each cluster as well as n_mice.
    returns results in dataframe'''
    #get lists of unique mice_id, groups, and markers
    mice_id = np.unique(meta_data_df.loc['ChipID'])
    groups = np.unique(meta_data_df.loc['Group'])
    m_list = np.unique(meta_data_df.loc['markers'])
    #build sex stats df
    sex_stats_df = pd.DataFrame(columns = ['markers','Breeder-F','Breeder-M' ,'Naïve-F' , 'Naïve-M','num_mice'])
    sex_stats_df['markers'] = np.unique(meta_data_df.loc['markers'])
    sex_stats_df = sex_stats_df.set_index('markers')
    #populate sex stats df
    for m in m_list:
        c_marker_subset = meta_data_df.loc[:,meta_data_df.loc['markers']==m]
        #total num cells in cluster
        num_cells = c_marker_subset.shape[1]
        for i,g in enumerate(groups):
            perc_g = round((c_marker_subset.loc[:,c_marker_subset.loc['Group'] == groups[i]].shape[1]/num_cells)*100,2)
            #print (g)
            #print (perc_g)
            sex_stats_df.loc[m,g] = perc_g

        n_mice = len(np.unique(c_marker_subset.loc['ChipID']))
        sex_stats_df.loc[m,'num_mice'] = n_mice
    
    return sex_stats_df