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


import dimorph_processing as dp
import cell_comparison as cc
import sex_stats as ss



##run sex stats


def run_sex_stats(df,metadata_df,cell_class,run_folder,run_subfolders,delta_folder):
    '''wraps functions in sex stats, run after determining dlr df/metadata'''
    today = str(date.today())
    # Create the parent folder if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)
    run_subfolders = ['gene_delta_plots', 'volcano_plots','ct_bar_plots','sig_gene_heatmaps']

    # Create each subfolder
    for subfolder in run_subfolders:
        path = os.path.join(run_folder, subfolder)
        os.makedirs(path, exist_ok=True)  # Use exist_ok=True to avoid errors if the folder already exists

    print("run folder and subfolders created successfully!")


    # Create the parent folder if it doesn't exist
    os.makedirs(delta_folder, exist_ok=True)
    delta_subfolders = ['data', 'plots', 'sig_plots', 'utest_data']
    # Create each subfolder
    for subfolder in delta_subfolders:
        path = os.path.join(delta_folder, subfolder)
        os.makedirs(path, exist_ok=True)  # Use exist_ok=True to avoid errors if the folder already exists

    print("delta folder and subfolders created successfully!")

    delta_data_folder = delta_folder + 'data/' 
    utest_output_folder  = delta_folder + 'utest_data/'
    volcano_output_folder = run_folder + 'volcano_plots/'

    volcano_subfolders = ['data','plots']
    # Create each subfolder
    for subfolder in volcano_subfolders:
        path = os.path.join(volcano_output_folder, subfolder)
        os.makedirs(path, exist_ok=True)  # Use exist_ok=True to avoid errors if the folder already exists

    print("volcano folder and subfolders created successfully!")


    #get full names of clusters, preservering order
    _, idx = np.unique(metadata_df.loc['full_name'], return_index=True)
    fn = np.array(metadata_df.loc['full_name'][np.sort(idx)])
    fn

    #run gene expression differences, generate and save delta plots
    all_counts_df = pd.DataFrame(columns=['N_f_cnts', 'B_f_cnts','N_m_cnts','B_m_cnts'])
    for c in fn:
        _,counts_df = ss.compute_group_gene_expression_differences(df,
                                                                metadata_df,
                                                                cluster_fn=c,
                                                                threshold_prc_h=70,
                                                                threshold_prc_l=10, 
                                                                r_bn = 2, 
                                                                r_mf = 2, 
                                                                cell_class=cell_class, 
                                                                folder = delta_folder,
                                                                normalize = True,
                                                                n_factor = 20000,          
                                                                mode = 'delta',
                                                                sig_genes_df = None,
                                                                savefig = True, 
                                                                write_to_file=True)
        print ('Finished processing cluster: ', c)
        all_counts_df = pd.concat([all_counts_df,counts_df])
    print ('delta analysis complete, results in: ', delta_folder)

    #run stat tests to get q values
    for c in fn:
        print ('running mann whitney test on cluster ', c, '...')
        _, _, _, _ = ss.run_stat_test(delta_data_folder,c,utest_output_folder,cell_class,write_to_file=True)
    print('stat tests complete. results in: ', utest_output_folder)

    #run volcano analysis
    for c in fn:
        print('running volcano analysis on cluster ', c, '...')
        _, _, _, _ = ss.run_volcano_analysis(delta_data_folder, utest_output_folder,volcano_output_folder, c, cell_class, all_counts_df, savefig = True, write_to_file = True)
    print('volcano analysis complete. results in: ', volcano_output_folder)

    #save df and metadata df
    metadata_df.to_json(run_folder + cell_class + '_metadata_df_dlr.json')
    df.to_feather(run_folder + cell_class + '_df_dlr.feather')

    return None