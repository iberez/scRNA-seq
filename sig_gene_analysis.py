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
from scipy.stats import mannwhitneyu, false_discovery_control, wilcoxon
import csv

import dimorph_processing as dp
import sex_stats as ss

def format_genes(genes_df, metadata, mg_cl_dict, output_folder, out_file_name, write_to_file = False):
    '''formats long form csv of outputted from volcano_plot() found in sex_stats.py into wide form, genes x test'''
    
    columns = [item for item in sorted(np.unique(metadata.loc['cluster_label'])) for _ in range(8)] 
    # Pivot the DataFrame to get delta and -log10(p_adj) in different columns for each gene
    genes_df_pivot_delta = genes_df.pivot_table(index='gene', columns=['id', 'test'], values=['delta'])
    # Pivot the DataFrame to get delta and -log10(p_adj) in different columns for each gene
    genes_df_pivot_p = genes_df.pivot_table(index='gene', columns=['id', 'test'], values=['p_adj'])
    # Interleave columns of A and B
    new_columns = [col for pair in zip(genes_df_pivot_delta.columns, genes_df_pivot_p.columns) for col in pair]
    genes_df_pivot_combined = pd.concat([genes_df_pivot_delta, genes_df_pivot_p], axis=1)
    genes_df_pivot_combined = genes_df_pivot_combined[new_columns]
    
    #add in cell types using mg_cl_dict
    genes_df_pivot_combined_w_ct = genes_df_pivot_combined.copy()
    ct = [mg_cl_dict[x] for x in genes_df_pivot_combined.columns.get_level_values(1)]
    ct_flat = ['-'.join(x) for x in ct]
    new_columns = pd.MultiIndex.from_arrays([ct_flat] + [genes_df_pivot_combined_w_ct.columns.get_level_values(i) for i in range(genes_df_pivot_combined_w_ct.columns.nlevels)])
    genes_df_pivot_combined_w_ct.columns = new_columns
    
    if write_to_file:
        genes_df_pivot_combined_w_ct.to_csv(output_folder + out_file_name + '.csv')

    return genes_df_pivot_combined_w_ct

def get_cts_per_gene(p_adj_test_df,delta_test_df,delta_fc = 1.5,p_adj = 0.05):
    '''takes long form sig_df for a single test, creates mask where sig criteria is true, and returns sorted dataframe of genes and cell types sorted by unique number of cell types'''
    #as a first pass, create boolean mask, true where value in p_adj matrix is <0.05 across all groups
    p_adj_test_df_mask = p_adj_test_df<p_adj

    tmp = np.where(p_adj_test_df_mask==True)

    # get test indices where gene changes index
    result = []
    current_group = []
    prev_value = tmp[0][0]
    
    # Iterate through both arrays, split second where first changes (new gene)
    for val1, val2 in zip(tmp[0],tmp[1]):
        if val1 != prev_value:  
            result.append(current_group)  
            current_group = []  
        current_group.append(val2)
        prev_value = val1
    
    # Append the last group
    result.append(current_group)
    print (result)
    #use ind_2_ct_dict to map column index results to cell type name
    ind_2_ct_dict = dict(zip(np.arange(0,len(p_adj_test_df.columns)),np.array(p_adj_test_df.columns.get_level_values(0))))
    #print (ind_2_ct_dict)
    cts_per_gene = []
    for r in result:
        l = []
        for ind in r:
            ct = ind_2_ct_dict[ind]
            l.append(ct)
        cts_per_gene.append(l)
    print (len(cts_per_gene))
    
    unique_cts_per_gene = []
    for cts in cts_per_gene:
        #unique_cts = []
        u = np.unique(cts)
        unique_cts_per_gene.append(u)

    unique_cts_per_gene_cnts = [len(x) for x in unique_cts_per_gene]

    cts_per_gene_df = pd.DataFrame(index=p_adj_test_df.index[:len(cts_per_gene)], columns= ['ct'])
    cts_per_gene_df['ct'] = cts_per_gene
    cts_per_gene_df['unique_cts'] = unique_cts_per_gene
    cts_per_gene_df['unique_cts_counts'] = unique_cts_per_gene_cnts
    cts_per_gene_df_uni_ct_sorted = cts_per_gene_df.sort_values(by = 'unique_cts_counts', ascending = False)

    return cts_per_gene_df_uni_ct_sorted

def plot_sig_gene_heatmap(delta_df,ind_2_title_dict,output_folder = None, outfile_name = None, savefig = False):
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=False)  # Adjust height to fit all heatmaps
    vmin = delta_df.min().min()
    vmax = delta_df.max().max()
    
    # Plot each heatmap on its respective subplot
    for i, ax in enumerate(axes):
        # Set the title for each subplot
        ax.set_title(ind_2_title_dict[i+1])
        
        # Plot the heatmap, selecting every 4th column starting at i for each subplot
        sns.heatmap(delta_df.iloc[:, i::4], ax=ax, vmin=vmin, vmax=vmax, cbar=True,  cbar_kws={'label': '<-' + str(ind_2_title_dict[i+1][2]) + ' ' + str(ind_2_title_dict[i+1][1]) +'->'})
        ax.set_xticklabels(delta_df.iloc[:, ::4].columns.get_level_values(2)) #use (2) for id, (0) for cell type
        ax.set_xlabel('')
    if savefig:
        plt.savefig(output_folder+outfile_name+'.png')
    # Show the entire figure with all 4 heatmaps
    plt.tight_layout()
    plt.show()

def plot_sig_gene_heatmap_igi(sig_deltas,
                              title,
                              subset = 20,
                              output_folder = None, 
                              outfile_name = None, 
                              savefig = False):
    '''same as plot_sig_gene_heatmap but uses stacked single, reindexed delta_df for independent group specific gene list index (igi) for each group'''
    fig, axes = plt.subplots(figsize=(15, 5), sharex=False)  # Adjust height to fit all heatmaps
    vmin = sig_deltas.min().min()
    vmax = sig_deltas.max().max()
    sns.heatmap(sig_deltas[:subset], 
                vmin=vmin, 
                vmax=vmax,
                cmap='seismic', 
                cbar=True,  
                cbar_kws={'label': '<-' + str(title[2]) + ' ' + str(title[1]) +'->'})
    #axes.collections[0].colorbar.set_label(str(title[2]) + '      ' + str(title[1]))
    axes.set_xticklabels(sig_deltas.columns.get_level_values(2)) #use (2) for id, (0) for cell type
    axes.set_xlabel('')
    axes.set_title(str(title))

    '''
    # Plot each heatmap on its respective subplot
    for i, ax in enumerate(axes):

        vmin = delta_df.min().min()
        vmax = delta_df.max().max()
        # Set the title for each subplot
        ax.set_title(ind_2_title_dict[i+1])
        
        # Plot the heatmap, selecting every 4th column starting at i for each subplot
        sns.heatmap(delta_df.iloc[:, i::4], ax=ax, vmin=vmin, vmax=vmax, cbar=True)
        ax.set_xticklabels(delta_df.columns.get_level_values(2)) #use (2) for id, (0) for cell type
        ax.set_xlabel('')
    '''
    if savefig:
        plt.savefig(output_folder+outfile_name+'.png')
    # Show the entire figure with all 4 heatmaps
    plt.tight_layout()
    plt.show()

def p_mask_max_abs(p_adj_test_df,delta_test_df,delta_fc = 1.5,p_adj = 0.05):
    '''create a boolean mask using p_adj df, use mask to filter FCs (p_delta df), setting all other values to zero.
    take max of abs value of filtered p_delta df and sort gene index high to low'''
    mask = p_adj_test_df < 0.05
    filtered_sig_deltas = delta_test_df.where(mask.values, 0)
    filtered_sig_deltas_abs = np.abs(filtered_sig_deltas)
    max_fc = filtered_sig_deltas_abs.max(axis = 1)
    max_fc_sorted = max_fc.sort_values(ascending = False)
    return max_fc_sorted
    