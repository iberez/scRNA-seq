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

import dimorph_processing as dp
import cell_comparison as cc
import sex_stats as ss
import sig_gene_analysis as sga

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


##run sex gene analysis

def run_sig_genes(all_genes_folder, sex_stats_folder, sig_gene_heatmap_folder, ct_bar_plots_folder, cell_class, log_fc = np.log2(1.5),alpha = 0.05, savefig = False, write_to_file = False):
    '''reads in long form sig genes csv, metadata_df_dlr and formats into wide form data useful for heatmaps. also determines 
    most dimorphic cell types.'''
    all_genes_df = pd.read_csv(all_genes_folder + 'all_genes.csv',header=None)
    all_genes_df.rename(columns = {0:'cluster_fn',1:'test',2:'gene',3:'delta',4:'p_adj'}, 
                inplace = True)
    metadata_df_dlr = pd.read_json(sex_stats_folder + cell_class + '_metadata_df_dlr.json')
    _, idx = np.unique(metadata_df_dlr.loc['full_name'], return_index=True)
    fn = np.array(metadata_df_dlr.loc['full_name'][np.sort(idx)])
    #format all stat results into wide format
    all_genes_df_formatted = sga.format_genes(all_genes_df,metadata_df_dlr, fn,all_genes_folder, 'all_genes_formatted',write_to_file=write_to_file)
    #filter long form data for sig genes 
    sig_genes = all_genes_df.iloc[np.where((np.abs(all_genes_df.loc[:,'delta'])>log_fc) & (all_genes_df['p_adj'] < alpha))]
    #use the unique list of genes in sig_genes to filter index of wide fomrmat dated
    index = list(np.unique(sig_genes['gene']))
    sig_full = all_genes_df_formatted.loc[index]
    #isolate just delta and p_adj values into their own dataframes
    sig_deltas = sig_full.iloc[:,::2]
    sig_p_adj = sig_full.iloc[:,1::2]
    #isolate delta and p values for each test
    sig_deltas_ΔBN_m = sig_deltas.iloc[:,::4]
    sig_p_adj_ΔBN_m = sig_p_adj.iloc[:,::4]
    sig_deltas_ΔBN_f = sig_deltas.iloc[:,1::4]
    sig_p_adj_ΔBN_f = sig_p_adj.iloc[:,1::4]
    sig_deltas_Δmf_B = sig_deltas.iloc[:,2::4]
    sig_p_adj_Δmf_B = sig_p_adj.iloc[:,2::4]
    sig_deltas_Δmf_N = sig_deltas.iloc[:,3::4]
    sig_p_adj_Δmf_N = sig_p_adj.iloc[:,3::4]
    #ranking most dimorphic cell types by counting number of 
    #stat significant genes in each cell type
    cluster_fn_list = []
    num_genes_per_ct = []
    ΔBN_m = []
    ΔBN_f = []
    Δmf_B = []
    Δmf_N = []
    for x in np.unique(sig_genes['cluster_fn']):
        subset = sig_genes.iloc[np.where(sig_genes['cluster_fn']==x)[0]]
        g = subset.shape[0]
        #print (subset.shape)
        #print (subset.loc[subset.loc[:,'test']=='ΔBN_m',:].shape)
        ΔBN_m.append(subset.loc[subset.loc[:,'test']=='ΔBN_m',:].shape[0])
        ΔBN_f.append(subset.loc[subset.loc[:,'test']=='ΔBN_f',:].shape[0])
        Δmf_B.append(subset.loc[subset.loc[:,'test']=='Δmf_B',:].shape[0])
        Δmf_N.append(subset.loc[subset.loc[:,'test']=='Δmf_N',:].shape[0])
        num_genes_per_ct.append(g)
        cluster_fn_list.append(x)

    dimorph_cell_types_df = pd.DataFrame()
    dimorph_cell_types_df['cluster_fn'] = cluster_fn_list
    dimorph_cell_types_df['gene_count'] = num_genes_per_ct
    dimorph_cell_types_df['ΔBN_m'] = ΔBN_m
    dimorph_cell_types_df['ΔBN_f'] = ΔBN_f 
    dimorph_cell_types_df['Δmf_B'] = Δmf_B 
    dimorph_cell_types_df['Δmf_N'] = Δmf_N 
    dimorph_cell_types_df = dimorph_cell_types_df.set_index('cluster_fn')
    #reverse order so plot shows most dimorphic cell types on top
    dimorph_cell_types_df_sorted = dimorph_cell_types_df.reindex(index=np.flip(fn))
    dimorph_cell_types_df_sorted.insert(5,'ct_size', None)
    ct_size_vector = []
    for ct in dimorph_cell_types_df_sorted.index:
        ct_size = metadata_df_dlr.loc[:,metadata_df_dlr.loc['full_name',:]==ct].shape[1]
        dimorph_cell_types_df_sorted.loc[ct,'ct_size'] = ct_size
        #print (ct, ct_size)
    #ax = dimorph_cell_types_df_sorted.sort_values(by='gene_count').iloc[:,1:5].dropna().plot.barh(stacked=True)
    #ax.set_xlabel('gene_counts')
    #plt.tight_layout()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    # Plot the stacked bar plot
    dimorph_cell_types_df_sorted.sort_values(by='gene_count').iloc[:, 1:5].dropna().plot(kind='barh', stacked=True, ax=axes[0],colormap='tab20b')
    axes[0].set_ylabel("cluster_fn")
    axes[0].set_xlabel("sig gene count")
    axes[0].legend(title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot the ct_size bar plot
    dimorph_cell_types_df_sorted.sort_values(by='gene_count').dropna()['ct_size'].plot(kind='barh', color='gray', ax=axes[1])
    axes[1].set_title("Size")
    axes[1].set_xlabel("num cells")

    # Adjust layout
    plt.tight_layout()
    plt.show()
    if savefig:
        plt.savefig(ct_bar_plots_folder + cell_class + '_ranked_cts_all_groups_w_num_cells.pdf', transparent = True)
    plt.show()

    #create a horizonatly stacked multiplot, using order of stacked
    colors = mpl.colormaps['tab20b'].colors
    c_mf_N = colors[19]
    c_mf_B = colors[13]
    c_BN_f = colors[6]
    c_BN_m = colors[0]


    fig,axs = plt.subplots(1,5, sharey=True, figsize = (10,8))
    axs[0].barh(y = dimorph_cell_types_df_sorted.index, width=dimorph_cell_types_df_sorted.loc[:,'ΔBN_m'], color = c_BN_m)
    axs[1].barh(y = dimorph_cell_types_df_sorted.index, width=dimorph_cell_types_df_sorted.loc[:,'ΔBN_f'], color = c_BN_f)
    axs[2].barh(y = dimorph_cell_types_df_sorted.index, width=dimorph_cell_types_df_sorted.loc[:,'Δmf_B'], color = c_mf_B)
    axs[3].barh(y = dimorph_cell_types_df_sorted.index, width=dimorph_cell_types_df_sorted.loc[:,'Δmf_N'], color = c_mf_N)
    axs[4].barh(y = dimorph_cell_types_df_sorted.index, width = dimorph_cell_types_df_sorted.loc[:,'ct_size'], color = 'gray')
    axs[0].set_title('ΔBN_m')
    axs[0].set_xlabel('no. significant genes')
    axs[1].set_title('ΔBN_f')
    axs[1].set_xlabel('no. significant genes')
    axs[2].set_title('Δmf_B')
    axs[2].set_xlabel('no. significant genes')
    axs[3].set_title('Δmf_N')
    axs[3].set_xlabel('no. significant genes')
    axs[4].set_title('num_cells')
    axs[4].set_xlabel('no. cells')

    plt.tight_layout()
    # Apply the same ticks and labels to the last two plots
    #for ax in axes[2:]:
        #ax.set_xticks(tick_positions)
        #ax.set_xticklabels(tick_labels)
    if savefig:
        plt.savefig(ct_bar_plots_folder + cell_class + '_ranked_cts_num_cells_horizontal.pdf')
    plt.show()
    if write_to_file:
        dimorph_cell_types_df_sorted.to_feather(ct_bar_plots_folder + cell_class + '_dimorph_cell_types_df_sorted.feather')
    return sig_deltas, sig_p_adj, sig_genes, dimorph_cell_types_df_sorted