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

def compute_cluster_sex_stats(meta_data_df):
    '''takes metadata and computes percentage of 'Breeder-F', 'Breeder-M', 'Naïve-F', 'Naïve-M' for each cluster as well as n_mice.
    returns results in dataframe'''
    #get lists of unique mice_id, groups, and markers
    mice_id = np.unique(meta_data_df.loc['ChipID'])
    sample_id = np.unique(meta_data_df.loc['SampleID'])
    groups = np.unique(meta_data_df.loc['Group'])
    m_list = np.unique(meta_data_df.loc['markers'])
    #build sex stats df
    sex_stats_df = pd.DataFrame(columns = ['markers','Breeder-F','Breeder-M' ,'Naïve-F' , 'Naïve-M','num_sample_ids','sample_ids'])
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

        n_sample_ids = len(np.unique(c_marker_subset.loc['SampleID']))
        sex_stats_df.loc[m,'num_sample_ids'] = n_sample_ids
        s_ids = list(np.unique(c_marker_subset.loc['SampleID']))
        #print (str(m_ids))
        sex_stats_df.loc[m,'sample_ids'] = str(s_ids)
    
    return sex_stats_df

def compute_class_sex_stats(meta_data_df):
    group_counter = Counter(np.array(meta_data_df.loc['Group',:]))
    total_grp_cnts = 0
    for k,v in group_counter.items():
        total_grp_cnts+=v
    sex_stats_cls = pd.DataFrame(columns=['prc'])
    for k,v in group_counter.items():
        perc_g = round(v/total_grp_cnts*100,2)
        sex_stats_cls.loc[k] = perc_g
    grp_n_sampleid_dict = dict.fromkeys(group_counter.keys())
    grp_sample_ids_dict = dict.fromkeys(group_counter.keys())
    #grp_n_sampleid_dict
    for k in grp_n_sampleid_dict.keys():
        grp_n_sampleid_dict[k] = len(np.unique(meta_data_df.loc['SampleID',meta_data_df.loc['Group',:]==k]))
        grp_sample_ids_dict[k] = list(np.unique(meta_data_df.loc['SampleID',meta_data_df.loc['Group',:]==k]))
    n_sample_id_df = pd.DataFrame.from_dict(grp_n_sampleid_dict, orient='index', columns= ['num_sample_ids'])
    #print (n_sample_id_df)
    sample_ids_df = pd.DataFrame(list(grp_sample_ids_dict.items()), columns=['Category', 'sample_ids'])
    sample_ids_df = sample_ids_df.set_index('Category')
    #print (sample_ids_df)
    tmp_df = pd.concat([n_sample_id_df,sample_ids_df], axis=1)
    #print (tmp_df)
    sex_stats_cls_updated = pd.concat([sex_stats_cls,tmp_df],axis=1)
    
    return sex_stats_cls_updated

def get_optimal_ax_lim(x_ser,y_ser):
    '''gets optimal ax lim (used for scatter plots in compute_group_gene_expression_differences())'''
    min_x = np.min(x_ser)
    max_x = np.max(x_ser)
    min_y = np.min(y_ser)
    max_y = np.max(y_ser)
    opt_lim = round(np.max(np.abs([min_x,max_x,min_y,max_y])))
    return opt_lim

def compute_group_gene_expression_differences(df, meta_data_df,cluster_label,threshold_prc, r_bn,r_mf,cell_class,folder, savefig = False, write_to_file = False):
    '''Inputs
    Parameters
    ----------
    df: dataframe
        raw (non log/std) expression matrix of cell_class 
    meta_data_df: dataframe
        corresponding metadata
    cluster_label: int
        cluster label/id to extract from df
    threshold_prc: int
        threshold percentage of genes to keep
    r_bn: int
        no labels within specied radius in breeder naive plot
    r_mf: int
        no labels within specied radius in male femalie plot
    cell_class: str
        e.g. 'GABA'
    savefig: bool
        if true save pair of gene difference scatter plots
    Returns
    -------
    expr_mlog_df: dataframe 
        log and meaned expression for each group, genes as rows'''
    #marker gene expression within a cluster
    c_expr = df.loc[:,meta_data_df.loc['cluster_label']==cluster_label]
    c_expr_bool = c_expr.mask(c_expr>0, other = 1)

    gene_sum =  np.array(c_expr_bool.sum(axis=1))
    gene_sum = np.reshape(gene_sum,(c_expr_bool.shape[0],1))
    gene_threshold = (threshold_prc/100)*c_expr_bool.shape[1]
    genes_to_keep_ind = []
    for i,v in enumerate(gene_sum):
        if v > gene_threshold:
            genes_to_keep_ind.append(i)
    #update c_expr keeping only genes above threshold
    c_expr = c_expr.iloc[genes_to_keep_ind,:]
    c_metadata = meta_data_df.loc[:,meta_data_df.loc['cluster_label']==cluster_label]
    #isolated expression of 4 groups within cluster
    N_f_expr = c_expr.loc[:,c_metadata.loc['Group',:]=='Naïve-F']
    B_f_expr = c_expr.loc[:,c_metadata.loc['Group',:]=='Breeder-F']
    N_m_expr = c_expr.loc[:,c_metadata.loc['Group',:]=='Naïve-M']
    B_m_expr = c_expr.loc[:,c_metadata.loc['Group',:]=='Breeder-M']
    #take mean of log2+1 for each gene
    N_f_expr_mlog = np.mean(np.log2(N_f_expr+1),axis = 1)
    B_f_expr_mlog = np.mean(np.log2(B_f_expr+1),axis = 1)
    N_m_expr_mlog = np.mean(np.log2(N_m_expr+1),axis = 1)
    B_m_expr_mlog = np.mean(np.log2(B_m_expr+1),axis = 1)
    expr_mlog_df = pd.concat([N_f_expr_mlog,B_f_expr_mlog,N_m_expr_mlog,B_m_expr_mlog], axis = 1)
    expr_mlog_df = expr_mlog_df.rename(columns={0: "N_f", 1: "B_f",2:"N_m", 3:"B_m" })
    
    #get delta Breeder - Naive, within each sex
    delta_B_N_m = expr_mlog_df['B_m'] - expr_mlog_df['N_m']
    delta_B_N_f = expr_mlog_df['B_f'] - expr_mlog_df['N_f']
    d_bn = np.sqrt(delta_B_N_m**2 + delta_B_N_f**2)
    
    fig,ax = plt.subplots()
    ax.set_title(cell_class + ' Delta Breeder-Naive, Cluster: ' + str(cluster_label))
    ax.scatter(delta_B_N_m,delta_B_N_f, s=1)
    ax.set_xlabel('delta_B_N_m')
    ax.set_ylabel('delta_B_N_f')
    plt.axvline(color = 'grey')
    plt.axhline(y=0, color = 'grey')
    plt.axline((0, 0), slope=1, color="grey", linestyle='--')
    # Draw a circle with the specified radius
    circle = plt.Circle((0, 0), r_bn, color='grey', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    opt_lim = get_optimal_ax_lim(delta_B_N_m,delta_B_N_f)

    ax.set_xlim([-opt_lim,opt_lim])
    ax.set_ylim([-opt_lim,opt_lim])
    
    ax.text(opt_lim-(0.6*opt_lim), opt_lim-(0.1*opt_lim), 'thresh = '+str(threshold_prc) + '%')
    ax.text(opt_lim-(0.6*opt_lim), opt_lim-(0.2*opt_lim), 'r_bn = '+str(r_bn))
    
    TEXTS = []
    for i, txt in enumerate(list(delta_B_N_m.index)):
        if d_bn.iloc[i] > r_bn:
            #standard labeling
            ax.annotate(txt, (delta_B_N_m.iloc[i], delta_B_N_f.iloc[i]),fontsize = 5)
            #if txt.startswith('S'):
            #labeling using adjust text to repelling algo
            #TEXTS.append(ax.text(delta_B_N_m.iloc[i], delta_B_N_f.iloc[i],txt, fontsize = 7))
                #TEXTS.append(ax.annotate(txt, (delta_B_N_m.iloc[i], delta_B_N_f.iloc[i]),fontsize = 7))   
    #avoid label overlap algo
    '''
    adjust_text(
        TEXTS, 
        expand=(2, 2),
        force_explode= (2,2),
        expand_axes=True,
        arrowprops=dict(
            arrowstyle="->",  
            lw=.2
        ),
        ax=fig.axes[0])
    '''
        
    if savefig:
        plt.savefig(folder + 'plots/' + cell_class + '_Gene_Delta_Plot_Breeder-Naive_c' + str(cluster_label))

    delta_m_f_N = expr_mlog_df['N_m'] - expr_mlog_df['N_f']
    delta_m_f_B = expr_mlog_df['B_m'] - expr_mlog_df['B_f']
    d_mf = np.sqrt(delta_m_f_N**2 + delta_m_f_B**2)
    #plot first 10
    fig,ax = plt.subplots()
    ax.set_title(cell_class + ' Delta Male-Female, Cluster: '+ str(cluster_label))
    ax.scatter(delta_m_f_N,delta_m_f_B, s=1)
    plt.axvline(color = 'grey')
    plt.axhline(y=0, color = 'grey')
    plt.axline((0, 0), slope=1, color="grey", linestyle='--')
    ax.set_xlabel('delta_m_f_N')
    ax.set_ylabel('delta_m_f_B')
    # Draw a circle with the specified radius
    circle = plt.Circle((0, 0), r_mf, color='grey', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    opt_lim = get_optimal_ax_lim(delta_m_f_N,delta_m_f_B)

    ax.set_xlim([-opt_lim,opt_lim])
    ax.set_ylim([-opt_lim,opt_lim])

    ax.text(opt_lim-(0.6*opt_lim), opt_lim-(0.1*opt_lim), 'thresh = '+str(threshold_prc) + '%')
    ax.text(opt_lim-(0.6*opt_lim), opt_lim-(0.2*opt_lim), 'r_mf = '+str(r_mf))
    
    for i, txt in enumerate(list(delta_B_N_m.index)):
        if d_mf.iloc[i] > r_mf:
            #standard labeling
            ax.annotate(txt, (delta_m_f_N.iloc[i], delta_m_f_B.iloc[i]),fontsize = 5)
            #labeling using adjust text to repelling algo
            #TEXTS.append(ax.text(delta_B_N_m.iloc[i], delta_B_N_f.iloc[i],txt, fontsize = 7))
                #TEXTS.append(ax.annotate(txt, (delta_B_N_m.iloc[i], delta_B_N_f.iloc[i]),fontsize = 7))   
    #avoid label overlap algo
    '''
    adjust_text(
        TEXTS, 
        expand=(1, 1),
        force_explode= (1,1),
        expand_axes=True,
        arrowprops=dict(
            arrowstyle="->",  
            lw=.2
        ),
        ax=fig.axes[0])    
    '''
    
    if savefig:
        plt.savefig(folder + 'plots/' + cell_class + '_Gene_Delta_Plot_Male-Female_c' + str(cluster_label))
    
    
    plt.show()
    if write_to_file:
        #write updated metadata to file
        file = cell_class + '_expr_mlog_df_c' + str(cluster_label)
        expr_mlog_df.to_json(folder + 'data/' +file+'.json')

    return expr_mlog_df
