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

def compute_cluster_sex_stats(meta_data_df):
    '''takes metadata and computes percentage of 'Breeder-F', 'Breeder-M', 'Naïve-F', 'Naïve-M' for each cluster as well as n_mice.
    returns results in dataframe'''
    #get lists of unique mice_id, groups, and markers
    mice_id = np.unique(meta_data_df.loc['ChipID'])
    sample_id = np.unique(meta_data_df.loc['SampleID'])
    groups = np.unique(meta_data_df.loc['Group'])
    m_list = np.unique(meta_data_df.loc['markers'])
    c_label = np.unique(meta_data_df.loc['cluster_label'])
    #build sex stats df
    sex_stats_df = pd.DataFrame(columns = ['markers','Breeder-F','Breeder-M' ,'Naïve-F' , 'Naïve-M','num_sample_ids','sample_ids'])
    marker_genes,idx = np.unique(meta_data_df.loc['markers'],return_index=True)
    sex_stats_df['markers'] = marker_genes[np.argsort(idx)]
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

def compute_group_gene_expression_differences(df, meta_data_df,cluster_label,threshold_prc, r_bn,r_mf, cl_mg_dict, cell_class,folder, normalize = False, n_factor = 20000, mode = 'delta', sig_genes_df = None, savefig = False, write_to_file = False):
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
    #toggle gene index threshold to determine if thresholding is done on clusters or entire class
    thresh_in_cluster = False
    #marker gene expression within a cluster
    c_expr = df.loc[:,meta_data_df.loc['cluster_label']==cluster_label]
    #print (c_expr.iloc[:3,:3])
    #print (c_expr.shape)
    if thresh_in_cluster:

        c_expr_bool = c_expr.mask(c_expr>0, other = 1)
        gene_sum =  np.array(c_expr_bool.sum(axis=1))
        #print (gene_sum)
        #print (gene_sum.shape)
        gene_sum = np.reshape(gene_sum,(c_expr_bool.shape[0],1))
        #print (gene_sum.shape)
        if normalize:
            c_expr = (c_expr/gene_sum)*n_factor
        #print (c_expr.iloc[:3,:3])
        #print (c_expr.shape)
        gene_threshold = (threshold_prc/100)*c_expr_bool.shape[1]
        genes_to_keep_ind = []
        for i,v in enumerate(gene_sum):
            if v > gene_threshold:
                genes_to_keep_ind.append(i)
        #update c_expr keeping only genes above threshold
        c_expr = c_expr.iloc[genes_to_keep_ind,:]

        #print (c_expr.shape)

    
    else:
        expr = df
        expr_bool = expr.mask(expr>0, other = 1)
        gene_sum =  np.array(expr_bool.sum(axis=1))
        gene_sum = np.reshape(gene_sum,(expr_bool.shape[0],1))
        #print (gene_sum.shape)
        if normalize:
            expr = (expr/gene_sum)*n_factor
        gene_threshold = (threshold_prc/100)*expr_bool.shape[1]
        genes_to_keep_ind = []
        for i,v in enumerate(gene_sum):
            if v > gene_threshold:
                genes_to_keep_ind.append(i)
        #update c_expr keeping only genes above threshold
        c_expr = c_expr.iloc[genes_to_keep_ind,:]

        print (c_expr.shape)


    n_genes = c_expr.shape[0]
    n_cells = c_expr.shape[1]
    c_metadata = meta_data_df.loc[:,meta_data_df.loc['cluster_label']==cluster_label]

    #isolated expression of 4 groups within cluster
    N_f_expr = c_expr.loc[:,c_metadata.loc['Group',:]=='Naïve-F']
    B_f_expr = c_expr.loc[:,c_metadata.loc['Group',:]=='Breeder-F']
    N_m_expr = c_expr.loc[:,c_metadata.loc['Group',:]=='Naïve-M']
    B_m_expr = c_expr.loc[:,c_metadata.loc['Group',:]=='Breeder-M']
    expr_raw_df = pd.concat([N_f_expr,B_f_expr,N_m_expr,B_m_expr], axis = 1)
    #expr_raw_df = expr_raw_df.rename(columns={0: "N_f", 1: "B_f",2:"N_m", 3:"B_m" })
    
    c_metadata_df = c_metadata.reindex(columns = expr_raw_df.columns)
    #get counts of each category
    N_f_expr_cnts = N_f_expr.shape[1]
    B_f_expr_cnts = B_f_expr.shape[1]
    N_m_expr_cnts = N_m_expr.shape[1]
    B_m_expr_cnts = B_m_expr.shape[1]
    counts_df = pd.DataFrame(index = [cluster_label], columns=['N_f_cnts', 'B_f_cnts','N_m_cnts','B_m_cnts'])
    counts_df.loc[cluster_label,'N_f_cnts'] = N_f_expr_cnts
    counts_df.loc[cluster_label,'B_f_cnts'] = B_f_expr_cnts
    counts_df.loc[cluster_label,'N_m_cnts'] = N_m_expr_cnts
    counts_df.loc[cluster_label,'B_m_cnts'] = B_m_expr_cnts

    #take mean of log2+1 for each gene
    N_f_expr_mlog = np.mean(np.log2(N_f_expr+1),axis = 1)
    B_f_expr_mlog = np.mean(np.log2(B_f_expr+1),axis = 1)
    N_m_expr_mlog = np.mean(np.log2(N_m_expr+1),axis = 1)
    B_m_expr_mlog = np.mean(np.log2(B_m_expr+1),axis = 1)
    expr_mlog_df = pd.concat([N_f_expr_mlog,B_f_expr_mlog,N_m_expr_mlog,B_m_expr_mlog], axis = 1)
    expr_mlog_df = expr_mlog_df.rename(columns={0: "N_f", 1: "B_f",2:"N_m", 3:"B_m" })
    
    #if sig_genes mode, get sig genes for each plot from sig_genes_df
    if mode == 'sig_genes':
        tmp  = sig_genes_df.loc[sig_genes_df.loc[:,0]==cluster_label,:]
        bn_sig_genes = tmp[tmp[1].isin(['ΔBN_m', 'ΔBN_f'])][2].tolist()
        mf_sig_genes = tmp[tmp[1].isin(['Δmf_B', 'Δmf_N'])][2].tolist()
        ΔBN_m_sig_genes = tmp[tmp[1].isin(['ΔBN_m'])][2].tolist()
        ΔBN_f_sig_genes = tmp[tmp[1].isin(['ΔBN_f'])][2].tolist()
        Δmf_B_sig_genes = tmp[tmp[1].isin(['Δmf_B'])][2].tolist()
        Δmf_N_sig_genes = tmp[tmp[1].isin(['Δmf_N'])][2].tolist()

    #get delta Breeder - Naive, within each sex
    delta_B_N_m = expr_mlog_df['B_m'] - expr_mlog_df['N_m']
    delta_B_N_f = expr_mlog_df['B_f'] - expr_mlog_df['N_f']
    d_bn = np.sqrt(delta_B_N_m**2 + delta_B_N_f**2)
    
    prc_top = 0.1 #get prc_top % of farthest out genes from origin
    far_out_index = round(n_genes*(prc_top/100))
    #get farthest out genes
    d_bn_tmp = pd.DataFrame(index=delta_B_N_m.index, data=d_bn, columns=['d_bn'])
    d_bn_sorted = d_bn_tmp.sort_values(by=['d_bn'], ascending=False)
    #print (far_out_index)
    far_out_genes = d_bn_sorted.index[:far_out_index]
    print (far_out_genes)
    print (len(far_out_genes))
    fig,ax = plt.subplots()
    ax.set_title(cell_class + ' Δ Breeder-Naive, Cluster: ' + str(cluster_label) + '-' + ' '.join(cl_mg_dict[str(cluster_label)]))
    #ax.scatter(delta_B_N_m,delta_B_N_f, s=1)
    ax.set_xlabel(f'Δ_B_m({B_m_expr_cnts})_N_m({N_m_expr_cnts})')
    ax.set_ylabel(f'Δ_B_f({B_f_expr_cnts})_N_f({N_f_expr_cnts})')
    plt.axvline(color = 'grey')
    plt.axhline(y=0, color = 'grey')
    plt.axline((0, 0), slope=1, color="grey", linestyle='--')
    # Draw a circle with the specified radius
    circle = plt.Circle((0, 0), r_bn, color='grey', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    #opt_lim = get_optimal_ax_lim(delta_B_N_m,delta_B_N_f)
    opt_lim = 6
    #if mode == 'delta':
    ax.set_xlim([-opt_lim,opt_lim])
    ax.set_ylim([-opt_lim,opt_lim])
    fs = 10
    
    s_factor = opt_lim/2
    ax.text(-.25*s_factor, 0.15*s_factor, 'thresh = '+str(threshold_prc) + '%', fontsize = fs)
    ax.text(-.25*s_factor, 0.05*s_factor, 'r_bn = '+str(r_bn), fontsize = fs)
    ax.text(-.25*s_factor, -.05*s_factor, 'n_genes = '+str(n_genes), fontsize = fs)
    ax.text(-.25*s_factor, -.15*s_factor, 'n_cells = '+str(n_cells), fontsize = fs)

    TEXTS = []
    #plot and label points outside of circle radius
    for i, txt in enumerate(list(delta_B_N_m.index)):
        if d_bn.iloc[i] > r_bn: 
            ax.scatter(delta_B_N_m.iloc[i],delta_B_N_f.iloc[i], s=1, c='blue')       
            #switch labeling depending if sig genes is passed
            if mode == 'sig_genes':
                #print ('sig gene detected')
                #print (txt)
                if txt in ΔBN_m_sig_genes:
                    ax.scatter(delta_B_N_m.iloc[i],delta_B_N_f.iloc[i], s=1, c='red', marker="^", label = 'ΔBN_m_sig_genes')
                    ax.annotate(txt, (delta_B_N_m.iloc[i], delta_B_N_f.iloc[i]),fontsize = fs, c = 'red')
                #for g in ΔBN_f_sig_genes:
                    #if g == txt:
                if txt in ΔBN_f_sig_genes: 
                    ax.scatter(delta_B_N_m.iloc[i],delta_B_N_f.iloc[i], s=1, c='green', marker="o", label = 'ΔBN_f_sig_genes')
                    ax.annotate(txt, (delta_B_N_m.iloc[i], delta_B_N_f.iloc[i]),fontsize = fs, c = 'green')    
            #conditional labeling if gene is far out
            if mode == 'delta':
                if txt in list(far_out_genes):
                    ax.annotate(txt, (delta_B_N_m.iloc[i], delta_B_N_f.iloc[i]),fontsize = fs)
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
    if mode == 'sig_genes':
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
    #ax.legend()
    if mode == 'delta':
        if savefig:
            plt.savefig(folder + 'plots/' + cell_class + '_Gene_Delta_Plot_Breeder-Naive_c' + str(cluster_label))
    if mode == 'sig_genes':
        if savefig:
            plt.savefig(folder + 'sig_plots/' + cell_class + '_Gene_Delta_Plot_Breeder-Naive_c' + str(cluster_label))

    delta_m_f_N = expr_mlog_df['N_m'] - expr_mlog_df['N_f']
    delta_m_f_B = expr_mlog_df['B_m'] - expr_mlog_df['B_f']
    d_mf = np.sqrt(delta_m_f_N**2 + delta_m_f_B**2)

    #get farthest out genes
    d_mf_tmp = pd.DataFrame(index=delta_m_f_N.index, data=d_mf, columns=['d_mf'])
    d_mf_sorted = d_mf_tmp.sort_values(by=['d_mf'], ascending=False)
    far_out_genes = d_mf_sorted.index[:far_out_index]

    #plot first 10
    fig,ax = plt.subplots()
    ax.set_title(cell_class + ' Δ Male-Female, Cluster: '+ str(cluster_label)  + '-' + ' '.join(cl_mg_dict[str(cluster_label)]))
    #ax.scatter(delta_m_f_N,delta_m_f_B, s=1)
    plt.axvline(color = 'grey')
    plt.axhline(y=0, color = 'grey')
    plt.axline((0, 0), slope=1, color="grey", linestyle='--')
    ax.set_xlabel(f'Δ_m_N({N_m_expr_cnts})_f_N({N_f_expr_cnts})')
    ax.set_ylabel(f'Δ_m_B({B_m_expr_cnts})_f_B({B_f_expr_cnts})')
    # Draw a circle with the specified radius
    circle = plt.Circle((0, 0), r_mf, color='grey', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    
    #opt_lim = get_optimal_ax_lim(delta_m_f_N,delta_m_f_B)
    #get_optimal_ax_lim seems to be buggy, use fixed lim
    opt_lim = 6
    #print (opt_lim)
    #if mode == 'delta':
    ax.set_xlim([-opt_lim,opt_lim])
    ax.set_ylim([-opt_lim,opt_lim])
    

    ax.text(-.25*s_factor, .15*s_factor, 'thresh = '+str(threshold_prc) + '%',fontsize = fs)
    ax.text(-.25*s_factor, 0.05*s_factor, 'r_mf = '+str(r_mf), fontsize = fs)
    ax.text(-.25*s_factor, -0.05*s_factor, 'n_genes = '+str(n_genes),fontsize = fs)
    ax.text(-.25*s_factor, -.15*s_factor, 'n_cells = '+str(n_cells), fontsize = fs)
    
    for i, txt in enumerate(list(delta_B_N_m.index)):
        if d_mf.iloc[i] > r_mf:
            ax.scatter(delta_m_f_N.iloc[i],delta_m_f_B.iloc[i], s=1, c = 'blue')
            #switch labeling depending if sig genes is passed
            if mode == 'sig_genes':
                #print ('sig gene detected')
                #print (txt)
                if txt in Δmf_B_sig_genes:
                    ax.scatter(delta_m_f_N.iloc[i],delta_m_f_B.iloc[i], s=1, c='red', marker="^", label = "Δmf_B_sig_genes")
                    ax.annotate(txt, (delta_m_f_N.iloc[i], delta_m_f_B.iloc[i]),fontsize = fs, c = 'red')
                if txt in Δmf_N_sig_genes:
                    ax.scatter(delta_m_f_N.iloc[i],delta_m_f_B.iloc[i], s=1, c='green', marker="o", label = "Δmf_N_sig_genes")
                    ax.annotate(txt, (delta_m_f_N.iloc[i], delta_m_f_B.iloc[i]),fontsize = fs, c = 'green')

            #conditional labeling if gene is far out
            if mode == 'delta':
                if txt in list(far_out_genes):
                    ax.annotate(txt, (delta_m_f_N.iloc[i], delta_m_f_B.iloc[i]),fontsize = fs)
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
    if mode == 'sig_genes':
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    if mode == 'delta':
        if savefig:
            plt.savefig(folder + 'plots/' + cell_class + '_Gene_Delta_Plot_Male-Female_c' + str(cluster_label))
    if mode == 'sig_genes':
        if savefig:
            plt.savefig(folder + 'sig_plots/' + cell_class + '_Gene_Delta_Plot_Male-Female_c' + str(cluster_label))
    
    
    plt.show()
    if mode == 'delta':
        if write_to_file:
            #write updated metadata to file
            file = cell_class + '_expr_mlog_df_c' + str(cluster_label)
            expr_mlog_df.to_json(folder + 'data/' +file+'.json')
            file2 = cell_class + '_expr_raw_df_c' + str(cluster_label)
            expr_raw_df.to_json(folder + 'data/' +file2+'.json')
            file3 = cell_class + '_metadata_df_c' + str(cluster_label)
            c_metadata_df.to_json(folder + 'data/' +file3+'.json')

    return expr_mlog_df, counts_df

def get_plot_labels(mg_cl_dict):
    '''formats key/values into single labels used for plots in sex stats nb'''
    labels = [f"{key} {value}" for key, value in mg_cl_dict.items()]
    formatted_labels = []
    # Iterate through each element in the list
    for item in labels:
        # Split the string into the number and the list part
        number, string_list = item.split(" ", 1)
        
        # Remove the brackets and quotes from the string list, and split into separate items
        string_list = string_list.strip("[]").replace("'", "").split(", ")
        
        # Join the strings with a hyphen
        joined_string = "-".join(string_list)
        
        # Format the result as "number joined_string" and append to the result list
        formatted_labels.append(f"{number} {joined_string}")
    
    return formatted_labels

def do_u_test_w_fdr(expr1,expr2):
    '''Computes mann whiteney u test for each gene (row) between expr1 and expr2. returns results in dataframe.'''
    #construct df to store results of mann whitney test for every gene
    u_test_df = pd.DataFrame(index = expr1.index, columns = ['U','p','p_adj'])
    #do mann whiteney for every gene
    for i in u_test_df.index:
        U, p = mannwhitneyu(expr1.loc[i,:],expr2.loc[i,:], alternative='two-sided', method="exact")
        u_test_df.loc[i,'U'] = U
        u_test_df.loc[i,'p'] = p
        #compute p_adj using false discovery rate
    u_test_df.loc[:,'p_adj'] = false_discovery_control(np.array(u_test_df['p']).astype(float), method = 'bh')
    
    return u_test_df

def run_stat_test(delta_data_folder, index ,output_folder, cell_class, write_to_file = False):
    '''wraps do_u_test_w_fdr(), stores results in dataframe in output folder'''
    with open (delta_data_folder + cell_class+ '_expr_raw_df_c'+str(index)+ '.json') as json_data:
        expr_raw_df = json.load(json_data)
    with open (delta_data_folder + cell_class + '_metadata_df_c' +str(index) + '.json') as json_data:
        metadata_df = json.load(json_data)
    
    expr_raw_df = pd.DataFrame.from_dict(expr_raw_df, orient='columns')

    metadata_df = pd.DataFrame.from_dict(metadata_df, orient='columns')
    
    #print (expr_raw_df.columns == metadata_df.columns)
    
    N_f_expr = expr_raw_df.loc[:,metadata_df.loc['Group',:]=='Naïve-F']
    B_f_expr = expr_raw_df.loc[:,metadata_df.loc['Group',:]=='Breeder-F']
    N_m_expr = expr_raw_df.loc[:,metadata_df.loc['Group',:]=='Naïve-M']
    B_m_expr = expr_raw_df.loc[:,metadata_df.loc['Group',:]=='Breeder-M']
    
    #print (B_m_expr.loc['Cops7a',:].shape)
    #print (B_f_expr.loc['Cops7a',:].shape)

    #do mann whiteney u test for each axis
    U_test_BN_m = do_u_test_w_fdr(B_m_expr,N_m_expr)
    U_test_BN_f = do_u_test_w_fdr(B_f_expr,N_f_expr)
    U_test_mf_B = do_u_test_w_fdr(B_m_expr,B_f_expr)
    U_test_mf_N = do_u_test_w_fdr(N_m_expr,N_f_expr)
    
    if write_to_file:
        U_test_BN_m.to_json(output_folder +cell_class + '_U_test_BN_m_c' + str(index)+'.json')
        U_test_BN_f.to_json(output_folder +cell_class + '_U_test_BN_f_c' + str(index)+'.json')
        U_test_mf_B.to_json(output_folder +cell_class + '_U_test_mf_B_c' + str(index)+'.json')
        U_test_mf_N.to_json(output_folder +cell_class + '_U_test_mf_N_c' + str(index)+'.json')
    
    return U_test_BN_m, U_test_BN_f, U_test_mf_B, U_test_mf_N

def volcano_plot(U_test_df,delta_df, cell_class, index, cl_mg_dict, all_counts_df, test_name, output_folder, savefig = False):    
    #build volcano dataframe for volcano plot
    v_df = pd.DataFrame(index = U_test_df.index, columns = ['delta', '-log10(p_adj)'])
    v_df['delta'] = delta_df
    v_df['-log10(p_adj)'] = -np.log10(U_test_df['p_adj'].astype('float64'))
    #print (v_df['-log10(p_adj)'])
    # Define significance threshold
    alpha = 0.05

    # Plot volcano plot
    fig,ax = plt.subplots()
    ax.scatter(v_df['delta'], v_df['-log10(p_adj)'], color='blue', alpha=0.5)
    ax.set_title(cell_class +  ' ' + test_name + ' ' + '(' + str(all_counts_df.iloc[0]) + ',' + str(all_counts_df.iloc[1]) + ')' + ' Cluster: ' + str(index) + '-' + ' '.join(cl_mg_dict[str(index)]))
    ax.axhline(-np.log10(alpha), color='red', linestyle='--', linewidth=1)
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1)
    ax.axvline(x=-1, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('ΔLog2 Fold Change')
    ax.set_ylabel('-log10(p_adj)')
    #add gene label if beyond alpha and log_fc<-1 or >1
    #also make note of index - append to text file
    for i, txt in enumerate(list(v_df.index)):
        #write out stat values for all genes
        csv_row = [[str(index),test_name,txt,v_df.loc[txt,'delta'], v_df.loc[txt,'-log10(p_adj)']]]
        file = open(output_folder + 'all_genes_fix_t.csv', 'a+', newline ='')
        with file:    
            write = csv.writer(file)
            write.writerows(csv_row)


        if v_df.loc[txt, '-log10(p_adj)'] > -np.log10(alpha):
            if v_df.loc[txt, 'delta'] > 1 or v_df.loc[txt, 'delta'] < -1:
                #standard labeling
                ax.annotate(txt, (v_df.loc[txt,'delta'], v_df.loc[txt,'-log10(p_adj)']),fontsize = 10)
                #with open(output_folder + "sig_gene_index_list.txt", "a") as myfile:
                    #myfile.write(str(test_name)+'_'+str(index) + '\n')
                #write index, test namem and genes as row into csv
                csv_row = [[str(index),test_name,txt,v_df.loc[txt,'delta'], v_df.loc[txt,'-log10(p_adj)']]]
                #print (csv_row)
                # opening the csv file in 'a+' mode
                file = open(output_folder + 'sig_genes_fix_t.csv', 'a+', newline ='')
                
                # writing the data into the file
                with file:    
                    write = csv.writer(file)
                    write.writerows(csv_row)


    plt.grid(True)
    plt.show()

    if savefig:
        plt.savefig(output_folder + 'plots/' + cell_class + '_volcano_plot_' + test_name +'_c' + str(index))
    
    return v_df

def run_volcano_analysis(delta_data_folder, utest_data_folder,output_folder, index , cell_class, mg_cl_dict, all_counts_df, savefig = 'False', write_to_file = False):
    '''wraps volcano_plot() to do volcano analysis for index specified cluster /cell class. if write to file is true, writes data for plots to file'''
    with open (utest_data_folder + cell_class + '_U_test_BN_m_c'+str(index)+ '.json') as json_data:
        U_test_BN_m = json.load(json_data)
    U_test_BN_m = pd.DataFrame.from_dict(U_test_BN_m, orient='columns')
    with open (utest_data_folder + cell_class + '_U_test_BN_f_c'+str(index)+'.json') as json_data:
        U_test_BN_f = json.load(json_data)
    U_test_BN_f = pd.DataFrame.from_dict(U_test_BN_f, orient='columns')
    with open (utest_data_folder + cell_class + '_U_test_mf_N_c'+str(index)+'.json') as json_data:
        U_test_mf_N = json.load(json_data)
    U_test_mf_N = pd.DataFrame.from_dict(U_test_mf_N, orient='columns')
    with open (utest_data_folder + cell_class + '_U_test_mf_B_c'+str(index)+'.json') as json_data:
        U_test_mf_B = json.load(json_data)
    U_test_mf_B = pd.DataFrame.from_dict(U_test_mf_B, orient='columns')
    
    with open (delta_data_folder + cell_class + '_expr_mlog_df_c'+str(index)+'.json') as json_data:
        expr_mlog_df = json.load(json_data)
    expr_mlog_df = pd.DataFrame.from_dict(expr_mlog_df, orient='columns')
    
    delta_B_N_m = expr_mlog_df['B_m'] - expr_mlog_df['N_m']
    delta_B_N_f = expr_mlog_df['B_f'] - expr_mlog_df['N_f']
    delta_m_f_N = expr_mlog_df['N_m'] - expr_mlog_df['N_f']
    delta_m_f_B = expr_mlog_df['B_m'] - expr_mlog_df['B_f']
    
    v_BN_m_df = volcano_plot(U_test_BN_m,delta_B_N_m,'GABA', index, mg_cl_dict, all_counts_df.loc[index,['B_m_cnts','N_m_cnts']], 'ΔBN_m',output_folder, savefig)
    v_BN_f_df = volcano_plot(U_test_BN_f,delta_B_N_f,'GABA', index, mg_cl_dict, all_counts_df.loc[index,['B_f_cnts','N_f_cnts']], 'ΔBN_f',output_folder, savefig)
    v_mf_B_df = volcano_plot(U_test_mf_B,delta_m_f_B,'GABA', index, mg_cl_dict, all_counts_df.loc[index,['B_m_cnts','B_f_cnts']], 'Δmf_B',output_folder, savefig)
    v_mf_N_df = volcano_plot(U_test_mf_N,delta_m_f_N,'GABA', index, mg_cl_dict, all_counts_df.loc[index,['N_f_cnts','N_m_cnts']], 'Δmf_N',output_folder, savefig)
    
    if write_to_file:
        v_BN_m_df.to_json(output_folder + 'data/' + cell_class + '_v_BN_m_df_c' + str(index)+'.json')
        v_BN_f_df.to_json(output_folder + 'data/' + cell_class + '_v_BN_f_df_c' + str(index)+'.json')
        v_mf_B_df.to_json(output_folder + 'data/' + cell_class + '_v_mf_B_df_c' + str(index)+'.json')
        v_mf_N_df.to_json(output_folder + 'data/' + cell_class + '_v_mf_N_df_c' + str(index)+'.json')
    
    return v_BN_m_df, v_BN_f_df, v_mf_B_df, v_mf_N_df