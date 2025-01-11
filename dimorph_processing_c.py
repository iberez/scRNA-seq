
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
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering, leaves_list
import harmonypy as hm
from matplotlib.cm import ScalarMappable
from datetime import date
#import mpld3
import dimorph_processing as dp
today = str(date.today())

def process(df_orig, meta_data_df_orig, sex_gene_list, IEG_list, folder, cell_class):
    '''Level 2 processing taking original dataframe, metadata from marker means, plots unfiltered heatmap and returns values need for cluster removal and filtering in level 2'''
    #meta_data_df_orig = meta_data_df_mm.drop(['cluster_label'])
    df_orig = df_orig.reindex(columns=meta_data_df_orig.columns)
    meta_data_df = meta_data_df_orig.loc[:,meta_data_df_orig.loc['cell_class',:] == cell_class]
    df = df_orig.loc[:,meta_data_df_orig.loc['cell_class',:] == cell_class]
    df_bool = df.mask(df>0, other = 1)
    avg_bool_mf_df_sorted = dp.avg_bool_gene_expression_by_sex(df_bool = df_bool,
                                                                meta_data_df=meta_data_df,
                                                                num_top_genes=10,
                                                                plot_flag=1)
    status_df = dp.intialize_status_df()
    status_df.loc['cell_exclusion (l1)',:] = True
    df_ge, df_bool, meta_data_df, status_df = dp.gene_exclusion(num_cell_lwr_bound=10,
                                                        percent_cell_upper_bound=50,
                                                        df_bool=df_bool,
                                                        df = df,
                                                        meta_data_df = meta_data_df,
                                                        status_df = status_df)
    #remove sex genes
    df_ge = dp.gene_remover(sex_gene_list,df_ge)

    #remove IEG genes
    df_ge = dp.gene_remover(IEG_list,df_ge)
    count = np.isinf(df_ge).values.sum() 
    print("It contains " + str(count) + " infinite values") 
    #feature selection
    cv_df = dp.analyze_cv(df = df_ge,
                      norm_scale_factor=20000,
                      num_top_genes=30,
                      plot_flag=1,
                     use_huber = True)
    status_df = dp.intialize_status_df()
    gene_index, df, status_df = dp.get_top_cv_genes(df = df_ge, cv_df=cv_df, plot_flag=1, status_df=status_df)
    print (df.shape)
    count = np.isinf(df).values.sum() 
    print("It contains " + str(count) + " infinite values") 
    print ('#nan values', df.isnull().sum().sum())
    log_std_arr,status_df = dp.log_and_standerdize_df(df,status_df)
    #log / standerdize
    df_ls = pd.DataFrame(data = log_std_arr.T, index = df.index, columns=df.columns)

    pca_index, arr_pca, status_df = dp.analyze_pca(arr = log_std_arr, #log_std_arr
                                                n_components=log_std_arr.shape[1], #log_std_arr.shape[1]
                                                optimize_n=True,
                                                plot_flag=1, 
                                                status_df=status_df)

    meta_data_df_pca = meta_data_df.T.copy()

    vars_use = ['SampleID']
    ho = hm.run_harmony(arr_pca,meta_data_df_pca,vars_use,max_iter_harmony=20)
    hm_arr = ho.Z_corr.T

    perplexity,status_df = dp.get_perplexity(pca_arr = hm_arr, cutoff=500, plot_flag=1, status_df = status_df)

    arr_tsne,status_df = dp.do_tsne(arr = hm_arr, 
                                n_components=2,
                                n_iter=1000,
                                learning_rate=50,
                                early_exaggeration=12,
                                init='pca', 
                                perplexity = perplexity,
                                status_df = status_df)
    #DBSCAN clustering
    
    #for visualizing how eps effect clustering, uncomment below:
    #for e in range(50,80,5):
        #epsilon, minpts, status_df = dp.compute_eps(minpts = 20, eps_prc=e, arr= arr_tsne, status_df = status_df)
        #labels,n_clusters, arr, status_df = dp.do_dbscan(epsilon = epsilon, minpts = minpts, arr = arr_tsne, status_df = status_df)

    epsilon, minpts, status_df = dp.compute_eps(minpts = 20, eps_prc=65, arr= arr_tsne, status_df = status_df)
    labels,n_clusters, arr, status_df = dp.do_dbscan(epsilon = epsilon, minpts = minpts, arr = arr_tsne, status_df = status_df)

    #sort by cluster label
    df_pre_linkage_ls, meta_data_df_pre_linkage, unique_labels = dp.sort_by_cluster_label(df_ls,
                                                                               meta_data_df,
                                                                               arr,
                                                                               labels)

    #inter and intra cluster sorting
    linkage_alg = 'ward'
    dist_metric = 'euclidean'
    df_pre_linkage_raw = df.reindex(columns=meta_data_df_pre_linkage.columns)
    
    print ('verify raw prelinkage df and raw metadata df have same columns:')
    print (np.all(df_pre_linkage_raw.columns == meta_data_df_pre_linkage.columns))
    
    df_post_linkage, meta_data_df_post_linkage, linkage_cluster_order_og, Z_ordered, mpg_pca, linkage_cluster_order_po = dp.inter_cluster_sort(df_pre_linkage_raw,
                                                meta_data_df_pre_linkage, 
                                                unique_labels,
                                                n_components = 10, 
                                                linkage_alg = linkage_alg,
                                                dist_metric = dist_metric)

    mpg_pca_df = pd.DataFrame(data = mpg_pca)
    plt.figure()
    ax = sns.heatmap(mpg_pca_df.corr(method='pearson'))
    plt.title('correlation pre_linkage_f')
    plt.show()

    mpg_pca_pl_df = mpg_pca_df.reindex(columns = linkage_cluster_order_og)
    plt.figure()
    ax = sns.heatmap(mpg_pca_pl_df.corr(method='pearson'), yticklabels=True, xticklabels=True)
    plt.title('correlation post linkage_f')
    #plt.savefig(folder + '_' 'cell_class_'+'mpg_pca_corr_post_linkage_f')
    plt.show()

    #intracluster sort
    df_s = df.reindex(columns = df_post_linkage.columns)
    df_plis, meta_data_df_plis_og, cluster_indices = dp.intra_cluster_sort(df_s, 
                                                                meta_data_df_post_linkage, 
                                                                linkage_cluster_order_og)

    #update labels to make sequential
    meta_data_df_plis,linkage_cluster_order = dp.update_metadata_cluster_labels(linkage_cluster_order_og,meta_data_df_plis_og)
    print ('check if df plis columns equal metadata df plis columns:')
    print (df_plis.columns == meta_data_df_plis.columns)
    #enrichment analysis
    marker_genes_sorted, pos, ind, ind_s, mgs = dp.compute_marker_genes(df_plis,
                                                    meta_data_df_plis,
                                                    cluster_indices,
                                                    linkage_cluster_order,
                                                    folder,
                                                    n_markers=5,
                                                    class_score_name=str(cell_class) + '_xi1_scores')
    df_marker = df_plis.loc[marker_genes_sorted,:]
    marker_log_and_std_arr, status_df = dp.log_and_standerdize_df(df_marker,status_df, log = False)
    df_marker_log_and_std = pd.DataFrame(index = df_marker.index, 
                                            columns=df_plis.columns, 
                                            data = marker_log_and_std_arr.T)
    df_marker_log_and_std_col = pd.DataFrame(data = df_marker_log_and_std.to_numpy(), 
                                            index = df_marker_log_and_std.index,
                                            columns = list(meta_data_df_plis.loc['cluster_label',:]))
    
    
    change_indices = dp.get_heatmap_cluster_borders(meta_data_df_plis)
    tg, tgfs = dp.get_heatmap_labels(mgs, ind, ind_s)
    #heatmap filtered/reenriched data
    #%matplotlib inline
    #sanity check - plotting only filtered df (clusters removed)
    
    fsw = dp.compute_fs_waterfall(marker_genes_sorted)
    if cell_class == 'Vglut1':
        fsw = fsw - 0.6
    if cell_class == 'GABA':
        fsw = fsw + 1.3
    dp.plot_marker_heatmap(df_marker_log_and_std_col, 
                        pos, 
                        linkage_cluster_order, 
                        change_indices, 
                        tg, 
                        tgfs, 
                        linkage_alg,
                        dist_metric,
                        folder,
                        fs_waterfall = fsw,
                        savefig = True,
                        cell_class = str(cell_class)+'_unfiltered_proc')
    
    print (np.all(df_marker_log_and_std.columns == meta_data_df_plis.columns))

    return df_marker, arr_tsne, meta_data_df_plis, linkage_cluster_order, df_marker_log_and_std_col, df_plis, cluster_indices, df_marker_log_and_std, df_ge