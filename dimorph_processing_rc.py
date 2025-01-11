
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
import csv
import matplotlib as mpl
today = str(date.today())

def reprocess(df_ge, meta_data_df_plis_filtered, linkage_cluster_order_filtered, cluster_indices_filtered, folder, cell_class, sort=False, write_to_file = False):
    '''uses plis_filtered metadata to reshuffle df_ge, then re process thru standard pipe except skip clustering'''
    #change matplotlib font type to make compatibile with illustrator
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    
    #get expr matrix with filtered cells but with before all genes (before feat selection)
    df_ge  = df_ge.reindex(columns = meta_data_df_plis_filtered.columns)
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
    print (np.all(df.columns == meta_data_df_plis_filtered.columns))
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

    meta_data_df_pca = meta_data_df_plis_filtered.T.copy()

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

    #linkage again for updated dendrogram
    linkage_alg = 'ward'
    dist_metric = 'euclidean'

    if sort:
        df, meta_data_df_plis_filtered_og, linkage_cluster_order_og, Z_ordered, mpg_pca, linkage_cluster_order_po = dp.inter_cluster_sort(df,
                                                    meta_data_df_plis_filtered, 
                                                    linkage_cluster_order_filtered,
                                                    n_components = 10, 
                                                    linkage_alg = linkage_alg,
                                                    dist_metric = dist_metric,
                                                    mode='rc')
        #print ('len cluster 33')
        #print (meta_data_df_plis_filtered_og.loc[:,meta_data_df_plis_filtered_og.loc['cluster_label',:]==33].shape[1])
        #print ('rc linkage matches input?:')
        #print (linkage_cluster_order_og == linkage_cluster_order_filtered)
        mpg_pca_df = pd.DataFrame(data = mpg_pca)
        plt.figure()
        ax = sns.heatmap(mpg_pca_df.corr(method='pearson'))
        plt.title('correlation pre_linkage_f')
        plt.savefig(folder + str(cell_class) + '_mpg_pca_corr_pre_linkage_f_ft42.pdf')
        plt.show()

        mpg_pca_pl_df = mpg_pca_df.reindex(columns = linkage_cluster_order_og)
        plt.figure()
        ax = sns.heatmap(mpg_pca_pl_df.corr(method='pearson'), yticklabels=True, xticklabels=True)
        plt.title('correlation post linkage_f')
        plt.savefig(folder + str(cell_class) + '_mpg_pca_corr_post_linkage_f_ft42.pdf', transparent = True)
        plt.show()

        #intracluster sort
        df_s = df.reindex(columns = meta_data_df_plis_filtered_og.columns)
        df, meta_data_df_plis_filtered_og, cluster_indices_filtered = dp.intra_cluster_sort(df_s, 
                                                                            meta_data_df_plis_filtered_og, 
                                                                            linkage_cluster_order_og,
                                                                            mode = 'rc')
        #update labels to make sequential
        meta_data_df_plis_filtered,linkage_cluster_order_filtered = dp.update_metadata_cluster_labels(linkage_cluster_order_og,meta_data_df_plis_filtered_og, mode = 'rc')
    
    
    #run enrichment again
    marker_genes_sorted_f, pos_f, ind_f, ind_s_f, mgs_f = dp.compute_marker_genes(df,
                                                    meta_data_df_plis_filtered,
                                                    cluster_indices_filtered,
                                                    linkage_cluster_order_filtered,
                                                    folder,
                                                    n_markers=5,
                                                    class_score_name=str(cell_class) + '_xi1_scores_filtered')
    df_marker_f = df.loc[marker_genes_sorted_f,:]
    marker_log_and_std_arr_f, status_df = dp.log_and_standerdize_df(df_marker_f,status_df, log = False)
    df_marker_log_and_std_f = pd.DataFrame(index = df_marker_f.index, 
                                            columns=df.columns, 
                                            data = marker_log_and_std_arr_f.T)
    df_marker_log_and_std_col_f = pd.DataFrame(data = df_marker_log_and_std_f.to_numpy(), 
                                            index = df_marker_log_and_std_f.index,
                                            columns = list(meta_data_df_plis_filtered.loc['cluster_label',:]))
    
    
    change_indices_f = dp.get_heatmap_cluster_borders(meta_data_df_plis_filtered)
    tg_f, tgfs_f = dp.get_heatmap_labels(mgs_f, ind_f, ind_s_f)
    #heatmap filtered/reenriched data
    #%matplotlib inline
    #sanity check - plotting only filtered df (clusters removed)

    fsw = dp.compute_fs_waterfall(marker_genes_sorted_f)
    #vglut1: fsw - 0.4
    #vglut: fsw - 0.6
    if cell_class == 'Vglut1':
        fsw-=0.4 
    if cell_class == 'GABA':
        fsw-=0.2 
    dp.plot_marker_heatmap(df_marker_log_and_std_col_f, 
                        pos_f, 
                        linkage_cluster_order_filtered, 
                        change_indices_f, 
                        tg_f, 
                        tgfs_f, 
                        linkage_alg,
                        dist_metric,
                        folder,
                        fs_waterfall = fsw,
                        savefig = True,
                        cell_class = str(cell_class)+'_filtered_tg_reenrich_reproc')
    
    #write to file
    if write_to_file:
        print ('writing to file')
        df.to_feather(folder + str(cell_class) + '_df_plis_filtered.feather')
        df_marker_f.to_feather(folder+str(cell_class) + '_df_marker_f' + today +'.feather')
        meta_data_df_plis_filtered.to_json(folder+str(cell_class)+ '_meta_data_df_plis_filtered' + today +'.json')

        #also save dict with clusters and markers for each clusters
        file3 = str(cell_class) + '_cl_mg_dict_f' + today

        #need to convert linkage cluster order to int (from int32) in order to store/write out dict
        lco_int = [int(x) for x in linkage_cluster_order_filtered]

        #store in dict
        cl_mg_dict_f = dict(map(lambda i,j : (i,j) , lco_int,tg_f))

        #write dict to file
        with open(folder+file3+'.json', "w") as outfile: 
            json.dump(cl_mg_dict_f, outfile)

        #write labels csv to file
        # File path for the output CSV
        output_file = folder + str(cell_class) + '_filtered_labels_rc.csv'

        # Writing the dictionary to the CSV
        with open(output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["lco_index", "tg"])
            # Write each key and values
            for key, values in cl_mg_dict_f.items():
                writer.writerow([key, ", ".join(values)])

    return df_marker_f, arr_tsne, meta_data_df_plis_filtered, meta_data_df_pca, mgs_f, ind_f, ind_s_f, df_marker_log_and_std_col_f, pos_f, linkage_cluster_order_filtered, change_indices_f
    

