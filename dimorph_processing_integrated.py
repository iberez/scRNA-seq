
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
from sklearn.cluster import KMeans
today = str(date.today())

def enhance_metadata(meta_data_df_pca, sd_metadata_path,amy_raw_md_path,output_folder,fn, write_to_file = False):
    '''enhance _pca metadata of integrated amy/sd datasets with group, sex, age informatin'''
    rows_2_add = ['Age','Group','Sex']
    meta_data_df = meta_data_df_pca.T
    sd_metadata_subset = meta_data_df.loc[:,meta_data_df.loc['dataset'] == 'sd']
    sd_metadata = pd.read_json(sd_metadata_path)
    sd_metadata = sd_metadata.reindex(columns = sd_metadata_subset.columns)
    sd_metadata_enhanced = sd_metadata.loc[rows_2_add,:]
    
    amy_metadata_subset = meta_data_df.loc[:,meta_data_df.loc['dataset'] == 'amy']
    amy_raw_md = pd.read_excel(amy_raw_md_path)
    #remove nans
    amy_raw_md = amy_raw_md.iloc[:23,:]
    a_sid_2_group_dict = dict(zip(np.array(amy_raw_md['SampleID:string']), np.array(amy_raw_md['Group'])))
    a_sid_2_age_dict = dict(zip(np.array(amy_raw_md['SampleID:string']), np.array(amy_raw_md['Age:string'])))
    a_sid_2_sex_dict = dict(zip(np.array(amy_raw_md['SampleID:string']), np.array(amy_raw_md['Sex:string'])))
    amy_age_vector = [a_sid_2_age_dict[x] for x in amy_metadata_subset.loc['SampleID']]
    amy_group_vector = [a_sid_2_group_dict[x] for x in amy_metadata_subset.loc['SampleID']]
    amy_sex_vector = [a_sid_2_sex_dict[x] for x in amy_metadata_subset.loc['SampleID']]
    amy_metadata_enhanced = pd.DataFrame(index = rows_2_add,
                                        columns= amy_metadata_subset.columns)
    amy_metadata_enhanced.loc[rows_2_add[0],:] = amy_age_vector
    amy_metadata_enhanced.loc[rows_2_add[1],:] = amy_group_vector
    amy_metadata_enhanced.loc[rows_2_add[2],:] = amy_sex_vector

    amy_sd_enhanced_metadata = pd.concat([amy_metadata_enhanced, sd_metadata_enhanced], axis=1)

    #update metadata with added rows for age, group, sex
    meta_data_df_ags = pd.concat([meta_data_df, amy_sd_enhanced_metadata], axis=0)
    if write_to_file:
        meta_data_df_ags.to_json(output_folder + fn + '.json')
    return meta_data_df_ags

def format_amy_df(amy_df):
    '''format amy_df to have genes as index and samples as columns'''
    #remove first row
    amy_df = amy_df.iloc[4:,:]
    #set first column as index
    amy_df = amy_df.set_index(amy_df.columns[0])
    #remove empty columns
    amy_df = amy_df.dropna(axis=1, how='all')
    #remove duplicates
    amy_df = amy_df.loc[~amy_df.index.duplicated(keep='first')]
    return amy_df

def initialize_expr_matrix(df_orig_amy, df_orig_sd, meta_data_df, sex_gene_list, IEG_list, output_folder):
        #remove sex genes and IEGs from df_orig_amy and df_orig_sd
    df_orig_sd = dp.gene_remover(IEG_list,df_orig_sd)
    df_orig_sd = dp.gene_remover(sex_gene_list,df_orig_sd)

    df_orig_amy = dp.gene_remover(IEG_list,df_orig_amy)
    df_orig_amy = dp.gene_remover(sex_gene_list,df_orig_amy)
    
    #split meta_data by dataset
    meta_data_df_sd = meta_data_df.loc[:,meta_data_df.loc['dataset'] == 'sd']
    meta_data_df_amy = meta_data_df.loc[:,meta_data_df.loc['dataset'] == 'amy']
    #order each df to match meta_data_df ordering
    df_orig_sd = df_orig_sd.reindex(columns=meta_data_df_sd.columns)
    df_orig_amy = df_orig_amy.reindex(columns=meta_data_df_amy.columns) 
    #concatenate amy and sd dataframes using union
    intersect_genes = df_orig_amy.index.intersection(df_orig_sd.index)
    print (len(intersect_genes), 'genes in intersection of amy and sd')
    df_orig_amy = df_orig_amy.reindex(index=intersect_genes)
    df_orig_sd = df_orig_sd.reindex(index=intersect_genes)
    df = pd.concat([df_orig_amy, df_orig_sd], axis=1)
    df = df.astype(int)
    return df


def process(df, meta_data_df, arr_tsne, cell_class, folder, write_to_file = False):
    '''Takes integrated integrated data, process like dimorph_processing.py but starting from dbscan'''
    


    log_std_arr = dp.log_and_standerdize_df(df, log = True)
    df_ls = pd.DataFrame(data = log_std_arr.T, index = df.index, columns=df.columns)
    
    '''
    #kmeans clustering
    k = 70
    kmeans = KMeans(n_clusters=k, random_state=42).fit(arr_tsne)
    labels = kmeans.labels_

    cmap = dict(zip(np.unique(labels), sns.color_palette("hsv", len(np.unique(labels)))))
    fig,ax = plt.subplots(figsize = (20,20))
    ax.set_box_aspect(1)
    ax.scatter(arr_tsne[:,0], arr_tsne[:,1], s=1, c=[cmap[x] for x in labels], alpha=0.5)
    arr_df = pd.DataFrame(arr_tsne, columns=['tsne-1','tsne-2'])
    for label in set(labels):
        if label != -1:
            cluster_median = arr_df[labels == label].median()
            ax.annotate(label, cluster_median, fontsize=8, color='black',
                        ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))
    plt.xticks([])
    plt.yticks([])
    plt.title('K-Means Clustering of GABA Cells, k = ' + str(k))
    #plt.savefig(dir + 'kmeans_plots/kmeans_gaba_k_' + str(k) + '.png', bbox_inches='tight')
    #plt.close(fig)
    print(f'K-Means clustering plot saved for k = {k}')
    plt.show()    
    '''

    
    
    #DBSCAN clustering 
    minpts = 30
    eps_prc = 80
    epsilon, minpts = dp.compute_eps(minpts = minpts, eps_prc=eps_prc, arr= arr_tsne)
    opt_eps = dp.optimal_eps(arr_tsne, k = minpts)
    outfolder = folder + str(cell_class) + '_l2_processed_eps_prc_'+str(eps_prc) + '_minpts_' + str(minpts) +'/'
    os.makedirs(outfolder, exist_ok=True)

    print ('Optimal epsilon:', opt_eps)
    labels,n_clusters, arr = dp.do_dbscan_keep_noise(epsilon = epsilon, minpts = minpts, arr = arr_tsne,savefig=True,out_folder=outfolder)
    
    
    #sort by cluster label
    df_pre_linkage_ls, meta_data_df_pre_linkage, unique_labels,arr_df_sorted = dp.sort_by_cluster_label(df_ls,
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
    plt.savefig(outfolder + '_' 'cell_class_'+'mpg_pca_corr_post_linkage_f')
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
                                                    outfolder,
                                                    n_markers=5,
                                                    class_score_name=str(cell_class) + '_xi1_scores')
    df_marker = df_plis.loc[marker_genes_sorted,:]
    marker_log_and_std_arr = dp.log_and_standerdize_df(df_marker, log = True)
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

    dp.plot_marker_heatmap(df_marker_log_and_std_col, 
                        pos, 
                        linkage_cluster_order, 
                        change_indices, 
                        tg, 
                        tgfs, 
                        linkage_alg,
                        dist_metric,
                        outfolder,
                        fs_waterfall = fsw,
                        savefig = True,
                        cell_class = str(cell_class) + '_heatmap',)
    
    print (np.all(df_marker_log_and_std.columns == meta_data_df_plis.columns))
    
    #sort arr_df_sorted by linkage_cluster_order_og 
    arr_df_sorted['labels'] = arr_df_sorted['labels'].astype(int)  # ensure int type if needed

    arr_df_sorted['labels'] = pd.Categorical(
        arr_df_sorted['labels'],
        categories=linkage_cluster_order_og,
        ordered=True
    )
    arr_df_sorted = arr_df_sorted.sort_values('labels')


    #updated dbscan plot with sorted cluster labels 
    l = np.array(meta_data_df_plis.loc['cluster_label',:])
    fig,ax = plt.subplots()
    ax.set_box_aspect(1)
    ax.axis('off')
    p = sns.scatterplot(data = arr_df_sorted,
                        x = 'tsne-1',
                        y= 'tsne-2',
                        hue = l, 
                        legend = "full", 
                        palette = "deep",
                        s = 1)
    #sns.move_legend(p, "upper right", bbox_to_anchor = (1.17, 1.), title = 'Clusters')
    # Annotate with cluster labels at the median of each cluster
    for label in set(l):
        if label != -1:
            cluster_data = arr_df_sorted[l == label]
            cluster_median = cluster_data[['tsne-1', 'tsne-2']].median()
            ax.annotate(label, cluster_median, fontsize=8, color='black',
                        ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))

    p.legend_.remove()

    plt.xticks([])
    plt.yticks([])
    plt.savefig(outfolder + 'dbscan_plot_sorted.png')
    plt.show()

    if write_to_file:
        print ('writing to file...')
        df.to_feather(outfolder + 'df.feather')
        df_marker.to_feather(outfolder + 'df_marker.feather')
        df_marker_log_and_std.to_feather(outfolder + 'df_marker_log_and_std.feather')   
        df_plis.to_feather(outfolder + 'df_plis.feather')
        meta_data_df_plis.to_json(outfolder + 'meta_data_df_plis.json')
        np.save(outfolder + 'linkage_cluster_order.npy', linkage_cluster_order)
        np.save(outfolder + 'linkage_cluster_order_og.npy', linkage_cluster_order_og)
        np.save(outfolder + 'arr_tsne.npy', arr_tsne)
        arr_df_sorted.to_feather(outfolder + 'arr_df_sorted.feather')
    
    return df_marker, arr_tsne, meta_data_df_plis, linkage_cluster_order, df_marker_log_and_std_col, df_plis, cluster_indices, df_marker_log_and_std, pos, tg, tgfs, change_indices, linkage_alg, dist_metric
    
def get_cells_per_dataset_per_cluster(meta_data_df, output_folder, write_to_file = False):
    '''returns a dataframe with number of cells per dataset per cluster'''
    cluster_dataset_df = pd.DataFrame(index=pd.unique(meta_data_df.loc['full_name',:]),columns=['n_cells_amy', 'n_cells_sd'])
    for c in pd.unique(meta_data_df.loc['full_name',:]):
        #print(f'Cluster {c} has {meta_data_df_plis_f[meta_data_df_plis_f["cluster_label"] == c].shape[0]} cells')
        cluster_subset = meta_data_df.loc[:,meta_data_df.loc['full_name',:] == c]
        cluster_dataset_df.loc[c, 'n_cells_amy'] = cluster_subset.loc[:,cluster_subset.loc['dataset',:]=='amy'].shape[1]
        cluster_dataset_df.loc[c, 'n_cells_sd'] = cluster_subset.loc[:,cluster_subset.loc['dataset',:]=='sd'].shape[1]
    if write_to_file:
        cluster_dataset_df.to_csv(output_folder + 'cells_per_dataset_per_cluster.csv')
        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_dataset_df.iloc[::-1].plot.barh(stacked=False, ax=ax, color=['#1f77b4', '#ff7f0e'])
        ax.set_xlabel('Number of Cells')
        ax.set_ylabel('Cluster')
        plt.tight_layout()
        plt.savefig(output_folder + 'cells_per_dataset_per_cluster_plot.png')
        plt.show()


    return cluster_dataset_df

def plot_cells_per_dataset_per_cluster(cluster_dataset_df, output_folder):
    '''plots number of cells per dataset per cluster'''
    plt.figure(figsize=(10, 6))
    cluster_dataset_df.plot(kind='bar', stacked=True)
    plt.title('Number of Cells per Dataset per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Cells')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_folder + 'cells_per_dataset_per_cluster.png')
    plt.show()