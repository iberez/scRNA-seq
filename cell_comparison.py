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

today = str(date.today())

def process_amy_data_class(amy_df,amy_metadata_df,IEG_list, sex_gene_list, cell_class = None):
    '''gets columns containting speicfied class, processes and prepares for comparision with sd_data, return processed data and metadata'''
    if type(cell_class) == str and cell_class in ['GABA','VGLUT1','VGLUT2']:    
        #isolate columns of specified class
        print ('processing for single class: ', cell_class)
        _cols = [c for c,x in zip(amy_df.loc['celltype'].index,np.array(amy_df.loc['celltype'])) if cell_class in x]
        #print (_cols)
        amy_df_ = amy_df.loc[:,_cols]
        amy_df__expr = amy_df_.iloc[4:,:]
        #get associated metadata
        amy_metadata_df_ = amy_metadata_df.loc[:,_cols]
        #add cluster label row to metadata
        amy__cluster_labels = np.array(amy_metadata_df_.loc['celltype'].apply(lambda x: int(re.search(r'-(\d+)-', x).group(1))))
        amy_metadata_df_.loc['cluster_label'] = amy__cluster_labels
    else:
        print ('non neuronal')
        #_cols = [c for c,x in zip(amy_df.loc['celltype'].index,np.array(amy_df.loc['celltype'])) if cell_class in x]
        _cols = amy_df.loc['celltype'].index
        #print (_cols)
        amy_df_ = amy_df.loc[:,_cols]
        amy_df__expr = amy_df_.iloc[4:,:]
        #get associated metadata
        amy_metadata_df_ = amy_metadata_df.loc[:,_cols]
        #add cluster label row to metadata

    #remove duplicate genes
    print ('removing # duplicate gene rows: ', amy_df__expr[amy_df__expr.index.duplicated()].shape[0])
    amy_df__expr = amy_df__expr[~amy_df__expr.index.duplicated(keep='first')]
    #remove IEG genes
    amy_df__expr = dp.gene_remover(IEG_list, amy_df__expr)
    #remove sex genes
    amy_df__expr = dp.gene_remover(sex_gene_list, amy_df__expr)
    #ensure all values are ints
    amy_df__expr = amy_df__expr.astype('int')
    #process using standard workflow/functions from dimorph_processing.py
    amy_df__bool = amy_df__expr.mask(amy_df__expr>0, other = 1)
    status_df = dp.intialize_status_df()
    amy_df__expr_ge, amy_df__bool, amy_metadata_df_, status_df = dp.gene_exclusion(num_cell_lwr_bound=10,
                                                      percent_cell_upper_bound=50,
                                                     df_bool=amy_df__bool,
                                                     df = amy_df__expr,
                                                     meta_data_df = amy_metadata_df_,
                                                     status_df = status_df)
    amy__cv_df = dp.analyze_cv(df = amy_df__expr_ge,
                      norm_scale_factor=20000,
                      num_top_genes=30,
                      plot_flag=1,
                     use_huber = True)
    amy__gene_index, amy_df__expr_ge_cv, status_df = dp.get_top_cv_genes(df = amy_df__expr_ge, cv_df=amy__cv_df, plot_flag=1, status_df=status_df)
    amy_arr__expr_ge_cv_ls,status_df = dp.log_and_standerdize_df(amy_df__expr_ge_cv,status_df)
    amy_df__expr_ge_cv_ls = pd.DataFrame(data=amy_arr__expr_ge_cv_ls.T, index=amy_df__expr_ge_cv.index, columns=amy_df__expr_ge_cv.columns)
    
    return amy_df__expr_ge_cv_ls, amy_metadata_df_

def get_df_gene_intersection(sd_df,amy_df, IEG_list):
    '''gets the index (gene) intersection of df1 and df2. assumes df2 processed using process_amy_class()'''
    #make sure IEGs are removed from sd_df
    sd_df = dp.gene_remover(IEG_list, sd_df)
    #transpose to make genes the index
    sd_df = sd_df.T
    #get intersecting gene index
    intersected_gene_ind = amy_df.index.intersection(sd_df.index)
    print (intersected_gene_ind)
    amy_df_i = amy_df.reindex(index = intersected_gene_ind)
    sd_df_i = sd_df.reindex(index = intersected_gene_ind)
    return amy_df_i,sd_df_i

def plot_correlation(sd_avgs,amy_avgs):
    '''takes in sd_avgs, amy_avgs, and lco (linkage cluster order) for sd_avgs, computes correlation matrix and plots interactive heatmap '''
    #sd_avgs_lco = sd_avgs.reindex(columns=lco)
    sd_avgs_lco = sd_avgs
    # Initialize an empty dataframe to hold the correlation coefficients
    corr_matrix_manual_alt_lco = pd.DataFrame(index=amy_avgs.columns, columns=sd_avgs_lco.columns)
    #corr_matrix_manual_alt_lco = pd.DataFrame(index=amy_avgs.columns, columns=np.arange(1, len(sd_avgs_lco.columns)+1))
    # Compute the correlation coefficients (note we could also have just taken transpose of corr_matrix_manual)
    for col1 in amy_avgs.columns:
        for col2 in sd_avgs_lco.columns:
            corr_matrix_manual_alt_lco.loc[col1, col2] = amy_avgs[col1].corr(sd_avgs_lco[col2])

    heatmap_argmax_df_alt_lco = pd.DataFrame(columns = ['argmax','max'], index=corr_matrix_manual_alt_lco.index)
    for i in corr_matrix_manual_alt_lco.index:
        #print (i)
        heatmap_argmax_df_alt_lco.loc[i,:] = (np.argmax(corr_matrix_manual_alt_lco.loc[i,:]),np.max(corr_matrix_manual_alt_lco.loc[i,:]))
    heatmap_argmax_df_alt_lco.sort_values(by = 'argmax')
    corr_matrix_manual_alt_lco_sorted = corr_matrix_manual_alt_lco.reindex(index = heatmap_argmax_df_alt_lco.sort_values(by = 'argmax').index)
    #make columns seq
    #corr_matrix_manual_alt_lco_sorted.columns = np.arange(1, len(sd_avgs_lco.columns)+1)
    pos_ylabel = [(pos,ylabel) for pos,ylabel in zip(np.arange(len(corr_matrix_manual_alt_lco_sorted.index)),corr_matrix_manual_alt_lco_sorted.index)]
    heatmap2 = corr_matrix_manual_alt_lco_sorted.reset_index().drop(columns='index').hvplot.heatmap(title='Arg Max sorted Amy_avgs (y) correlated with Sd_avgs_linkage_sorted (x)',  
                                                                                                    yticks = corr_matrix_manual_alt_lco_sorted.index,
                                                                                                    cmap='viridis', width=900, height=900, colorbar=True)
    heatmap2.opts(xlabel='Cluster IDs (Sd_avgs)',ylabel='Cluster IDs (Amy_avgs)',yticks = pos_ylabel )

    return heatmap2, heatmap_argmax_df_alt_lco, corr_matrix_manual_alt_lco, corr_matrix_manual_alt_lco_sorted

def amy_gene_spell_checker(amy_df,amy_metadata_df):
    '''takes markers listed in metadata df 'markers' row, compares with index of amy_df, returns genes not found in amy_df index'''
    #get list of lists of markers, then list of unique marker list, preserving order
    all_m = [amy_metadata_df.loc['markers'][x] for x in range((amy_metadata_df.loc['markers'].shape[0]))]

    all_m_u = []
    for x in all_m:
        if x not in all_m_u:
            all_m_u.append(x)

            #sanity check of markers from meta data in index of expression matrix
    error_genes = []
    for x in all_m_u:
        for g in x:
            if g in amy_df.index:
                pass
            else:
                print ('gene, ',g ,'not found in index')
                error_genes.append(g)
    return error_genes, all_m_u

def correct_error_genes(error_genes,correct_gene_names, all_m_u):
    for eg in error_genes:
        for cg in correct_gene_names:
            for i,x in enumerate(all_m_u):
                for j,g in enumerate(x):
                    if g == eg:
                        print (g)
                        #print (all_m_u[i][j])
                        #if first and last letter of eg matches cg, replace with cg
                        if cg[:1] == g[:1] and cg[-1:] == g[-1:]:
                            #print (cg)
                            print (all_m_u[i][j])
                            all_m_u[i][j] = cg
    return all_m_u

def gene_explorer(gene,dataset_name,df,metadata_df,output_folder= None,markers = False, savefig = False):
    '''Takes in a gene and dataset_name as strings, dataframe and corresponding metadata, returns
    jitter plot of all cells categorized into into cluster id. option to include markers if known and included as row in metadata.'''
    #extract gene expr data
    jitter_test = df.loc[gene,:].T
    #get x axis labels, combining cluster label + marker if known
    if markers:
        jitter_x = metadata_df.loc['cluster_label'].astype(str).T + '-' + metadata_df.loc['markers'].astype(str).T
    else:
        jitter_x = metadata_df.loc['cluster_label'].astype(str).T
    # remove unwanted characters in x axis labels
    jitter_x_fmt = pd.Series([re.sub("[',']","",x) for x in jitter_x])
    #match up index
    jitter_x_fmt.index = jitter_test.index
    
    jitter_test_df = pd.concat([jitter_test,jitter_x_fmt,metadata_df.loc['cluster_label'].T],axis=1)
    if markers:
        jitter_test_df = jitter_test_df.rename({0:'Cluster_ID-Markers'}, axis = 'columns')
    else:
        jitter_test_df = jitter_test_df.rename({0:'Cluster_ID'}, axis = 'columns')
    jitter_test_df_sorted = jitter_test_df.sort_values(by = 'cluster_label')
    
    fig,ax = plt.subplots(figsize = (15,10))
    sns.stripplot(x=jitter_test_df_sorted.columns[1], y=jitter_test_df_sorted.columns[0], data=jitter_test_df_sorted, 
                  jitter = 0.4, s = 2)
    #plt.xticks(rotation = 45)
    if markers:
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha="right")

    cell_class = np.unique(metadata_df.loc['cell_class'])[0]
    plt.title(cell_class + '-' +  gene)
    plt.ylabel('log/standerdized expr')
    plt.tight_layout()
    if savefig:    
        plt.savefig(output_folder + dataset_name + '_jitter_plot_' + cell_class +'_' + gene +'.png')
    plt.show()
    
    return ax.get_xticklabels()

def get_updated_cl_mg_dict(cl_mg_dict, df_prelinkage_ls):
    '''#create updated dict with only genes found in <cell_class>_df_prelinkage_ls
    #since <cell_class>_df_prelinkage_ls is the gene intersected version of the _orig'''
    cl_mg_dict_updated = deepcopy(cl_mg_dict)
    for k,v in cl_mg_dict_updated.items():
        for g in v:
            #print (g)
            if str(g) not in df_prelinkage_ls.index:
                v.remove(g)
        cl_mg_dict_updated[k] = v
    return cl_mg_dict_updated

def get_max_markers(cl_mg_dict,sd_avgs,cluster_id):
    #UNUSED FUNCTION - sort markers for a given cluster id by max expression, return sorted gene list 
    g_lst = []
    g_max_lst = []
    for g in cl_mg_dict[str(cluster_id)]:
        if g in sd_avgs.index:
            print (g)
            g_lst.append(g)
            g_max = np.max(sd_avgs.loc[g])
            g_max_lst.append(g_max)
    #print (g_max_lst)
    cluster_maxs = dict(zip(g_lst, g_max_lst))
    print (cluster_maxs)
    sorted_cluster_maxs_genes = sorted(cluster_maxs, key=cluster_maxs.get, reverse=True)
    return sorted_cluster_maxs_genes

def create_mg_cl_dict_final(cl_mg_dict,sd_shared_cl_mg_dict):
    '''creates final cl_mg_dict, using shared_cl_mg_dict keys if present, otherwise just use first two from cl_mg_dict'''
    mg_cl_dict_final = {}
    #make keys of shared dict integers to match those in cl_mg_dict...
    sd_shared_cl_mg_dict_tmp = {int(k): v for k, v in sd_shared_cl_mg_dict.items()}
    for k,v in cl_mg_dict.items():
        #print (k) #use shared keys
        if k in sd_shared_cl_mg_dict_tmp.keys():
            mg_cl_dict_final.update({int(k):sd_shared_cl_mg_dict_tmp[k]})
        else:
            mg_cl_dict_final.update({int(k):cl_mg_dict[k][:2]})
    
    
    return mg_cl_dict_final

def filter_mg_cl_dict_final_sorted(mg_cl_dict_final_sorted, clusters_to_drop):
    mg_cl_dict_final_sorted_filtered = deepcopy(mg_cl_dict_final_sorted)
    for k,v in mg_cl_dict_final_sorted.items():
        if k in clusters_to_drop:
            del (mg_cl_dict_final_sorted_filtered[k])
    return mg_cl_dict_final_sorted_filtered

def build_corr_table_shared_top(heatmap_argmax_df_alt_lco,corr_matrix_manual_alt_lco, all_m_u, amy_metadata_df, cl_mg_dict_updated):
    '''identfy shared markers between amy and sd correlated cluster, then get the top sd marker, store in '''
    connector_df_alt_lco = heatmap_argmax_df_alt_lco.sort_values(by = 'argmax').copy()
    connector_df_alt_lco.insert(2, 'corr_cluster', corr_matrix_manual_alt_lco.columns[np.array(connector_df_alt_lco['argmax']).astype('int')])
    connector_df_alt_lco = connector_df_alt_lco.reset_index()
    cluster_2_markers_dict = dict(zip(pd.unique(amy_metadata_df.loc['cluster_label']),all_m_u))
    connector_df_alt_lco_marker = connector_df_alt_lco.copy()
    connector_df_alt_lco_marker.insert(1, 'amy_marker',[cluster_2_markers_dict[v] for v in np.array(connector_df_alt_lco.loc[:,'index'])])
    connector_df_alt_lco_marker_shared = connector_df_alt_lco_marker.copy() 
    connector_df_alt_lco_marker_shared.insert(5, 'sd_shared_marker', '')
    #print (connector_df_alt_lco_marker_shared)
    #1st order to get sd makers - get shared overlaps with amy data
    for i in np.array(connector_df_alt_lco_marker.loc[:,'index']):
        corr_cluster = np.array(connector_df_alt_lco_marker.loc[connector_df_alt_lco.loc[:,'index']==i,'corr_cluster'])
        #print (corr_cluster[0])
        if corr_cluster[0] in cl_mg_dict_updated.keys():
            #print ('matching key!')
            mgs_og = cl_mg_dict_updated[corr_cluster[0]]
        #hack since gaba_cl_mg_dict missing a few keys. need to debug compute_marker_genes function.
        #else:
            #mgs_og = []
        #print ('og')
        #print (mgs_og)
        mgs_amy = (np.array(connector_df_alt_lco_marker.loc[connector_df_alt_lco.loc[:,'index']==i,'amy_marker'])[0])
        #print ('amy')
        #print (mgs_amy)
        shared_genes = list(set(mgs_og).intersection(mgs_amy))
        #print ('shared genes')
        #print (shared_genes)
        #add shared genes to sd_shared_marker
        connector_df_alt_lco_marker_shared.loc[connector_df_alt_lco_marker.loc[:,'index']==i,'sd_shared_marker'] = connector_df_alt_lco_marker_shared.loc[connector_df_alt_lco_marker.loc[:,'index']==i,'sd_shared_marker'].apply(lambda x: shared_genes)
        #print (connector_df_alt_lco_marker_shared)
        #if len(shared_genes)>1:
            #connector_df_alt_lco_marker_shared.loc[connector_df_alt_lco_marker.loc[:,'index']==i,'sd_shared_marker'] = shared_genes
    connector_df_alt_lco_marker_shared_top = connector_df_alt_lco_marker_shared.copy()
    connector_df_alt_lco_marker_shared_top.insert(6, 'sd_shared_top_marker', '')
    
    #2nd pass - for all corr cluster with 1 sd_shared marker, get top marker from sd data cluster/marker dict. if already 2 markers, skip
    current_cluster = [] #track cluster number

    for i in np.array(connector_df_alt_lco_marker.loc[:,'index']):
        #get sd_shared_marker list
        l = list(connector_df_alt_lco_marker_shared.loc[connector_df_alt_lco.loc[:,'index']==i,'sd_shared_marker'])[0]
        corr_cluster = np.array(connector_df_alt_lco_marker.loc[connector_df_alt_lco.loc[:,'index']==i,'corr_cluster'])
        #print (corr_cluster)
        if len(l)==1 and len(cl_mg_dict_updated[corr_cluster[0]])>=2:
            #use next most specific gene from the dict
            l_cp = l.copy()
            ng_l = []
            for x in cl_mg_dict_updated[corr_cluster[0]]:
                if x not in l_cp:
                    ng_l.append(x)
            #print (ng)
            l_cp.append(ng_l[0])
            connector_df_alt_lco_marker_shared_top.loc[connector_df_alt_lco_marker.loc[:,'index']==i,'sd_shared_top_marker'] = connector_df_alt_lco_marker_shared_top.loc[connector_df_alt_lco_marker.loc[:,'index']==i,'sd_shared_top_marker'].apply(lambda x: l_cp)  
        
        if len(l) == 2:
            #copy over the two overlapping genes
            l_cp = l.copy()
            connector_df_alt_lco_marker_shared_top.loc[connector_df_alt_lco_marker.loc[:,'index']==i,'sd_shared_top_marker'] = connector_df_alt_lco_marker_shared_top.loc[connector_df_alt_lco_marker.loc[:,'index']==i,'sd_shared_top_marker'].apply(lambda x: l_cp)
    #print (connector_df_alt_lco_marker_shared_top)
    #build sd_shared_dict
    sd_shared_cl_mg_dict = {}
    for i in np.array(connector_df_alt_lco_marker.loc[:,'index']):
        corr_cluster = np.array(connector_df_alt_lco_marker.loc[connector_df_alt_lco.loc[:,'index']==i,'corr_cluster'])
        
        
        l1 = list(connector_df_alt_lco_marker_shared_top.loc[connector_df_alt_lco.loc[:,'index']==i,'sd_shared_marker'])[0]
        l2 = list(connector_df_alt_lco_marker_shared_top.loc[connector_df_alt_lco.loc[:,'index']==i,'sd_shared_top_marker'])[0]
        
        if l1 and l2:
            if len(l2)>len(l1):
                sd_shared_cl_mg_dict.update({str(corr_cluster[0]):l2})
            if len(l2)==len(l1):
                #prioritize overlapped markers
                sd_shared_cl_mg_dict.update({str(corr_cluster[0]):l1})

    
    return sd_shared_cl_mg_dict, connector_df_alt_lco_marker_shared_top

def generate_amy_labels_df(connector_df_alt_lco_marker_shared_top, corr_matrix_manual_alt_lco):
    amy_labels = (connector_df_alt_lco_marker_shared_top.loc[:,['index','amy_marker']])
    amy_labels = amy_labels.sort_values(by = 'index')
    amy_labels = [f"{row['index']} {'-'.join(row['amy_marker'])}" for _, row in amy_labels.iterrows()]
    amy_labels_df = pd.DataFrame(data = amy_labels, index = corr_matrix_manual_alt_lco.index)
    return amy_labels_df, amy_labels

def generate_sd_labels_df(mg_cl_dict_final):
    
    
    # sd_labels_df = pd.DataFrame(data = corr_matrix_manual_alt_lco.columns, columns = ['lco_index'])
    # #sd_labels_df = pd.DataFrame(data = np.arange(1, len(corr_matrix_manual_alt_lco.columns)+1), columns = ['lco_index'])
    # mg_cl_dict_final_seq = {new_key: mg_cl_dict_final[old_key] for new_key, old_key in enumerate(mg_cl_dict_final.keys(), start=1)}
    # mg_cl_conv_dict = {old_key:new_key  for new_key, old_key in enumerate(mg_cl_dict_final.keys(), start=1)}
    # sd_labels_df.insert(1, column= 'sd_label', value='')
    # sd_labels_df['sd_label'] = sd_labels_df['sd_label'].astype(object)
    # for i,v in enumerate(sd_labels_df['lco_index']):
    #     if v in mg_cl_dict_final_seq.keys():
    #         #print (gaba_mg_cl_dict_final[i])
    #         l = mg_cl_dict_final_seq[v]
    #         sd_labels_df.at[i,'sd_label'] = l
    # sd_labels_df['sd_label_complete'] = sd_labels_df.apply(lambda row: f"{row['lco_index']} {'-'.join(row['sd_label'])}", axis=1)
    
    # mg_cl_dict_final_sorted = {key:value for key, value in sorted(mg_cl_dict_final_seq.items(), key=lambda item: int(item[0]))}
    # sd_labels = [f"{key} {'-'.join(value)}" for key, value in mg_cl_dict_final_sorted.items() if value]
    sd_labels_df = pd.DataFrame(list(mg_cl_dict_final.items()), columns = ['lco_index','sd_label'])
    sd_labels_df['sd_label_complete'] = sd_labels_df.apply(lambda row: f"{row['lco_index']} {'-'.join(row['sd_label'])}", axis=1)
    
    return sd_labels_df 

def plot_connector_plot_with_labels(connector_df_alt_lco_marker_shared_top, mg_cl_dict_final_sorted, sd_labels,amy_labels, folder, cell_class,savefig=False):
    fig,ax1 = plt.subplots(figsize = (20,20))
    ax1.set_title('Amy Cluster-Markers Correlated to Sd Cluster-Markers: ' + str(cell_class))

    ax2 = ax1.twinx()

    x1 = np.zeros((1,len(connector_df_alt_lco_marker_shared_top['index'])))[0]
    y1 = np.array(connector_df_alt_lco_marker_shared_top['index'])

    x2 = np.ones((1,len(connector_df_alt_lco_marker_shared_top['corr_cluster'])))[0]
    y2 = np.array(connector_df_alt_lco_marker_shared_top['corr_cluster'])

    ax1.scatter(x = x1, 
                y = y1, color = 'b', label = 'amy_clusters')


    ax1.set_yticks(ticks = np.sort(np.array(connector_df_alt_lco_marker_shared_top['index'])),labels = amy_labels)
    #plt.xticks([])
    ax1.set_xticks([])
    ax1.set_ylabel('Amy Cluster IDs', color = 'b')
    ax1.tick_params(labelcolor = 'b')



    ax1.scatter(x = x2, 
                y = y2, color = 'r', label = 'sd_clusters')

    for i in range(len(connector_df_alt_lco_marker_shared_top)):
        plt.plot([x1[i], x2[i]], [y1[i], y2[i]], '-') 


    #plt.ylabel('cluster_id')
    #plt.yticks(np.arange(1,57))

    sd_tics = [k for k,v in mg_cl_dict_final_sorted.items() if v]
    #plt.tight_layout()
    ax2.set_yticks(ticks = sd_tics, labels = sd_labels)

    ax2.set_ylabel('SD Cluster IDs', color = 'r')
    ax2.tick_params(labelcolor = 'r')

    #ax1.legend()
    #plt.autoscale()
    if savefig:
        plt.savefig(folder+cell_class+'_connector_plot_amy_sd_lco_w_marker_labels' + today + '.png')
        plt.show()

def plot_correlation_w_labels(corr_matrix_manual_alt_lco_sorted, sd_labels_df, amy_labels_df, folder,cell_class, savefig = False):
    #hack to get sd labels on x axis - set columns of corr_matrix to the labels... using xticks = pos_xlabel, same method as y ticks produces blank plot..
    corr_matrix_manual_alt_lco_sorted_col_labeled = corr_matrix_manual_alt_lco_sorted.copy()
    corr_matrix_manual_alt_lco_sorted_col_labeled.columns = sd_labels_df['sd_label_complete']
    pos_ylabel = [(pos,ylabel) for pos,ylabel in zip(np.arange(len(corr_matrix_manual_alt_lco_sorted.index)),np.array(amy_labels_df.reindex(index = corr_matrix_manual_alt_lco_sorted.index)[0]))]
    #ylabel_w_idx = [str(i)+ ':'+ s for i,s in pos_ylabel]
    #pos_ylabel_w_idx = [(pos,ylabel) for pos,ylabel in zip(np.arange(len(corr_matrix_manual_alt_lco_sorted.index)),ylabel_w_idx)]
    heatmap2 = corr_matrix_manual_alt_lco_sorted_col_labeled.reset_index().drop(columns='index').hvplot.heatmap(title=cell_class + ' ' + 'Arg Max sorted Amy_avgs (y) correlated with Sd_avgs_linkage_sorted (x)',  
                                                                                                    yticks = corr_matrix_manual_alt_lco_sorted.index,
                                                                                                    rot = 90,
                                                                                                    #xticks = pos_xlabel,
                                                                                                    cmap='viridis', width=900, height=900, colorbar=True)
    heatmap2.opts(xlabel='Cluster IDs + Gene Markers (Sd_avgs)',ylabel='Cluster IDs + Gene Markers (Amy_avgs)', yticks = pos_ylabel)

    #heatmap2.opts(xticks = pos_xlabel)

    # Display the plot
    hvplot.show(heatmap2)

    # Ensure output is displayed inline
    hv.output(heatmap2, backend='bokeh')
    today = str(date.today())
    if savefig:
        panel_object = pn.pane.HoloViews(heatmap2)
        pn.pane.HoloViews(heatmap2).save(folder+cell_class+'_corr_plot_amy_sd_lco_w_labels' + today, embed=True, resources=INLINE)