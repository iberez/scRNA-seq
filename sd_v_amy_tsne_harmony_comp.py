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
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import PyPDF2
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfReader, PdfWriter
from matplotlib.backends.backend_pdf import PdfPages
from pdf2image import convert_from_path
import fitz  # PyMuPDF

import dimorph_processing as dp
import cell_comparison as cc

""" def cluster_transfer_majority_vote(arr_xy_sd,arr_xy_amy,distance_threshold,sd_metadata_df_subset,amy_metadata_df_subset):
    '''takes 2, 2D tsne x/y arrays, iterates through each point of arr1, finds all neighbors within distance_threshold. If more than
    10 neighbors, uses first 10. For each pts with at least 5 neighbors, uses index to count representation of cell types in metadata_df, 
    and performs majority voting to assign cell type and cell class.'''
    
    # Convert to shape (n_points, 2) for easier iteration
    arr_xy_sd_points = np.column_stack((arr_xy_sd[0], arr_xy_sd[1]))
    arr_xy_amy_points = np.column_stack((arr_xy_amy[0], arr_xy_amy[1]))
    
    # Dictionary to store closest points for each point in arr_xy_sd
    closest_points = {}
    
    # Iterate through each point in arr_xy_sd
    for idx, point1 in enumerate(arr_xy_sd_points):
        # Calculate distances to all points in arr_xy_amy
        distances = np.sqrt((arr_xy_amy_points[:, 0] - point1[0])**2 + (arr_xy_amy_points[:, 1] - point1[1])**2)
        
        # Find indices where distances are within the threshold
        within_threshold_indices = np.where(distances <= distance_threshold)[0]
        
        # If more than 10 points meet the threshold, keep only the first 10
        if len(within_threshold_indices) > 10:
            within_threshold_indices = within_threshold_indices[:10]
        
        # Store the points within threshold distance
        closest_points[idx] = arr_xy_amy_points[within_threshold_indices]
    
        #only consider points with at least 5 neighbors
        if len (within_threshold_indices)>=5:
            #print (dict(Counter(list(amy_metadata_df_subset_tmp.iloc[1,within_threshold_indices]))))
            #create dict counting instances of each unique cell type
            amy_marker_count_dict = dict(Counter(list(amy_metadata_df_subset.iloc[1,within_threshold_indices])))
            amy_marker_count_dict_sorted = dict(sorted(amy_marker_count_dict.items(), key=lambda item: item[1], reverse=True))
            #create dict counting intances of each unique class, use _c class corrected version
            amy_class_count_dict = dict(Counter(list(amy_metadata_df_subset.iloc[0,within_threshold_indices])))
            amy_class_count_dict_sorted = dict(sorted(amy_class_count_dict.items(), key=lambda item: item[1], reverse=True))
            
            #print (idx)
            #print (amy_marker_count_dict_sorted)
            #print (amy_class_count_dict_sorted)
        #print (list(amy_marker_count_dict_sorted.values())[0])
        
            #simplest case, all points from same amy cell type that's not blank, mark as amy cell type
            if len(list(amy_marker_count_dict_sorted.values()))==1 and len(list(amy_marker_count_dict_sorted.keys())[0])>0: 
                #print ('here')
                #print (len(list(amy_marker_count_dict_sorted.keys())[0]))
                #print (list(amy_marker_count_dict_sorted.keys())[0])
                sd_metadata_df_subset.iloc[3,idx] = list(amy_marker_count_dict_sorted.keys())[0]
                sd_metadata_df_subset.iloc[4,idx] = list(amy_class_count_dict_sorted.keys())[0]
                    
            #in case of more than one cell type,
            elif len(list(amy_marker_count_dict_sorted.values()))>1:
                # with equal represenation in top 2 cell types, mark as 'no-majority'
                if list(amy_marker_count_dict_sorted.values())[0] == list(amy_marker_count_dict_sorted.values())[1]:
                    #print ('no majority')
                    sd_metadata_df_subset.iloc[3,idx] = 'no majority ct'
                    sd_metadata_df_subset.iloc[4,idx] = 'NA - no majority ct'
                #case with multiple cell types but first is blank, just use original sd type (NN subclass)
                elif len(list(amy_marker_count_dict_sorted.keys())[0])==0:
                    sd_metadata_df_subset.iloc[3,idx] = sd_metadata_df_subset.iloc[1,idx]
                    sd_metadata_df_subset.iloc[4,idx] = list(amy_class_count_dict_sorted.keys())[0]
                #use majority voting, i.e. grab first key with highest representation
                else:
                    sd_metadata_df_subset.iloc[3,idx] = list(amy_marker_count_dict_sorted.keys())[0]
                    sd_metadata_df_subset.iloc[4,idx] = list(amy_class_count_dict_sorted.keys())[0]
                    
            #amy nononerual missing cell types (show as blank), so just use the original sd cell type (NN)
            elif len(list(amy_marker_count_dict_sorted.keys())[0])==0: 
                #print ('nn here')
                sd_metadata_df_subset.iloc[3,idx] = sd_metadata_df_subset.iloc[1,idx]
                sd_metadata_df_subset.iloc[4,idx] = list(amy_class_count_dict_sorted.keys())[0]
        
        #in case where no neighbors found, mark as 'no neighbors'
        else:
            #print ('here')
            sd_metadata_df_subset.iloc[3,idx] = 'no neighbors'
            sd_metadata_df_subset.iloc[4,idx] = 'NA - no neighbors'
        
    return sd_metadata_df_subset """


def plot_tsne_class_colors(sd_metadata_df_subset_w_amc, amy_sd_arr_tsne, meta_data_df_pca, outputfolder, outputname, savefig = False):
    #get sd cell type labels determined from majority voting using amy labels
    sd_labels = list(np.unique(sd_metadata_df_subset_w_amc.loc['amy_markers']))

    #get class labels
    sd_classes = list(np.unique(sd_metadata_df_subset_w_amc.loc['amy_class']))
    
    fig,ax = plt.subplots(figsize = (20,20))
    #label fontsize
    label_fs = 10
    #all pts
    x = amy_sd_arr_tsne[:,0]
    y = amy_sd_arr_tsne[:,1]
    #sd pts
    x_sd = x[meta_data_df_pca['dataset'] == 'sd']
    y_sd = y[meta_data_df_pca['dataset'] == 'sd']
    #ax.scatter(x_sd,y_sd,s=2,c = 'b', label = 'sd',alpha=.25)
    for sd_cls in sd_classes:
        #ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls))
        #manually set colors to match AMY paper
        if sd_cls == 'GABA':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = '#9ACD32')
        if sd_cls == 'VGLUT1':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = '#F97306')
        if sd_cls == 'VGLUT2':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = '#C20078')
        if sd_cls == 'Nonneuronal':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = '#0000FF')
        if sd_cls == 'NA - no majority ct':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = 'r')
        if sd_cls == 'NA - no neighbors':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = '#000000')
        #ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls))

    #combine into 2D array
    arr_xy_sd = np.vstack([x_sd,y_sd])
    for sd_ct in sd_labels:
        #print (arr_xy_sd.T[sd_metadata_df_subset.T['markers']==sd_ct])
        cluster_median = np.median(arr_xy_sd.T[sd_metadata_df_subset_w_amc.T['amy_markers']==sd_ct],axis = 0)
        ax.annotate(text = sd_ct, xy=cluster_median, fontsize=label_fs, color='black',
                        ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.add_patch(plt.Circle((-25, 25), 2*distance_threshold, color='r', alpha=0.5))
    ax.set_title()
    ax.legend(markerscale=2 )

    if savefig:
        plt.savefig(outputfolder + outputname + '.pdf')
    plt.show()

def plot_transferred_cell_labels(sd_metadata_df_subset_w_amc, amy_sd_arr_tsne, meta_data_df_pca, distance_threshold, outputfolder, outputname, savefig = False):
    #get sd cell type labels determined from majority voting using amy labels
    sd_labels = list(np.unique(sd_metadata_df_subset_w_amc.loc['amy_full_name']))

    #get class labels
    sd_classes = list(np.unique(sd_metadata_df_subset_w_amc.loc['amy_class']))
    
    fig,ax = plt.subplots(figsize = (20,20))
    #label fontsize
    label_fs = 10
    #all pts
    x = amy_sd_arr_tsne[:,0]
    y = amy_sd_arr_tsne[:,1]
    #sd pts
    x_sd = x[meta_data_df_pca['dataset'] == 'sd']
    y_sd = y[meta_data_df_pca['dataset'] == 'sd']
    #ax.scatter(x_sd,y_sd,s=2,c = 'b', label = 'sd',alpha=.25)
    for sd_cls in sd_classes:
        #ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls))
        #manually set colors to match AMY paper
        if sd_cls == 'GABA':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = '#9ACD32')
        if sd_cls == 'VGLUT1':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = '#F97306')
        if sd_cls == 'VGLUT2':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = '#C20078')
        if sd_cls == 'Nonneuronal':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = '#0000FF')
        if sd_cls == 'no majority ct':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = 'r')
        if sd_cls == 'no neighbors':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls), c = '#000000')
        #ax.scatter(x_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls],y_sd[sd_metadata_df_subset_w_amc.T['amy_class']==sd_cls], s=2, label = str(sd_cls))

    #combine into 2D array
    arr_xy_sd = np.vstack([x_sd,y_sd])
    for sd_ct in sd_labels:
        #print (arr_xy_sd.T[sd_metadata_df_subset.T['markers']==sd_ct])
        cluster_median = np.median(arr_xy_sd.T[sd_metadata_df_subset_w_amc.T['amy_full_name']==sd_ct],axis = 0)
        ax.annotate(text = sd_ct, xy=cluster_median, fontsize=label_fs, color='black',
                        ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))

    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.add_patch(plt.Circle((-25, 25), 2*distance_threshold, color='r', alpha=0.5))
    ax.set_title('sd with amy cell type names and class (cell type majority voting), r = ' + str(distance_threshold))
    ax.legend(markerscale=2 )

    ax.axis('off')

    if savefig:
        plt.savefig(outputfolder + outputname + '.pdf')
    plt.show()

def plot_transferred_cell_labels_hex(sd_metadata_df_subset_w_amc_hex,amy_sd_arr_tsne, meta_data_df_pca, distance_threshold, outputfolder, outputname, savefig = False,ref = False):
    '''plot transferred labels colored by hex codes'''
    #get sd cell type labels determined from majority voting using amy labels
    sd_labels = list(np.unique(sd_metadata_df_subset_w_amc_hex.loc['amy_full_name']))

    #get class labels
    sd_classes = list(np.unique(sd_metadata_df_subset_w_amc_hex.loc['amy_class']))

    #get amy transferred hexacode colors
    sd_hexacode = list(set(sd_metadata_df_subset_w_amc_hex.loc['hexacode']))

    fig,ax = plt.subplots(figsize = (20,20))
    #label fontsize
    label_fs = 10
    #all pts
    x = amy_sd_arr_tsne[:,0]
    y = amy_sd_arr_tsne[:,1]
    #sd pts
    x_sd = x[meta_data_df_pca['dataset'] == 'sd']
    y_sd = y[meta_data_df_pca['dataset'] == 'sd']
    #ax.scatter(x_sd,y_sd,s=2,c = 'b', label = 'sd',alpha=.25)
    for hc in sd_hexacode:
        #print (hc)
        if hc != 'NA':
            ax.scatter(x_sd[sd_metadata_df_subset_w_amc_hex.T['hexacode']==hc],y_sd[sd_metadata_df_subset_w_amc_hex.T['hexacode']==hc], s = 2, c = '#' + hc)

    #combine into 2D array
    arr_xy_sd = np.vstack([x_sd,y_sd])
    for sd_ct in sd_labels:
        #print (arr_xy_sd.T[sd_metadata_df_subset.T['markers']==sd_ct])
        cluster_median = np.median(arr_xy_sd.T[sd_metadata_df_subset_w_amc_hex.T['amy_full_name']==sd_ct],axis = 0)
        ax.annotate(text = sd_ct, xy=cluster_median, fontsize=label_fs, color='black',
                        ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_title('sd with amy cell type names and class (cell type majority voting), matched cluster colors, r = ' + str(distance_threshold))
    #ax.legend(markerscale=2 )
    if ref == True:
        x_amy = x[meta_data_df_pca['dataset'] == 'amy']
        y_amy = y[meta_data_df_pca['dataset'] == 'amy']
        ax.scatter(x_amy,y_amy,s=2,c = 'gray', label = 'amy')
    ax.axis('off')
    ax.set_box_aspect(1)
    if savefig:
        plt.savefig(outputfolder + outputname + '.pdf')
    plt.show()

def get_hcodes(folder, txt_file, cell_class, old_2_new_amy_dict, metadata_df):
    '''using txt file in folder with hexacodes for each cluster, updates name in text file with old_2_new_amy_dict. 
    then iterates through metadata_df clusters matching cluster names to get corresponding hexacode.
    returns as dataframe with subset of metadata_df columns.'''
    hexa_df = pd.read_csv(folder + txt_file, delimiter="\t")
    #use old_2_new_amy_dict to get new cluster names and insert in hexa_df file
    hexa_df.insert(2,'updated_name', '')
    hexa_df['updated_name'] = [old_2_new_amy_dict[x] for x in list(hexa_df['Name'])]
    #isolate marker names and insert as dedicated column
    m = [x.split('-',2)[2] for x in list(hexa_df['updated_name'])]
    hexa_df.insert(3,'marker_name',m)
    #print (hexa_df.head())
    #loop thru metadata isolated for specified class, get hexacodes
    hcodes = []
    #print (hexa_df)
    for x in list(metadata_df.loc['amy_markers',metadata_df.loc['amy_class']==cell_class]):
        hc = hexa_df.iloc[np.where(hexa_df.loc[:,'marker_name']==x)[0],4].values
        #print (hc)
        if hc == np.NaN:
            print ('NAN')
        #assign any blanks as NA
        if not hc:
            #print ('NA')
            hc = np.array(['NA'])
        hcodes.append(hc)
    hcodes = np.concatenate(hcodes,axis=0)
    hcodes = np.reshape(hcodes,(1,len(hcodes)))
    #get cols specific to cell_class
    cols = metadata_df.loc[:,metadata_df.loc['amy_class']==cell_class].columns
    #organize into dataframe using class specific columns
    hcodes_row = pd.DataFrame(hcodes,index = ['hexacode'], columns = cols)

    return hcodes_row

def cluster_transfers_2_class_and_fn(sd_metadata_df_subset_w_amc, amy_metadata_df_subset):
    '''takes output of cluster_transfer_majority_vote and uses amy_metadata to map cluster 
    transferred labels to class and full name'''    
    _, idx = np.unique(amy_metadata_df_subset.loc['markers'], return_index=True)
    am = np.array(amy_metadata_df_subset.loc['markers'][np.sort(idx)])

    _, idx = np.unique(amy_metadata_df_subset.loc['full_name'], return_index=True)
    afn = np.array(amy_metadata_df_subset.loc['full_name'][np.sort(idx)])

    am_2_afn_dict = dict(zip(am,afn))

    #add case handeling for no majority or no neighbors (set full name to the same thing)
    am_2_afn_dict['no majority ct'] = 'no majority ct'
    am_2_afn_dict['no neighbors'] = 'no neighbors'

    fn_transferred = []
    for x in sd_metadata_df_subset_w_amc.loc['amy_markers']: 
        fn_transferred.append(am_2_afn_dict[x])

    sd_metadata_df_subset_w_amc.loc['amy_full_name'] = fn_transferred

    #and finally, fill amy_class row using first part of amy_full_name
    sd_metadata_df_subset_w_amc.loc['amy_class'] = [x.split('-')[0] for x in sd_metadata_df_subset_w_amc.loc['amy_full_name']]

    return sd_metadata_df_subset_w_amc

def cluster_colormap(metadata,cmap_name):
    '''takes metadata df and returns a colormap for each unique cell type'''
    #get unique clusters
    clusters = np.unique(metadata.loc['full_name'])
    #evenly sample colors from colormap
    cmap_samples = mpl.colormaps[cmap_name].resampled(len(clusters))
    #create dict mapping clusters to colors
    cmap_dict = dict(zip(clusters, 
                         cmap_samples.colors))
    #get vector of colors for each cell in metadata
    colors = [cmap_dict[x] for x in np.array(metadata.loc['full_name'])]
    return colors


def tsne_plot(hm_arr, i, meta_data_df_pca, out_folder, savefig = False, write_to_file = False):
    '''takes harmonized pca array and metadata df, performs tSNE, and plots resulting 2D array plotted by dataset'''
    status_df = dp.intialize_status_df()
    perplexity,status_df = dp.get_perplexity(pca_arr = hm_arr, cutoff=500, plot_flag=1, status_df = status_df)

    final_hm_amy_sd_arr_tsne,status_df = dp.do_tsne(arr = hm_arr, 
                                n_components=2,
                                n_iter=1000,
                                learning_rate=hm_arr.shape[0]//12,
                                early_exaggeration=20,
                                init='pca', 
                                perplexity = perplexity,
                                metric='correlation',
                                status_df = status_df)
    
    fig,ax = plt.subplots()
    #all pts
    x = final_hm_amy_sd_arr_tsne[:,0]
    y = final_hm_amy_sd_arr_tsne[:,1]
    #ax.scatter(x,y,s=2)

    #amy pts
    x_amy = x[meta_data_df_pca['dataset'] == 'amy']
    y_amy = y[meta_data_df_pca['dataset'] == 'amy']
    ax.scatter(x_amy,y_amy,s=1,c = 'r', label = 'amy',alpha=.25)

    #sd pts
    x_sd = x[meta_data_df_pca['dataset'] == 'sd']
    y_sd = y[meta_data_df_pca['dataset'] == 'sd']
    ax.scatter(x_sd,y_sd,s=1,c = 'b', label = 'sd',alpha=0.25)


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('tSNE of harmonized amy and sd data, iter = ' + str((2*i)))
    ax.legend()
    ax.axis('off')
    ax.set_box_aspect(1)
    if savefig:
        plt.savefig(out_folder + str((2*i)) + '_iter_tsne_overlay_sd_on_amy_union.pdf')
    plt.show()
    if write_to_file:
        np.save(out_folder + 'data/' + str((2*i)) + '_iter_amy_sd_arr_tsne_union.npy', final_hm_amy_sd_arr_tsne)
        np.save(out_folder + 'data/' + str((2*i)) + '_iter_amy_sd_hm_arr.npy', hm_arr)

    return final_hm_amy_sd_arr_tsne

def run_harmony_convergence_test(input_folder, out_folder, iso_class_amy, arr_pca, meta_data_df_pca, amy_sd_hm_arr, amy_sd_arr_tsne, vars_use):
    x1 = hm.run_harmony(arr_pca,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x2 = hm.run_harmony(x1.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x3 = hm.run_harmony(x2.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x4 = hm.run_harmony(x3.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x5 = hm.run_harmony(x4.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x6 = hm.run_harmony(x5.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x7 = hm.run_harmony(x6.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x8 = hm.run_harmony(x7.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)

    x0_pca = arr_pca
    x1_pca = x1.Z_corr.T
    x2_pca = x2.Z_corr.T
    x3_pca = x3.Z_corr.T
    x4_pca = x4.Z_corr.T
    x5_pca = x5.Z_corr.T
    x6_pca = x6.Z_corr.T
    x7_pca = x7.Z_corr.T
    x8_pca = x8.Z_corr.T

    pca_matrices = [x0_pca,x1_pca,x2_pca,x3_pca,x4_pca,x5_pca,x6_pca,x7_pca,x8_pca]
    #compute distances between consecutive pca matrices
    avg_distances = []
    for k in range(len(pca_matrices)-1):
        #difference between consecutive pca matrices
        diff = pca_matrices[k+1] - pca_matrices[k]
        #euclidean distance for each cell
        #print (diff.shape)
        distances = np.linalg.norm(diff, axis=1)
        #print (distances.shape)
        avg_distance = np.mean(distances)
        avg_distances.append(avg_distance)

    fig,ax = plt.subplots()
    ax.plot(range(1, len(pca_matrices)), avg_distances, marker='o')
    ax.set_xticks(range(1, len(pca_matrices)),labels = [f'{2*i}' for i in range(1,len(pca_matrices))])
    #ax.set_xticklabels(range(1, len(pca_matrices)), labels = [f'{2*i}-{2*i+1}' for i in range(len(pca_matrices)-1)])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Distance')
    ax.set_title('Average Distance Between Consecutive PCA Matrices')
    plt.savefig(out_folder + 'avg_distances_harmony_convergence.pdf')
    plt.show()

    for i,m in enumerate(pca_matrices):
        tsne_plot(m,i,meta_data_df_pca,out_folder,savefig=True, write_to_file=True)

    files = [x for x in os.listdir(out_folder) if 'iter' in x]
    files = sorted(files, key=lambda x: int(x.split('_')[0].split('.')[0]))
    # Create a PdfFileWriter object to write the final PDF
    output_pdf = PdfWriter()

    # Create a 2x2 grid of the plots
    fig, axs = plt.subplots(2, 4, figsize=(20, 20))

    # Read and add each PDF file to the grid
    for ax, pdf_file in zip(axs.flat, files):
        pdf_path = os.path.join(out_folder, pdf_file)

        # Convert the first page of the PDF to an image
        images = convert_from_path(pdf_path, first_page=0, last_page=1)
        image = images[0]

        # Display the image in the grid
        #fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image, aspect = 'equal')
        ax.axis('off')
        ax.set_title(pdf_file)


    # Save the final PDF
    output_pdf_path = os.path.join(out_folder, 'hm_stiched.pdf')
    with PdfPages(output_pdf_path) as pdf:
        pdf.savefig(fig)

    print(f"Stitched PDF saved as {output_pdf_path}")

    
    return None

