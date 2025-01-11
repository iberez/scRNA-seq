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
    #loop thru metadata isolated for specified class, get hexacodes
    hcodes = []
    #print (hexa_df)
    for x in list(metadata_df.loc['amy_markers',metadata_df.loc['amy_class']==cell_class]):
        hc = hexa_df.iloc[np.where(hexa_df.loc[:,'marker_name']==x)[0],4].values
        if hc == np.NaN:
            print ('NAN')
        #assign any blanks as NA
        if not hc:
            print ('NA')
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