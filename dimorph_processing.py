# functions used in dimorph_processing_nb.ipynb
# to process sexual dimorphism samples 
# after initial construction by sample_reader

# Isaac Berez
# 24.01.23 

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

def create_meta_data_df(meta_data, df):
    ''' creates a new meta data df with dim (# meta data features, e.g. serial number, x # cells from big data dataframe)'''
    meta_data_df = pd.DataFrame(index = meta_data.index, columns = df.columns)
    for i,v in enumerate(meta_data.keys()):
        if len(meta_data_df.columns[meta_data_df.columns.str.contains(meta_data.keys()[i])])>0:
            meta_data_df.loc[meta_data_df.index[:],
                            meta_data_df.columns[meta_data_df.columns.str.contains(meta_data.keys()[i])]] = meta_data.loc[meta_data.index[:], meta_data.columns[i]]
    return meta_data_df

def intialize_status_df():
    '''create a status dataframe for tracking completion of each function/processing step'''
    steps = ['cell_exclusion (l1)',
         'gene_exclusion (l1)',
         'get_top_cv_genes',
         'log_and_standerdize',
         'analyze_pca',
         'get_perplexity',
         'do_tsne',
         'compute_eps',
         'do_dbscan']
    status_df = pd.DataFrame(columns =['completion_status'], 
                        index = steps)
    
    return status_df

def load_data(metadata_file, bigdata_file):
    ''' reads in metadata json, big data (gene expression) feather file, returns dataframe versions for each, boolean version dataframe of the gene expression matrix,
    and status dataframe'''
    meta_data = pd.read_json(metadata_file)
    df = pd.read_feather(bigdata_file)
    df.set_index('gene', inplace=True)
    meta_data_df = create_meta_data_df(meta_data=meta_data, df = df)
    meta_data_df = meta_data_df.fillna('')
    df = df.loc[:,meta_data_df.loc['Strain'].str.contains('Cntnp')==False]
    meta_data_df = meta_data_df.loc[:,meta_data_df.loc['Strain'].str.contains('Cntnp')==False]
    # create boolean version of the dataframe, where any expression >0 = 1,
    df_bool = df.mask(df>0, other = 1)
    status_df = intialize_status_df()
    
    return meta_data_df,df,df_bool,status_df

def find_knee(x,y):
    '''given set of points x,y, find closest point to origin. Returns sorted df by ascending dist'''
    knee_df = pd.DataFrame({'x':x, 
                            'y':y})
    x = [x for x in knee_df.loc[:,'x']]
    y = [y for y in knee_df.loc[:,'y']]
    d = [np.sqrt(x**2 + y**2) for x,y in zip(x,y)]
    knee_df.insert(2, 'dist',d)
    #sort by smallest distance, and extract corresponding index
    knee_df.sort_values(by='dist', inplace=True)
    
    return knee_df

def cell_exclusion(threshold_m,threshold_g, meta_data_df, df_bool, df, status_df):
    '''computes total molecules per cell and total genes per cell, 
    excludes cells below specified threshold_m (molecules) and threshold_g (genes), 
    and returns cell filtered gene expression dataframe and corresponding boolean representation'''
    total_molecules_per_cell = df.sum(axis=0)
    total_molecules_per_cell = np.reshape(np.array(total_molecules_per_cell),(1,len(total_molecules_per_cell)))
    # create boolean version of the dataframe, where any expression >0 = 1,
    # used to calculate total genes/cell
    #df_bool = df.mask(df>0, other = 1)
    total_genes_per_cell = df_bool.sum(axis=0)
    total_genes_per_cell = np.reshape(np.array(total_genes_per_cell),(1,len(total_genes_per_cell)))
    # use masking to create boolean vector to keep columns (cells) == True
    mol_cell_mask = (total_molecules_per_cell>threshold_m)[0]
    genes_cell_mask = (total_genes_per_cell>threshold_g)[0]
    mol_AND_gene_cell_mask = np.logical_and(mol_cell_mask,genes_cell_mask)
    df_updated = df.loc[:,mol_AND_gene_cell_mask]
    #update gene boolean mask with l2 filtered cells
    df_bool_updated = df_bool.loc[:,df_updated.columns]
    #update meta_data_df
    meta_data_df_updated = meta_data_df.loc[:,df_updated.columns]
    print (f'Total cells reduced from {df.shape[1]} to {df_updated.shape[1]}')
    status_df.loc['cell_exclusion (l1)',:] = True
    
    return df_updated, df_bool_updated, meta_data_df_updated, status_df

def gene_exclusion(num_cell_lwr_bound, percent_cell_upper_bound, df, df_bool, meta_data_df, status_df):
    '''computes sum of each bool gene row and keeps only genes expressed in
    lwr_cell_bound < gene < upper_cell_bound. Returns gene filtered gene expression dataframe.'''
    #sum each row of l2 filtered boolean gene mask, get vector of dim (len(row genes)x1)
    gene_sum =  np.array(df_bool.sum(axis=1))
    gene_sum = np.reshape(gene_sum,(gene_sum.shape[0],1))

    df_updated = pd.DataFrame(columns = df.columns)
    gene_exclusion_lwr_bound = num_cell_lwr_bound
    gene_exclusion_upper_bound = 0.01*percent_cell_upper_bound*df.shape[1]
    genes_to_keep_indices = []
    for i,v in enumerate(gene_sum):
        if gene_exclusion_lwr_bound < v < gene_exclusion_upper_bound:
            genes_to_keep_indices.append(i)
    #use indices from genes_to_keep_indices to remaining genes after gene exlusion
    df_updated = df.iloc[genes_to_keep_indices,:]
    df_bool_updated = df_bool.loc[df_updated.index,:]
    #update meta_data_df
    meta_data_df_updated = meta_data_df.loc[:,df_updated.columns]
    print (f'Total genes reduced from {df.shape[0]} to {df_updated.shape[0]}')
    status_df.loc['gene_exclusion (l1)',:] = True
    
    return df_updated, df_bool_updated, meta_data_df_updated, status_df

def avg_bool_gene_expression_by_sex(df_bool, meta_data_df, num_top_genes, plot_flag = 0):
    '''computes the mean of each gene (row) from bool expressed genes, isolated by sex, 
    and outputs these values and the delta of each mean (male - female) as a dataframe. 
    If plot flag = 1,  plots a scatter plot using these values and labeling top genes given by num_top_genes.'''
    avg_bool_expr_m = df_bool.loc[:,meta_data_df.loc['Sex',:] == 'M'].mean(axis=1)
    avg_bool_expr_f = df_bool.loc[:,meta_data_df.loc['Sex',:] == 'F'].mean(axis=1)
    #print (f"avg bool expr m dim: {avg_bool_expr_m.shape}")
    #print (f"avg bool expr f dim: {avg_bool_expr_f.shape}")
    print (f"num m cells: {df_bool.loc[:,meta_data_df.loc['Sex',:] == 'M'].shape[1]} num f cells: {df_bool.loc[:,meta_data_df.loc['Sex',:] == 'F'].shape[1]}")
    #construct avg gene bool dataframe for male/female
    avg_bool_mf_df = pd.DataFrame({'m': avg_bool_expr_m, 'f': avg_bool_expr_f})
    #compute delta for each average and add to dataframe
    delta = avg_bool_expr_m - avg_bool_expr_f
    avg_bool_mf_df.insert(2,'delta_m-f', delta)
    #sort by low to high delta
    avg_bool_mf_df_sorted = avg_bool_mf_df.sort_values(by = 'delta_m-f')
    #get num_top_genes for female and male values and corresponding gene indices
    f_pos = np.array(avg_bool_mf_df_sorted.iloc[0:num_top_genes,0:2])
    f_genes = avg_bool_mf_df_sorted.iloc[:num_top_genes].index
    m_pos = np.array(avg_bool_mf_df_sorted.iloc[-num_top_genes:,0:2])
    m_genes = avg_bool_mf_df_sorted.iloc[-num_top_genes:].index

    if plot_flag == 1:
        fig,ax = plt.subplots(figsize = (7.5,7.5))
        plt.scatter(avg_bool_expr_m,avg_bool_expr_f, s = 1)
        plt.xlabel('avg bool expression - male')
        plt.ylabel('avg bool expression - female')
        plt.title('Male vs. Female of Avg Bool Gene Expression')
        # define offset amount for labeled genes on scatterplot
        offset = 0.01
        for i in range(len(f_pos)):
            plt.text(f_pos[i][0]+offset,f_pos[i][1]+offset,f_genes[i])
            plt.text(m_pos[i][0]+offset,m_pos[i][1]+offset,m_genes[i])
        plt.show()

    return avg_bool_mf_df_sorted

def analyze_cv(df,norm_scale_factor,num_top_genes,plot_flag, use_huber = False):
    '''performs normalization by summing all genes expressed per cell and diving each by sum, then scaling by norm_scale_factor. 
    Then computes log2cv and log2mu, fits using linear and huber regression, and plots result if plot flag = 1. 
    Labels the most highly expressed genes according to num_top_genes. Returns sorted log2mu and log2cv dataframe.'''
    #sum each column and divide all values by sum
    column_sums = df.loc[:,df.columns].sum(axis=0)
    df_n = df.div(column_sums)
    #scale by norm_scale_factor
    df_n = df_n.multiply(norm_scale_factor)
    #compute relevant stats
    mu = df_n.mean(axis=1)
    sigma = df_n.std(axis=1)
    cv = sigma/mu
    log2_mu = np.log2(mu)
    log2_cv = np.log2(cv)
    #get fit and use to get predicted values
    X = np.reshape(np.array(log2_mu),(log2_mu.shape[0],1))
    y = np.reshape(np.array(log2_cv),(log2_cv.shape[0],1))
    paramfit = LinearRegression().fit(X, y)
    log2_cv_pred = paramfit.predict(X)
    paramfit_h = HuberRegressor().fit(X,np.ravel(y))
    log2_cv_pred_h = paramfit_h.predict(X)
    #compute distance from regression line to actual CV values
    if use_huber:
        delta_y_2_pred = y - np.reshape(log2_cv_pred_h,(log2_cv_pred_h.shape[0],1))
    else:
        delta_y_2_pred = y - log2_cv_pred
    delta_y_2_pred = np.reshape(delta_y_2_pred, (delta_y_2_pred.shape[0],))
    #create dataframe and sort low to high by distance from fit line
    log_mucv_df = pd.DataFrame({'log2mu':log2_mu, 'log2cv':log2_cv, 'delta_cv': delta_y_2_pred })
    log_mucv_df_sorted = log_mucv_df.sort_values(by = 'delta_cv',ascending=False)
    #get position and gene names for plotting
    pos = np.array(log_mucv_df_sorted.iloc[:num_top_genes,0:2])
    genes = log_mucv_df_sorted.iloc[:num_top_genes].index
    
    if plot_flag==1:
        ax, fig = plt.subplots()
        plt.scatter(log2_mu,log2_cv, marker = '.')
        plt.xlabel('log2_mu')
        plt.ylabel('log2_cv')
        plt.plot(log2_mu, log2_cv_pred, c= 'r', label = 'linear regression')
        plt.plot(log2_mu, log2_cv_pred_h, c = 'y', label = 'huber regression')
        plt.legend()
        offset = 0.01
        for i in range(len(pos)):
            plt.text(pos[i][0]+offset,pos[i][1]+offset,genes[i])
    plt.show()
    
    return log_mucv_df_sorted

def get_top_cv_genes(df, cv_df, plot_flag, status_df):
    '''takes output of analyze cv, the cv_df sorted from high to low of cv values to fit line,
    and determines top number of genes to select based on gene index of point closest to origin in knee plot'''
    delta = cv_df.loc[:,'delta_cv']
    #add + 1 since end value in arange is exclusive
    x = np.arange(1,np.sum(delta>0)+1)
    y = np.flip(np.sort(delta.loc[delta>0]))
    #normalized by dividing by max value for each
    x_n = (x-np.min(x))/(np.max(x)-np.min(x))
    y_n = (y-np.min(y))/(np.max(y) - np.min(y))
    # find closest point to origin
    knee_df = find_knee(x_n,y_n)
    closest_point_to_origin = knee_df.iloc[0,:]
    gene_index = knee_df.index[0]
    if plot_flag==1:
        annotation_offset = 0.1
        fig,ax = plt.subplots()
        plt.xlabel('1:sum(delta_cv>0), normalized')
        plt.ylabel('sort(delta_cv(delta_cv>0), normalized')
        l = ax.scatter(x_n,y_n, marker = '.')
        cp = ax.scatter(closest_point_to_origin.iloc[0],closest_point_to_origin.iloc[1], c = 'r')
        plt.annotate('closest point to origin' + '\n' + 'gene index = '+str(gene_index), xy=(closest_point_to_origin.iloc[0],closest_point_to_origin.iloc[1]),
                     xytext=(closest_point_to_origin.iloc[0]+annotation_offset,closest_point_to_origin.iloc[1]+annotation_offset),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    )
        plt.show()
    #use gene index to update df accordingly
    updated_df = df.loc[cv_df.iloc[:gene_index,:].index,:]
    status_df.loc['get_top_cv_genes',:] = True
    
    return gene_index, updated_df, status_df

def log_and_standerdize_df(df, status_df, log=True):
    '''takes log and then performs standardization of gene expression matrix, returns np array'''
    if log:
        df = np.log2(df+1)
    #transpose since standard scaler expects X as n_samples x n_features in order to compute mean/std along features axis
    std_scale = preprocessing.StandardScaler().fit(df.T)
    log_std_arr = std_scale.transform(df.T)
    print ('column (gene) mean after standardization: {:.2f}'.format(log_std_arr[:,0].mean()))
    print ('column (gene) sigma after standardization: {:.2f}'.format(log_std_arr[:,0].std()))
    status_df.loc['log_and_standerdize',:] = True
    
    return log_std_arr,status_df

def analyze_pca(arr, n_components, optimize_n, plot_flag, status_df):
    '''performs pca on arr using n_components. plots PCA explained variance ratio as a function of components, 
    then plots a normalized version for both axes and uses closest point to origin to determine number of componets to keep'''
    pca = PCA(n_components).fit(arr)
    arr_pca = pca.transform(arr)
    if plot_flag == 1:
        ax,fig = plt.subplots()
        plt.title('PCA - Explained Variance Ratio')
        plt.plot(pca.explained_variance_ratio_)
        plt.ylabel('explained variance ratio (%)')
        plt.xlabel('component')
        #plt.xlim([0,15])
        plt.show()
    if optimize_n:
        #compute normalized variance ratio and component list
        variance_ratio_n = (pca.explained_variance_ratio_ - np.min(pca.explained_variance_ratio_))/(np.max(pca.explained_variance_ratio_)-np.min(pca.explained_variance_ratio_))
        max_component = arr.shape[1]
        min_component = 1
        component_list_n = ((np.arange(min_component, max_component+1)) - min_component)/(max_component - min_component)
        # find closest point to origin
        knee_df = find_knee(component_list_n,variance_ratio_n)
        pca_index = knee_df.index[0]
        if plot_flag == 1:
            closest_point_to_origin = knee_df.iloc[0,:]
            annotation_offset = 0.05
            fig,ax = plt.subplots()
            l = ax.scatter(component_list_n,variance_ratio_n, marker = '.')
            cp = ax.scatter(closest_point_to_origin.iloc[0],closest_point_to_origin.iloc[1], c = 'r')
            plt.annotate('closest point to origin' + '\n' + 'pca comp index = ' + str(pca_index), xy=(closest_point_to_origin.iloc[0],closest_point_to_origin.iloc[1]),
                        xytext=(closest_point_to_origin.iloc[0]+annotation_offset,closest_point_to_origin.iloc[1]+annotation_offset),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        )
            plt.xlabel('normalized component list')
            plt.ylabel('normalized variance ratio')
            plt.show()
        #update arr to contain up to pca_index number of components
        arr_pca_indexed = arr_pca[:,:pca_index]
    else:
        pca_index = n_components
        arr_pca_indexed = arr_pca
    if plot_flag == 1 and n_components>=3:
        #visualize first three pca components
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(arr_pca_indexed[:, 0], 
                arr_pca_indexed[:, 1], 
                arr_pca_indexed[:, 2], 
                s = 5)
        ax.set_title("First three PCA components")
        plt.show()
    status_df.loc['analyze_pca',:] = True
    
    return pca_index, arr_pca_indexed, status_df

def get_perplexity(pca_arr, cutoff, plot_flag, status_df):
    '''computes optimal perplexity param for use in do_tsne() using pairwise distance matrix. see comments in function for step by step details.'''
    #1) compute pairwise distance matrix (n_cells x n_cells) from PCA reduced matrix.
    D = squareform(pdist(pca_arr,metric='correlation'))
    #2) sort columns by ascending values
    D_sorted = np.sort(D, axis=0)
    if plot_flag == 1:
        #visualize first 4 columns of distance matrix
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(np.arange(D.shape[0]),D_sorted[:,0])
        axs[0, 0].set_title('D_sorted[:,0]')
        axs[0, 1].plot(np.arange(D.shape[0]),D_sorted[:,1], 'tab:orange')
        axs[0, 1].set_title('D_sorted[:,1]')
        axs[1, 0].plot(np.arange(D.shape[0]),D_sorted[:,2], 'tab:green')
        axs[1, 0].set_title('D_sorted[:,2]')
        axs[1, 1].plot(np.arange(D.shape[0]),D_sorted[:,3], 'tab:red')
        axs[1, 1].set_title('D_sorted[:,3]')

        for ax in axs.flat:
            ax.set(xlabel='cell_index', ylabel='counts')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.show()

    xn_list = []
    yn_list = []
    cut_off = 500
    x = np.arange(cut_off)
    #for every sorted column from the distance matrix:
    #3) Compute angle from first point to last point of column values from index 1:cutoff
    #4) use rotation matrix to rotate column values by this angle
    #5) take argmax for each rotated set of column values and store in list
    for i in range(D.shape[0]):
        y_e = float(D_sorted[cut_off:cut_off+1,i])
        y_1 = float(D_sorted[:1,i])
        x_e = cut_off
        x_1 = 0
        a = np.arctan((y_e-y_1)/(x_e-x_1))
        #rotate by -theta
        x = np.arange(cut_off)
        y = D_sorted[:cut_off,i]
        xn = (x*np.cos(a))+(y*np.sin(a))
        yn = -(x*np.sin(a))+(y*np.cos(a))
        yn_list.append(max(yn))
        xn_list.append(np.argmax(yn))
        if i == 0 and plot_flag == 1:
            #plot showing theta rotation on first column of distance matrix
            fig,ax = plt.subplots(2,1, sharex=True)
            ax[0].set_title('original 1st column of D')
            ax[0].plot(x,y, '.', ms = 1)
            ax[1].set_title('-theta rotated 1st column of D')
            ax[1].plot(xn,yn, '.', ms = 1)
            plt.show()

    #6) take median of list created in step 5, this is perplexity value
    perplexity = np.median(np.sort(xn_list))
    status_df.loc['get_perplexity',:] = True
    
    return perplexity, status_df

def do_tsne(arr,n_components, n_iter, learning_rate, early_exaggeration, init, perplexity, status_df):
    '''performs tsne on inputted arr with specified perplexity'''
    print ('creating tsne object with the following parameters: \n' + 
           'n_components:{}'.format(n_components) + '\n' +
           'n_iter: {}'.format(n_iter) + '\n' +
           'learning_rate: {}'.format(learning_rate) + '\n' +
           'early_exaggeration: {}'.format(early_exaggeration) + '\n' +
           'init: {}'.format(init) + '\n' +   
           'perplexity: {}'.format(perplexity)     
           )
    tsne = TSNE(n_components=n_components,
                n_iter=n_iter,
                learning_rate=learning_rate,
                early_exaggeration=early_exaggeration,
                init=init, 
                perplexity = perplexity)

    # Apply t-SNE on the arr
    X_tsne = tsne.fit_transform(arr)
    #visualise tsne
    ax, fig = plt.subplots()
    fig.scatter(X_tsne[:, 0], X_tsne[:, 1], s = 2)
    plt.show()
    status_df.loc['do_tsne',:] = True
    
    return X_tsne, status_df

def compute_eps(minpts, eps_prc, arr, status_df):
    '''Amit's method for computing epsilon parameter used in dbscan:
        1) compute distance matrix for input arr
        2) sort columns by ascending values
        3) determine knn_radius for each point in order to satisfy min_pts (get sorted row value @ row = minpts)
        Parameters
    ----------
    minpts: int
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    eps_prc: int
        epsilon percintile
    arr: numpy.ndarray
        2D input array, typically output from tsne
    Returns
    -------
    sorted_ind: 1-D array
        indexes that order the matrix
    '''
    D = squareform(pdist(arr, metric='euclidean'))
    D_sorted = np.sort(D, axis=0)
    knn_rad = D_sorted[minpts,:]
    epsilon = float(np.percentile(knn_rad, eps_prc))
    print ('params for dbscan')
    print ('minpts: ', minpts)
    print ('epsilon: ', str(epsilon) + '\n')
    status_df.loc['compute_eps',:] = True
    
    return epsilon, minpts, status_df

def do_dbscan(epsilon, minpts, arr, status_df):
    ''' Do dbscan using scikit-learn implementation:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        Parameters
    ----------
    epsilon:
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
    minpts: int
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    arr: numpy.ndarray
        2D input array, typically output from tsne
    Returns
    -------
    labels: 1-D array
       Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
    n_clusters_: int
        number of clusters (noise cluster removed)
    '''
    print (f"running dbscan with epsilon: {epsilon}  and minpts: {minpts}")
    db = DBSCAN(eps=epsilon, min_samples=minpts).fit(arr)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    
    arr_df = pd.DataFrame(arr, columns=['tsne-1','tsne-2'])
    #remove noise were label = -1
    arr_df_noise_rm = arr_df.drop(np.ravel(np.where(labels==-1)))
    labels_noise_rm = (db.labels_[db.labels_!=-1])
    
    fig,ax = plt.subplots()
    p = sns.scatterplot(data = arr_df_noise_rm,
                        x = 'tsne-1',
                        y= 'tsne-2',
                        hue = labels_noise_rm, 
                        legend = "full", 
                        palette = "deep",
                        s = 1)
    #sns.move_legend(p, "upper right", bbox_to_anchor = (1.17, 1.), title = 'Clusters')
    # Annotate with cluster labels at the median of each cluster
    for label in set(labels_noise_rm):
        if label != -1:
            cluster_median = arr_df_noise_rm[labels_noise_rm == label].median()
            ax.annotate(label, cluster_median, fontsize=8, color='black',
                        ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))
    
    p.legend_.remove()
    plt.title(f"Estimated number of clusters: {n_clusters_}, noise removed")
    plt.show()
    status_df.loc['do_dbscan',:] = True
    
    return labels_noise_rm, n_clusters_, arr_df_noise_rm, status_df

def histogram_pts_per_cluster(labels, minpts):
    '''Using labels output from DBSCAN, this function uses the counter method to 
    plots histogram of points per cluster'''
    cluster_pts = Counter(labels)
    fig,ax = plt.subplots()
    plt.bar(cluster_pts.keys(), cluster_pts.values())
    plt.xlabel('unique cluster')
    plt.ylabel('num pts (cells)')
    #plt.ylim([0,100])
    plt.axhline(y = minpts, color = 'r', linestyle = 'dashed', label = 'min_pts')
    plt.legend()
    plt.show()

def sort_by_cluster_label(df,meta_data_df,arr_df,labels):
    '''
    Sorts metadata and gene_expression data by cluster labels returned from do_dbscan in ascending order.
        Parameters
    ----------
    df: pandas.core.frame.DataFrame
        gene expression dataframe 
    meta_data_df: pandas.core.frame.DataFrame
        gene expression metadata
    arr_df: pandas.core.frame.DataFrame
        2D input array, typically output from tsne, in dataframe format
    Returns
    -------
    df_updated:
        reshuffled df orderded by ascending cluster label
    meta_data_df_updated:
        updated/reordered meta data with added labels row
    '''
    #arr_df = arr_df.insert(2, 'labels',labels)
    arr_df_sorted = arr_df.sort_values(by = 'labels')
    unique_labels = np.unique(labels) 
    #Sort meta data using arr_df_sorted index, then add cluster labels row
    meta_data_df  = meta_data_df.iloc[:,arr_df_sorted.index]
    cluster_labels = pd.DataFrame([list(arr_df_sorted['labels'])], columns=meta_data_df.columns, index = ['cluster_label'])
    # append new line to dataframe
    meta_data_df = pd.concat([meta_data_df, cluster_labels])
    #sort df using arr_df_sorted
    df_updated = pd.DataFrame(data = df.T, index=df.T.index, columns=df.T.columns)
    df_updated = df_updated.iloc[arr_df_sorted.index,:]
    
    return df_updated, meta_data_df, unique_labels

def inter_cluster_sort(df, meta_data_df, unique_labels, n_components, linkage_alg, dist_metric):
    '''
    Returns inter cluster sorted df and metadata, order determined as follows:
    1) compute mean per gene per cluster(n_genes x n_clusters)
    2) PCA reduce genes to n_components (10)
    3) Compute distance matrix on PCA reduced array using 'correlation' as distance metric
    4) Compute linkage on distance matrix using 'ward' linkage alg to determine cluster order
    '''
    #transpose since we PCA reduce on genes 
    df = df.T
    #compute mean for each gene, for each cluster
    mean_per_gene_per_cluster_arr = np.zeros((len(df.index),len(unique_labels)))
    for i in range(len(unique_labels)):
        cluster_mean_expr = np.mean(df.loc[:,meta_data_df.loc['cluster_label',:] == i], axis = 1)
        mean_per_gene_per_cluster_arr[:,i] = cluster_mean_expr
    #do pca
    pca = PCA(n_components).fit(mean_per_gene_per_cluster_arr.T)
    mpg_pc_pca = pca.transform(mean_per_gene_per_cluster_arr.T)
    #Compute condensed distance matrix
    D_cond = pdist(mpg_pc_pca, metric='correlation')
    #do linkage
    #31.03.24 - can enable optimal ordering with linkage(, , optimal_ordering = True)
    #31.03.24, try doing linkage on mpg_pc_pca (prj in Amits code), then get optimal order after 
    Z = linkage(mpg_pc_pca, linkage_alg, metric= dist_metric)
    fig = plt.figure()
    plt.title('Raw Linkage: ' + linkage_alg + '_' + dist_metric)
    dn = dendrogram(Z)
    plt.show()
    #get linkage order
    #linkage_cluster_order = dn['leaves']
    linkage_cluster_order = leaves_list(optimal_leaf_ordering(Z,D_cond))
    Z_ordered = optimal_leaf_ordering(Z,D_cond)
    print (linkage_cluster_order)
    
    fig = plt.figure()
    plt.title('Optimal Leaf Ordered Linkage: ' + linkage_alg + '_' + dist_metric)
    dn = dendrogram(Z_ordered)
    plt.show()
    #build new df using linkage order
    df_post_linkage = pd.DataFrame(index = df.index, columns = [])
    for i,v in enumerate(linkage_cluster_order):
        #print(meta_data_df.loc['cluster_label',meta_data_df.loc['cluster_label',:] == v])
        #print (df.iloc[:,np.where(meta_data_df.loc['cluster_label',:] == v)[0]])
        tmp = df.iloc[:,np.where(meta_data_df.loc['cluster_label',:] == v)[0]]
        df_post_linkage = pd.concat([df_post_linkage, tmp],axis = 1)
    #update metadata
    meta_data_df = meta_data_df.reindex(columns = df_post_linkage.columns)
    
    return df_post_linkage, meta_data_df, linkage_cluster_order, Z_ordered, mean_per_gene_per_cluster_arr

def intra_cluster_sort(df, meta_data_df, linkage_cluster_order):
    '''
    return intra cluster sorted df, meta_data_df. Function performs intra cluster sorting by taking 1D tsne for each cluster, then resorting
    by ascending value.
    '''
    df_post_linkage_intra_sorted = pd.DataFrame(index = df.index, columns = [])
    cluster_indices = []
    for c in linkage_cluster_order:
        x = df.iloc[:,np.where(meta_data_df.loc['cluster_label',:] == c)[0]]
        x_col = x.columns.to_list()
        x_arr = x.to_numpy()
        #perform tsne to reduce to 1D
        tsne = TSNE(n_components=1, perplexity=30)
        # Apply t-SNE to reduce to 1 feature x num_cells. end up with a 1D vector for each unique label. 
        X_tsne = tsne.fit_transform(x_arr.T)
        #create temp dataframe of x_col and 1D tsne
        tmp_df = pd.DataFrame({'x_col':x_col})
        tmp_df['X_tsne'] = X_tsne.tolist()
        #sort by ascending 1D tsne values
        tmp_df_sorted = tmp_df.sort_values(by = 'X_tsne')
        cluster_indices.append(np.array(tmp_df_sorted.index))
        #use index of tmp_df_sorted to reshuffle intra cluster
        x_intra_sorted = x.reindex(columns = tmp_df_sorted['x_col'])
        df_post_linkage_intra_sorted = pd.concat([df_post_linkage_intra_sorted,x_intra_sorted], axis = 1)
    #update metadata
    meta_data_df = meta_data_df.reindex(columns = df_post_linkage_intra_sorted.columns)
    
    return df_post_linkage_intra_sorted, meta_data_df, cluster_indices

def compute_marker_genes(df, meta_data_df, cluster_indices, linkage_cluster_order, n_markers):
    #store avg index for each cluster (used for plutting cluster label ticks), and set pointer to 0
    tmp = 0
    pos = [] 
    #initialize array to store cluster_expr_ratio to store ratio of mean gene_expression per cluster/mean gene expression across all cells
    cluster_expr_ratios = np.zeros((len(df.index),len(cluster_indices))) 
    #compute mean per gene across all cells
    mean_per_gene = np.mean(df.loc[:,:], axis=1)
    #initialize array to store mean of all postively expressed cells
    cluster_mean_pos = np.zeros((len(df.index),len(cluster_indices))) #intialize for mean of all postive cells
    #create dataframe where 1 = postive expression of cell
    arr_pos_bin = np.array(df>0).astype(int) #create array of same dim, setting all positive values to 1, otherwise 0
    df_pos = pd.DataFrame(arr_pos_bin, index = df.index, columns = df.columns)

    for idx, c in enumerate(zip(linkage_cluster_order,cluster_indices)):
        #append mean of cell indices btwn tmp and tmp + length of cluster
        pos.append(np.mean(np.arange(tmp,tmp+len(c[1]))))
        #update pointer
        tmp+=len(c[1])
        #compute mean expression for each cluster in linkage cluster order
        cluster_mean_expr = np.mean(df.loc[:,meta_data_df.loc['cluster_label',:] == c[0]], axis = 1)
        #store in arr
        cluster_expr_ratios[:,idx] = cluster_mean_expr/mean_per_gene
        #compute mean of postive expressed cells
        cluster_mean_p = np.mean(df_pos.loc[:,meta_data_df.loc['cluster_label',:] == c[0]], axis = 1)
        #store in arr
        cluster_mean_pos[:,idx] = cluster_mean_p
        
    #set any cluster mean pos <0.2 to 0
    cluster_mean_pos[cluster_mean_pos<0.2] = 0
    
    #compute gene index arrays for three different weights, slice off n_markers number of rows from top
    xi0 = np.multiply(cluster_expr_ratios, (cluster_mean_pos**0.001))
    xi0_df = pd.DataFrame(data = xi0, index = df.index)
    xi0_ind_arr = np.column_stack((xi0_df.iloc[:,i].sort_values(ascending=False).index for i in range(len(xi0_df.columns))))
    xi0_ind_arr = xi0_ind_arr[:n_markers,:]

    xi0p5 = np.multiply(cluster_expr_ratios, (cluster_mean_pos**0.5))
    xi0p5_df = pd.DataFrame(data = xi0p5, index = df.index)
    xi0p5_ind_arr = np.column_stack((xi0p5_df.iloc[:,i].sort_values(ascending=False).index for i in range(len(xi0p5_df.columns))))
    xi0p5_ind_arr = xi0p5_ind_arr[:n_markers,:]

    xi1 = np.multiply(cluster_expr_ratios, (cluster_mean_pos**1))
    xi1_df = pd.DataFrame(data = xi1, index = df.index)
    xi1_ind_arr = np.column_stack((xi1_df.iloc[:,i].sort_values(ascending=False).index for i in range(len(xi1_df.columns))))
    xi1_ind_arr = xi1_ind_arr[:n_markers,:]

    #vertically stack the gene index arrays
    stack = np.vstack((xi0_ind_arr,xi0p5_ind_arr,xi1_ind_arr))
    #flatten to 1D vector
    #By default, flatten() uses row major order, use 'F' to change to column major, since columns correspond to each cluster
    smash = stack.flatten('F')

    #get unique marker genes, preserving order
    marker_genes, idx = np.unique(smash, return_index=True)
    marker_genes_sorted = marker_genes[np.argsort(idx)]
    #print (f'len marker_Genes_sorted {len(marker_genes_sorted)}')
    
    
    #NEW########
    df_marker = df.loc[marker_genes_sorted,:]
    mean_per_gene_marker = np.mean(df_marker.loc[:,:], axis=1)
    arr_marker_pos_bin = np.array(df_marker>0).astype(int) #create array of same dim, setting all positive values to 1, otherwise 0
    df_marker_pos = pd.DataFrame(arr_marker_pos_bin, index = df_marker.index, columns = df_marker.columns)
    #initialize expr rations and mean pos again for just marker genes
    cluster_expr_ratios_marker = np.zeros((len(marker_genes_sorted),len(cluster_indices))) 
    cluster_mean_pos_marker = np.zeros((len(marker_genes_sorted),len(cluster_indices))) #intialize for mean of all postive cells
    
    #add additional looping over sorted markers
    for idx, c in enumerate(zip(linkage_cluster_order,cluster_indices)):
        #compute mean expression for each cluster in linkage cluster order
        cluster_mean_expr_marker = np.mean(df_marker.loc[:,meta_data_df.loc['cluster_label',:] == c[0]], axis = 1)
        #store in arr
        cluster_expr_ratios_marker[:,idx] = cluster_mean_expr_marker/mean_per_gene_marker
        #compute mean of postive expressed cells
        cluster_mean_p_marker = np.mean(df_marker_pos.loc[:,meta_data_df.loc['cluster_label',:] == c[0]], axis = 1)
        #store in arr
        cluster_mean_pos_marker[:,idx] = cluster_mean_p_marker
    
    xi0p5_marker = np.multiply(cluster_expr_ratios_marker, (cluster_mean_pos_marker**0.5))
    #for each row, get index of max column value
    ind = np.argmax(xi0p5_marker, axis=1)
    print (ind)
    #get indices of sorted list
    ind_s = np.argsort(ind)
    print (ind_s)
    #reorder marker genes accordingly
    marker_genes_sorted_final = [marker_genes_sorted[i] for i in ind_s]
    print (f'len marker_genes_sorted_final {len(marker_genes_sorted_final)}')

    return marker_genes_sorted_final, pos, ind, ind_s, marker_genes_sorted