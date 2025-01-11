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
#import harmonypy as hm
from matplotlib.cm import ScalarMappable
from datetime import date
import matplotlib as mpl
#import mpld3

today = str(date.today())
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

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



def gene_remover(gene_list, df):
    '''Takes in a list of genes (strings), and gene expression dataframe with genes as index. Checks which genes from list
    are also in the dataframe, and drops these rows. Returns updated dataframe with genes removed.'''
    genes_in_df = []
    count = 0
    for g in gene_list:
        if g in df.index:
            count+=1
            #print (g)
            genes_in_df.append(g)
    print ('removing ', len(genes_in_df), ' genes found in ', str(gene_list))
    #remove genes found in df
    df_updated = df.drop(labels = genes_in_df)
    
    return df_updated



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
    #print (df.shape)
    #drop 0 rows (otherwise std vector with zeros creates nan/infinity problems in CV)
    df = df.loc[~(df == 0).all(axis=1)]
    #print (df.shape)
    #sum each column and divide all values by sum
    column_sums = df.loc[:,df.columns].sum(axis=0)
    df_n = df.div(column_sums)
    #scale by norm_scale_factor
    df_n = df_n.multiply(norm_scale_factor)
    #print(np.where(df_n.sum(axis=1)==0))
    #compute relevant stats
    mu = df_n.mean(axis=1)
    sigma = df_n.std(axis=1) 
    #print ('num 0s in sigma ', len(np.where(sigma==0)[0]))
    cv = sigma/mu
    #add 1 so 0 expr values become (log(0+1)=0)
    log2_mu = np.log2(mu)
    log2_cv = np.log2(cv)
    #print (np.shape(log2_mu))
    #print ('num inf in log2cv ', len(np.where(np.isinf(log2_cv))[0])) 
    #print ('num NaNs in log2cv ', len(np.where(np.isnan(log2_cv))[0]))
    #get fit and use to get predicted values
    X = np.reshape(log2_mu,(log2_mu.shape[0],1))
    y = np.reshape(log2_cv,(log2_cv.shape[0],1))
    #print (np.all(np.isfinite(X)))
    #print (np.all(np.isfinite(y)))
    #print (X.shape)
    #print (y.shape)
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
    plt.xticks([])
    plt.yticks([])
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
    print ('epsilon percentile', eps_prc)
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
    plt.xticks([])
    plt.yticks([])
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
    #remove old cluster_labels from meta_data if already present
    if 'cluster_label' in meta_data_df.index:
        print('found existing cluster_labels in meta_data_df... removing old labels...')
        meta_data_df = meta_data_df.drop(['cluster_label'])

    #pandas bug 'nonetype .sort values' sometimes requires commenting out arr_df.insert() line since the labels are already added
    #, and then just running function again...
    #arr_df = arr_df.insert(2, 'labels',labels)
    arr_df.insert(2, 'labels',labels)
    arr_df_sorted = arr_df.sort_values(by = 'labels')
    unique_labels = np.unique(labels) 
    #Sort meta data using arr_df_sorted index, then add cluster labels row
    meta_data_df  = meta_data_df.iloc[:,arr_df_sorted.index]
    cluster_labels = pd.DataFrame([list(arr_df_sorted['labels'])], columns=meta_data_df.columns, index = ['cluster_label'])
    # append new line to dataframe
    meta_data_df = pd.concat([meta_data_df, cluster_labels])
    #tranpose to have features as columns
    df_updated = pd.DataFrame(data = df.T, index=df.T.index, columns=df.T.columns)
    #sort df using arr_df_sorted index
    df_updated = df_updated.iloc[arr_df_sorted.index,:]
    
    return df_updated, meta_data_df, unique_labels

def inter_cluster_sort(df, meta_data_df, unique_labels, n_components, linkage_alg, dist_metric, mode = None):
    '''
    Returns inter cluster sorted df and metadata, order determined as follows:
    1) compute mean per gene per cluster(n_genes x n_clusters)
    2) PCA reduce genes to n_components (10)
    3) Compute distance matrix on PCA reduced array using 'correlation' as distance metric
    4) Compute linkage on distance matrix using 'ward' linkage alg to determine cluster order
    '''
    #transpose since we PCA reduce on genes (df _ls is inputted as cellsxgenes)
    #remove transpose for _raw (already genesxcells)
    #df = df.T
    #compute mean for each gene, for each cluster
    mean_per_gene_per_cluster_arr = np.zeros((len(df.index),len(unique_labels)))
    for i in range(len(unique_labels)):
        #cluster_mean_expr = np.mean(df.loc[:,meta_data_df.loc['cluster_label',:] == i], axis = 1)
        #using raw expr (not _ls), take mean of log2x + 1 to match amit processing   
        if mode == None:
            cluster_mean_expr = np.mean(np.log2(df.loc[:,meta_data_df.loc['cluster_label',:] == i]+1), axis = 1)
        else:
            #print ('rc mode inter cluster sort')
            cluster_mean_expr = np.mean(np.log2(df.loc[:,meta_data_df.loc['cluster_label',:] == i+1]+1), axis = 1)
        mean_per_gene_per_cluster_arr[:,i] = cluster_mean_expr
    #print("mean per gene per cluster array contains " + np.isnan(mean_per_gene_per_cluster_arr).any() + " nan values") 
    #do pca
    pca = PCA(n_components).fit(mean_per_gene_per_cluster_arr.T)
    mpg_pc_pca = pca.transform(mean_per_gene_per_cluster_arr.T)
    #Compute condensed distance matrix
    D_cond = pdist(mpg_pc_pca, metric=dist_metric)
    #do linkage
    #31.03.24 - can enable optimal ordering with linkage(, , optimal_ordering = True)
    #31.03.24, try doing linkage on mpg_pc_pca (prj in Amits code), then get optimal order after 
    Z = linkage(mpg_pc_pca, linkage_alg, metric= dist_metric)
    fig = plt.figure()
    plt.title('Raw Linkage: ' + linkage_alg + '_' + dist_metric)
    dn = dendrogram(Z)
    plt.show()
    #get linkage order
    linkage_cluster_order_po = dn['leaves']
    linkage_cluster_order = leaves_list(optimal_leaf_ordering(Z,D_cond))
    Z_ordered = optimal_leaf_ordering(Z,D_cond)
    print (linkage_cluster_order)
    
    fig = plt.figure()
    plt.title('Optimal Leaf Ordered Linkage: ' + linkage_alg + '_' + dist_metric)
    dn = dendrogram(Z_ordered)
    plt.savefig('/bigdata/isaac/' + 'dd_'+today + '.pdf',transparent = True)
    plt.show()
    #build new df using linkage order
    df_post_linkage = pd.DataFrame(index = df.index, columns = [])
    #try using _po to resolve problematic ordering
    #for i,v in enumerate(linkage_cluster_order_po):
    for i,v in enumerate(linkage_cluster_order):
        #print(meta_data_df.loc['cluster_label',meta_data_df.loc['cluster_label',:] == v])
        #print (df.iloc[:,np.where(meta_data_df.loc['cluster_label',:] == v)[0]])
        if mode == None:
            tmp = df.iloc[:,np.where(meta_data_df.loc['cluster_label',:] == v)[0]]
        if mode == 'rc':
            tmp = df.iloc[:,np.where(meta_data_df.loc['cluster_label',:] == v+1)[0]]
        df_post_linkage = pd.concat([df_post_linkage, tmp],axis = 1)
    #update metadata
    meta_data_df = meta_data_df.reindex(columns = df_post_linkage.columns)
    print ('metadata df shape post inter cluster sort')
    print (meta_data_df.shape)
    return df_post_linkage, meta_data_df, linkage_cluster_order, Z_ordered, mean_per_gene_per_cluster_arr, linkage_cluster_order_po

def intra_cluster_sort(df, meta_data_df, linkage_cluster_order, mode = None):
    '''
    return intra cluster sorted df, meta_data_df. Function performs intra cluster sorting by taking 1D tsne for each cluster, then resorting
    by ascending value.
    '''
    df_post_linkage_intra_sorted = pd.DataFrame(index = df.index, columns = [])
    cluster_indices = []
    for c in linkage_cluster_order:
        if mode == None:
            x = df.iloc[:,np.where(meta_data_df.loc['cluster_label',:] == c)[0]]
        else:
            x = df.iloc[:,np.where(meta_data_df.loc['cluster_label',:] == c+1)[0]]
        x_col = x.columns.to_list()
        x_arr = x.to_numpy()
        #print (c)
        #print(np.shape(x_arr))
        #perform tsne to reduce to 1D
        #changed perplexity from 30 to 15 after harmony integration (for original dimorph_processing_nb)
        #for gaba vglut1 analysis - 6
        # vglut2 had to drop to 5 (cluster with 6 cells)
        tsne = TSNE(n_components=1, perplexity=5)
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

def compute_marker_genes(df, meta_data_df, cluster_indices, linkage_cluster_order, folder, n_markers, class_score_name = None):
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
    #2.12.24 added list() in each np.column stack calls
    #compute gene index arrays for three different weights, slice off n_markers number of rows from top
    xi0 = np.multiply(cluster_expr_ratios, (cluster_mean_pos**0.001))
    xi0_df = pd.DataFrame(data = xi0, index = df.index)
    #print (xi0_df.shape)
    #print (type(list(xi0_df.iloc[:,i].sort_values(ascending=False).index for i in range(len(xi0_df.columns)))))
    xi0_ind_arr = np.column_stack((list(xi0_df.iloc[:,i].sort_values(ascending=False).index for i in range(len(xi0_df.columns)))))
    xi0_ind_arr = xi0_ind_arr[:n_markers,:]

    xi0p5 = np.multiply(cluster_expr_ratios, (cluster_mean_pos**0.5))
    xi0p5_df = pd.DataFrame(data = xi0p5, index = df.index)
    xi0p5_ind_arr = np.column_stack((list(xi0p5_df.iloc[:,i].sort_values(ascending=False).index for i in range(len(xi0p5_df.columns)))))
    xi0p5_ind_arr = xi0p5_ind_arr[:n_markers,:]

    xi1 = np.multiply(cluster_expr_ratios, (cluster_mean_pos**1))
    xi1_df = pd.DataFrame(data = xi1, index = df.index)
    if class_score_name:
        xi1_df.to_csv(folder+str(class_score_name)+'_raw.csv')
    xi1_scores = np.column_stack((list(xi1_df.iloc[:,i].sort_values(ascending=False) for i in range(len(xi1_df.columns)))))
    if class_score_name:
        xi1_scores_df = pd.DataFrame(xi1_scores,columns=linkage_cluster_order)
        xi1_scores_df.to_csv(folder+str(class_score_name)+'.csv')
    xi1_ind_arr = np.column_stack((list(xi1_df.iloc[:,i].sort_values(ascending=False).index for i in range(len(xi1_df.columns)))))
    if class_score_name:
        xi1_ind_arr_df = pd.DataFrame(xi1_ind_arr, columns=linkage_cluster_order)
        xi1_ind_arr_df.to_csv(folder+str(class_score_name)+'_ind.csv')
    xi1_ind_arr = xi1_ind_arr[:n_markers,:]
    #print (xi1_ind_arr)

    #vertically stack the gene index arrays
    stack = np.vstack((xi0_ind_arr,xi0p5_ind_arr,xi1_ind_arr))
    #flatten to 1D vector
    #By default, flatten() uses row major order, use 'F' to change to column major, since columns correspond to each cluster
    smash = stack.flatten('F')

    #get unique marker genes, preserving order
    marker_genes, idx = np.unique(smash, return_index=True)
    marker_genes_sorted = marker_genes[np.argsort(idx)]
    print (f'len marker_Genes_sorted {len(marker_genes_sorted)}')
    
    
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
    #print (ind)
    #print (len (ind))
    #get indices of sorted list
    ind_s = np.argsort(ind,axis = 0)
    #print (ind_s)
    #print (len(ind_s))
    #reorder marker genes accordingly
    marker_genes_sorted_final = [marker_genes_sorted[i] for i in ind_s]
    print (f'len marker_genes_sorted_final {len(marker_genes_sorted_final)}')

    return marker_genes_sorted_final, pos, ind, ind_s, marker_genes_sorted

def get_tgfs_from_tg(tg):
    #compute formated tgfs from tg
    #formatting fix #1 - add newline char after each gene so labels are stacked vertically
    tgf = []
    for i,x in enumerate(tg):
        a = '\n '.join(tg[i])
        tgf.append(a)

    #formatting fix#2 - add space before 1st char in for each gene string list to fix alignment issue 
    tgfs = []
    for i in tgf:
        x = ' ' + i
        tgfs.append(x)
    return tgfs

def get_heatmap_labels(mgs, ind, ind_s):
    '''Uses sorted list of marker genes (mgs), column index of max marker value (ind), and arg sorted version of ind (ind_s) outputted by compute_marker_genes() to get
    "waterfall" heatmap gene labels'''
    
    #use list of indices corresponding to column with max value of marker array
    #to get sorted version
    indy = np.sort(ind)
    #get unique indy
    seen = set()
    indy_unique = []
    for item in indy:
        if item not in seen:
            indy_unique.append(item)
            seen.add(item)
    print (indy_unique)
    print (len(indy_unique))
    #get gene indices
    g = []
    #for i in np.arange(0,len(ind)):
        #print(i)
        #x = ind_s[np.where(indy==i)]
        #g.append(x)
    for i,v in enumerate(indy_unique):
        #print(i)
        x = ind_s[np.where(indy==v)]
        g.append(x)
    
    print (g)
    print (len(g))
    #use full marker gene list to convert list of indices to gene names
    tg = []
    for g_sub in g:
        gene = [mgs[x] for x in g_sub]
        #print(gene)
        tg.append(gene)
    #drop empty arrays
    #tg = [arr for arr in tg if len(arr) > 0]
    #formatting fix #1 - add newline char after each gene so labels are stacked vertically
    tgf = []
    for i,x in enumerate(tg):
        a = '\n '.join(tg[i])
        tgf.append(a)

    #formatting fix#2 - add space before 1st char in for each gene string list to fix alignment issue 
    tgfs = []
    for i in tgf:
        x = ' ' + i
        tgfs.append(x)
        
    return tg, tgfs

def get_heatmap_cluster_borders(meta_data_df):
    '''Use cluster labels in meta_data_df to determine index position of vertical border lines shown on heatmap.'''
    x = list(meta_data_df.loc['cluster_label',:])
    change_indices = [0]  # Initialize with the index of the first element

    # Iterate through the list starting from the second element
    for i in range(1, len(x)):
        # Check if the current value is different from the previous value
        if x[i] != x[i - 1]:
            # If a change is detected, append the index to the list
            change_indices.append(i)
    
    #add last element
    change_indices.append(len(x)-1)
    print("Indices where the value changes:", change_indices)

    change_indices = change_indices[1:] #ignore initial value set
    
    return change_indices

def plot_marker_heatmap(df, pos, linkage_cluster_order, change_indices, tg, tgfs, linkage_alg, dist_metric, folder, fs_waterfall = None, cluster_labels = None, savefig = False, cell_class = 'OG'):
    '''Plots markerheatmap, input expects df to be log/std. Cell_class used to update filename when savefig is true,
    defaults to OG.'''
    fig, ax = plt.subplots(figsize = (10,10))
    sns.heatmap(df, robust=True,  cmap="viridis", yticklabels=True)
    ax.set_xticks(ticks = pos, labels = linkage_cluster_order)
    ax.set_yticks([])
    ax.vlines(change_indices, 0 ,len(df.index), colors='gray', lw = 0.1)
    ypos = 0
    vertical_spacing = 0.2
    for i,v in enumerate(tg):
    #for i,v in enumerate(change_indices):
        xpos = change_indices[i]
        #print (i)
        plt.text(xpos,ypos, tgfs[i], 
                 verticalalignment='top', horizontalalignment = 'left', color="gray", fontsize = fs_waterfall, fontweight = 'bold', fontname = 'monospace')
        ypos+=int(len(tg[i]))
    if cluster_labels:
        ypos = 0
        for i,l in enumerate(zip(tg,cluster_labels)):
        #for i,v in enumerate(change_indices):
            xpos = change_indices[i]
            plt.text(xpos,ypos, cluster_labels[i], 
                    verticalalignment='top', horizontalalignment = 'right', color="white", fontsize = fs_waterfall, fontweight = 'bold',
                    rotation = 'vertical')
            ypos+=int(len(tg[i]))

    if savefig:
        plt.savefig(folder+'heatmap_' + cell_class + '_' +linkage_alg+'_'+dist_metric+'_' +today+'.jpeg', dpi = 1200)
        #use mpld3 to save interactive plot as html
        #html_str = mpld3.fig_to_html(fig)
        #Html_file= open(folder+ 'mlpd3_heatmap_' + cell_class + '_' +linkage_alg+'_'+dist_metric + '_' +today+'.html',"w")
        #Html_file.write(html_str)
        #Html_file.close()
        #filename = folder+ 'sh_mlpd3_heatmap_' + cell_class + '_' +linkage_alg+'_'+dist_metric + '_' +today +'.html'
        #mpld3.save_html(fig, filename)

    plt.show()

def get_cluster_labels(folder,sd_labels_df_fn):
    sd_labels_df = pd.read_csv(folder + sd_labels_df_fn)
    # corr_labels = [l for l in list(sd_labels_df.loc[:,'sd_label_complete']) if len(l)>3]
    # corr_labels_filtered = corr_labels.copy()
    # for i,l in enumerate(corr_labels_filtered):
    #     l_id = int(l.split(' ')[0])
    #     if l_id in all_dropped_clusters:
    #         print ('removing id: ', l_id)
    #         corr_labels_filtered.pop(i)
    #corr_labels = [l for l in list(sd_labels_df.loc[:,'sd_label_complete'])]
    #cluster_labels = [x.split(' ')[1] if ' ' in x else x for x in corr_labels]
    labels = [l for l in list(sd_labels_df.loc[:,'markers'])]
    return labels

def update_metadata_cluster_labels(linkage_cluster_order, meta_data_df_plis, mode = None):
    '''take plis sorted metadata, rename IDS to be sequential for aesthetics down the road'''
    if mode == None:
        mg_cl_conv_dict = {old_key:new_key  for new_key, old_key in enumerate(linkage_cluster_order, start=1)}
    else:
        mg_cl_conv_dict = {old_key+1:new_key  for new_key, old_key in enumerate(linkage_cluster_order, start=1)}
    print (mg_cl_conv_dict)
    tmp = np.array(meta_data_df_plis.loc['cluster_label'])
    print (np.unique(tmp))
    tmp_c = [mg_cl_conv_dict[x] for x in tmp]
    meta_data_df_plis.loc['cluster_label'] = tmp_c
    linkage_cluster_order_updated = np.arange(1,len(linkage_cluster_order)+1)
    return meta_data_df_plis, linkage_cluster_order_updated

def update_metadata_w_markers(folder, meta_data_df_plis_filtered, linkage_cluster_order_filtered, labels, cell_class, write_to_file = False):
    '''updates metadata with markers from cell comp analysis'''
    meta_data_df_plis_filtered_markers = meta_data_df_plis_filtered.copy()
    #with open(folder + mg_cl_dict_final_fn) as json_data:
        #mg_cl_dict_final = json.load(json_data)
    tmp_dict = dict(zip(linkage_cluster_order_filtered,labels))
    mg_cl_dict_final_sorted_filtered = {int(k):v for k,v in tmp_dict.items()}
    m_list = []
    for v in np.array(meta_data_df_plis_filtered.loc['cluster_label']):
        m = tmp_dict[v]
        #comment out .join() when running nn
        #mc = '-'.join(m)
        m_list.append(m)
    m_list = np.reshape(np.array(m_list),(1,len(m_list)))
    #print (m_list.shape)
    #build marker row
    markers = pd.DataFrame(m_list, columns = meta_data_df_plis_filtered.columns, index = ['markers'])
    meta_data_df_plis_filtered_markers = pd.concat([meta_data_df_plis_filtered, markers])
    if write_to_file:
        #write updated metadata to file
        file = cell_class + 'meta_data_df_plis_f_markers_' + today
        meta_data_df_plis_filtered_markers.to_json(folder+file+'.json')
    return meta_data_df_plis_filtered_markers

def get_boolean_vecs(meta_data_df):
    '''Takes in metadata dataframe, maps relevant feature, and returns boolean vectors for each'''
    
    #map male/female cells to boolean
    s_map = {'M':True,'F':False}
    s_bool = np.array(meta_data_df.loc['Sex'].map(s_map))
    #map breeder/naive cells to boolean
    bn_map = {"Breeder-M":True, "Breeder-F":True, "Naïve-M":False, "Naïve-F":False}
    bn_bool = np.array(meta_data_df.loc['Group'].map(bn_map))
    #map gaba cells to boolean
    gaba_map = {'Doublet':False, 'GABA':True, 'Nonneuronal':False, 'Vglut1':False, 'Vglut2':False}
    gaba_bool = np.array(meta_data_df.loc['cell_class'].map(gaba_map))
    #map doublet cells to boolean
    doublet_map = {'Doublet':True, 'GABA':False, 'Nonneuronal':False, 'Vglut1':False, 'Vglut2':False}
    doublet_bool = np.array(meta_data_df.loc['cell_class'].map(doublet_map))
    #map vglut1 cells to boolean
    vglut1_map = {'Doublet':False, 'GABA':False, 'Nonneuronal':False, 'Vglut1':True, 'Vglut2':False}
    vglut1_bool = np.array(meta_data_df.loc['cell_class'].map(vglut1_map))
    #map vglut2 cells to boolean
    vglut2_map = {'Doublet':False, 'GABA':False, 'Nonneuronal':False, 'Vglut1':False, 'Vglut2':True}
    vglut2_bool = np.array(meta_data_df.loc['cell_class'].map(vglut2_map))
    #map nonneuronal cells to boolean
    nonneuronal_map = {'Doublet':False, 'GABA':False, 'Nonneuronal':True, 'Vglut1':False, 'Vglut2':False}
    nonneuronal_bool = np.array(meta_data_df.loc['cell_class'].map(nonneuronal_map))
    
    return s_bool,bn_bool, gaba_bool, doublet_bool, vglut1_bool, vglut2_bool, nonneuronal_bool

def plot_marker_heatmap_w_bool_bars(df, pos, linkage_cluster_order, change_indices, tg, tgfs, linkage_alg, dist_metric, folder, sex_bool, group_bool, gaba_bool, doublet_bool, vglut1_bool, vglut2_bool, nonneuronal_bool, savefig = False, cell_class = 'OG'):
    '''Same as plot marker heatmap above, but additionally takes in boolean vectors sex, group, and class and displays these boolean vectors above heatmap.'''
    
    
    fig, (ax1,ax2,ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8,1, figsize = (10,10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 1, 15]})
    #sex boolean vector visual bar
    ax1.imshow([sex_bool],cmap='gray', aspect = 'auto')
    ax1.set_ylabel('sex')
    ax1.set_yticks([])
    ax1.set_xticks([])

    #group (boolean/naive) boolean vector visual bar
    ax2.imshow([group_bool],cmap='gray', aspect = 'auto')
    ax2.set_ylabel('group')
    ax2.set_yticks([])
    ax2.set_xticks([])

    #gaba boolean vector visual bar
    ax3.imshow([gaba_bool],cmap='gray', aspect = 'auto')
    ax3.set_ylabel('gaba')
    ax3.set_yticks([])
    ax3.set_xticks([])
    
    #doublet boolean vector visual bar
    ax4.imshow([doublet_bool],cmap='gray', aspect = 'auto')
    ax4.set_ylabel('doublet')
    ax4.set_yticks([])
    ax4.set_xticks([])

    #vglut1 boolean vector visual bar
    ax5.imshow([vglut1_bool],cmap='gray', aspect = 'auto')
    ax5.set_ylabel('vglut1')
    ax5.set_yticks([])
    ax5.set_xticks([])

    #vglut2 boolean vector visual bar
    ax6.imshow([vglut2_bool],cmap='gray', aspect = 'auto')
    ax6.set_ylabel('vglut2')
    ax6.set_yticks([])
    ax6.set_xticks([])

    #nonneuronal boolean vector visual bar
    ax7.imshow([nonneuronal_bool],cmap='gray', aspect = 'auto')
    ax7.set_ylabel('nn')
    ax7.set_yticks([])
    ax7.set_xticks([])

    #use code beloew in heatmap() to configure colobar loc if desired, otherwise hide it
    #cbar_kws = dict(use_gridspec=False,location="top")'''
    ax8 = sns.heatmap(df, robust=True,  cmap="viridis", yticklabels=True,  cbar=False)
    ax8.set_xticks(ticks = pos, labels = linkage_cluster_order)
    ax8.set_yticks([])
    ax8.vlines(change_indices, -100 ,300, colors='gray', lw = 0.1)

    ypos = 0

    for i,v in enumerate(tg):
        xpos = change_indices[i]
        plt.text(xpos,ypos, tgfs[i], 
                 verticalalignment='top', horizontalalignment = 'left', color="gray", fontsize = 2.9)
        ypos+=int(len(tg[i]))
    
    if savefig:
        plt.savefig(folder + 'heatmap_w_bool_bars_' + cell_class + '_' +linkage_alg+'_'+dist_metric+'_' +today+'.png', dpi = 1200)

    plt.tight_layout()
    plt.show()

def plot_subclass_marker_heatmap_w_bool_bars(df, pos, linkage_cluster_order, change_indices, tg, tgfs, cluster_labels, linkage_alg, dist_metric, folder, sex_bool, group_bool, savefig = False, cell_class = 'OG'):
    '''Same as plot marker heatmap w bool bars, but use for a subclass (i.e. gaba, glut, nn), so class bool bars are removed'''
    
    
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (10,10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 15]})
    #sex boolean vector visual bar
    ax1.imshow([sex_bool],cmap='gray', aspect = 'auto')
    ax1.set_ylabel('sex')
    ax1.set_yticks([])
    ax1.set_xticks([])

    #group (boolean/naive) boolean vector visual bar
    ax2.imshow([group_bool],cmap='gray', aspect = 'auto')
    ax2.set_ylabel('group')
    ax2.set_yticks([])
    ax2.set_xticks([])

    #use code beloew in heatmap() to configure colobar loc if desired, otherwise hide it
    #cbar_kws = dict(use_gridspec=False,location="top")'''
    ax3 = sns.heatmap(df, robust=True,  cmap="viridis", yticklabels=True,  cbar=False)
    ax3.set_xticks(ticks = pos, labels = linkage_cluster_order)
    ax3.set_yticks([])
    ax3.vlines(change_indices, -100 ,300, colors='gray', lw = 0.1)

    ypos = 0

    for i,v in enumerate(tg):
        xpos = change_indices[i]
        plt.text(xpos,ypos, tgfs[i], 
                 verticalalignment='top', horizontalalignment = 'left', color="gray", fontsize = 1.3)
        ypos+=int(len(tg[i]))
    
    #add cluster labels
    ypos = 0
    for i,l in enumerate(zip(tg,cluster_labels)):
        xpos = change_indices[i]
        plt.text(xpos,ypos, cluster_labels[i], 
                verticalalignment='top', horizontalalignment = 'right', color="white", fontsize = 1.7, fontweight = 'bold',
                rotation = 'vertical')
        ypos+=int(len(tg[i]))
        
    if savefig:
        plt.savefig(folder+ 'subclass_heatmap_w_bool_bars_' + cell_class + '_' +linkage_alg+'_'+dist_metric+'_' +today+'.png', dpi = 1200)

    plt.tight_layout()
    plt.show()

def compute_marker_means(GABA_marker, 
                         Vglut1_marker, 
                         Vglut2_marker, 
                         exclude_markers_updated,
                         df_marker,
                         meta_data_df,
                         linkage_cluster_order,
                         m_factor):
    
    #dictionary mapping flag to class type
    flag_dict = {1:'GABA', 2:'Vglut1', 3:'Vglut2', 4:'Nonneuronal' , 5:'Doublet'}
    
    #initialize flag vector to store flags correspondings class type for each cluster
    gabaglut = np.zeros((1,len(linkage_cluster_order)))[0]
    
    #intialize empty cell class dataframe to be added to meta data, indicating class type for each cell
    cell_class = pd.DataFrame(columns=meta_data_df.columns, index = ['cell_class'])
    
    #initialize lists to store mean of each marker within each cluster
    mu_g = []
    mu_vg1 = []
    mu_vg2 = []
    mu_nn = []

    #std dev of mean of each marker within each cluster
    std_g = []
    std_vg1 = []
    std_vg2 = []
    std_nn = []

    #loop through each cluster
    for i,c in enumerate(linkage_cluster_order):
        #extract expression data of cluster c
        tmp = df_marker.iloc[:,np.where(meta_data_df.loc['cluster_label']==c)[0]]
        #get row means for markers within cluster
        GABA_marker_mean = np.mean(tmp.loc[GABA_marker,:])
        Vglut1_marker_mean = np.mean(tmp.loc[Vglut1_marker,:])
        Vglut2_marker_mean = np.mean(tmp.loc[Vglut2_marker,:])
        nonneuro_mean = np.mean(tmp.loc[exclude_markers_updated,:].sum(axis=0))

        #get row stds for markers within cluster
        GABA_marker_std = np.std(tmp.loc[GABA_marker,:], axis=0)
        Vglut1_marker_std = np.std(tmp.loc[Vglut1_marker,:], axis=0)
        Vglut2_marker_std = np.std(tmp.loc[Vglut2_marker,:], axis=0)
        nonneuro_std = np.std(tmp.loc[exclude_markers_updated,:].sum(axis=0), axis=0)

        #append mean to respective list
        mu_g.append(GABA_marker_mean)
        mu_vg1.append(Vglut1_marker_mean)
        mu_vg2.append(Vglut2_marker_mean)
        mu_nn.append(nonneuro_mean)

        #append std to respective list
        std_g.append(GABA_marker_std)
        std_vg1.append(Vglut1_marker_std)
        std_vg2.append(Vglut2_marker_std)
        std_nn.append(nonneuro_std)    

        #sort means descending
        marker_means = np.flip(np.sort(np.array([GABA_marker_mean,
                                                 max(Vglut1_marker_mean,
                                                 Vglut2_marker_mean),
                                                 nonneuro_mean])))
        #print (marker_means)

        #classify cluster based on greatest mean, adding flag to gabaglut, and corresponding class type to cell_class
        if marker_means[0]>m_factor*marker_means[1]:
            if marker_means[0] == GABA_marker_mean:
                #print ('gaba', GABA_marker_mean)
                gabaglut[i] = 1
                cell_class.iloc[:,np.where(meta_data_df.loc['cluster_label']==c)[0]] = flag_dict[1]
            if marker_means[0] == Vglut1_marker_mean:
                #print ('vglut1', Vglut1_marker_mean)
                gabaglut[i] = 2
                cell_class.iloc[:,np.where(meta_data_df.loc['cluster_label']==c)[0]] = flag_dict[2]
            if marker_means[0] == Vglut2_marker_mean:
                #print ('vglut2', Vglut2_marker_mean)
                gabaglut[i] = 3
                cell_class.iloc[:,np.where(meta_data_df.loc['cluster_label']==c)[0]] = flag_dict[3]
            if marker_means[0] == nonneuro_mean:
                #print ('nonneuro', nonneuro_mean)
                gabaglut[i] = 4
                cell_class.iloc[:,np.where(meta_data_df.loc['cluster_label']==c)[0]] = flag_dict[4]
        #if first mean is not at least 2x second mean in descending list, flag clustter as doublet
        #note factor of two is arbitrary...
        else:
            gabaglut[i] = 5
            cell_class.iloc[:,np.where(meta_data_df.loc['cluster_label']==c)[0]] = flag_dict[5]
        #print (tmp.shape)
        
    #use flag dict to convert flag vector to list of class type
    gabaglut_l = [flag_dict[i] for i in gabaglut]

    # create another dictionary mapping cluster label to class type
    label_to_class_map = dict(zip(linkage_cluster_order, gabaglut_l))

    #append cell class to metadata
    meta_data_df = pd.concat([meta_data_df, cell_class])
    
    return mu_g, mu_vg1, mu_vg2, mu_nn, std_g, std_vg1,std_vg2,std_nn, label_to_class_map, meta_data_df

def plot_cell_class(arr_df, labels_to_class_map,labels):
    
    arr_df_sorted = arr_df.sort_values(by = 'labels')
    arr_df_class = arr_df_sorted.copy()
    arr_df_class.insert(3, 'cell_class', value = None)
    #use labels to class map 
    for i,l in enumerate(arr_df_class['labels']):
        arr_df_class.iloc[i,3] = labels_to_class_map[l]
    #seperate array needed for labeling clusters
    arr_xy = arr_df.drop('labels', axis = 'columns')

    fig,ax = plt.subplots(figsize = (9,6))
    for n, grp in arr_df_class.groupby('cell_class'):
        ax.scatter(x = 'tsne-1',y = 'tsne-2', data=grp, label=n, s = 1)
    lgnd = ax.legend()
    for handle in lgnd.legend_handles:
        handle.set_sizes([10.0])

    for label in set(labels):
        if label != -1:
            cluster_median = arr_xy[labels == label].median()
            #print (cluster_median)
            ax.annotate(text = label, xy=cluster_median, fontsize=8, color='black',
                        ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))

    plt.xticks([])
    plt.yticks([])
    plt.show()  

    return arr_df_class, arr_xy

def plot_marker_means(mu_g,mu_vg1,mu_vg2,mu_nn,linkage_cluster_order):
    fig,ax = plt.subplots(figsize = (9,6))

    x = linkage_cluster_order
    xt =  np.arange(len(linkage_cluster_order))
    plt.bar(xt-0.3,mu_g, width=0.2, color = 'orange', label = 'mu_gaba')
    plt.bar(xt-.1, mu_vg1, width=0.2,color = 'red', label = 'mu_vglut1')
    plt.bar(xt+0.1, mu_vg2, width = 0.2, color = 'purple', label = 'mu_vglut2')
    plt.bar(xt+0.3, mu_nn, width = 0.2, color = 'green', label = 'mu_nn')
    plt.legend()
    plt.xlabel('cluster label')
    plt.ylabel('mean expression')
    plt.xticks(ticks = np.arange(len(linkage_cluster_order)),labels=linkage_cluster_order)
    plt.show()

def plot_marker_mean_std(marker, mu_,std_, linkage_cluster_order):
    fig,ax = plt.subplots(figsize = (9,6))

    x = linkage_cluster_order
    xt =  np.arange(len(linkage_cluster_order))
    plt.bar(xt-0.3,mu_, width=0.2, color = 'orange', label = marker)
    plt.errorbar(xt-0.3,mu_, yerr=std_, fmt="o", color="r")

    plt.legend()
    plt.xlabel('cluster label')
    plt.ylabel('mean expression w/ std dev')
    plt.xticks(ticks = np.arange(len(linkage_cluster_order)),labels=linkage_cluster_order)
    plt.show()

def plot_marker_on_tsne(tsne_df,expr_df,marker_name,labels, cluster_labels, offset = None, nn=False):
    
    x = np.array(tsne_df['tsne-1'])
    y = np.array(tsne_df['tsne-2'])
    if nn==False:
        #not non-neuronal
        z = np.array(expr_df.loc[marker_name,:])
        fig, ax = plt.subplots( figsize = (9,6))
        scatter = ax.scatter(x, y, c = z , cmap = 'seismic' , s = 1)
        # Create colorbar
        sm = ScalarMappable(cmap='seismic')
        sm.set_array(z)
        cbar = fig.colorbar(sm)
        cbar.set_label('Expr')
        plt.title('Log/Standerdized '+ marker_name + ' Expression')

    else:
        #non-neuronal
        #z = marker_name
        z = np.array(expr_df.loc[marker_name,:])
        fig, ax = plt.subplots( figsize = (9,6))
        scatter = ax.scatter(x, y, c = z , cmap = 'seismic' , s = 1)
        # Create colorbar
        sm = ScalarMappable(cmap='seismic')
        sm.set_array(z)
        cbar = fig.colorbar(sm)
        cbar.set_label('Expr')
        plt.title(f"{marker_name}")
        #plt.title('Log/Standerdized Nonneuronal Expression (Summed Exclude Markers)')

    arr_xy = tsne_df.drop('labels', axis = 'columns')
    labels_filtered = set(labels)
    #labels_filtered = [label for label in set(labels) if label not in drop_clusters_list]
    #print (len(set(labels_filtered)))
    for i,cl in enumerate(zip(labels_filtered,cluster_labels)):
        cluster_median = arr_xy[labels == cl[0]].median()
        #add offset
        cluster_median.iloc[0]+=offset
        #print (cluster_median)
        #print (np.array(cluster_median[0]))
        ax.annotate(text = cl[1].split(' ')[1], xy=cluster_median, fontsize=8, color='Black',
                            ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))

    plt.xticks([])
    plt.yticks([])

    plt.show()

def plot_all_markers(tsne_df, arr_xy, expr_df,GABA_marker, Vglut1_marker, Vglut2_marker, nonneuro, labels, date, label_clusters = False, savefig = False):
    x = np.array(tsne_df['tsne-1'])
    y = np.array(tsne_df['tsne-2'])
    z_gaba = np.array(expr_df.loc[GABA_marker,:])
    z_vglut1 = np.array(expr_df.loc[Vglut1_marker,:])
    z_vglut2 = np.array(expr_df.loc[Vglut2_marker,:])
    z_nn = nonneuro


    fig, ax = plt.subplots(2,2, figsize = (9,6))

    scatter = ax[0,0].scatter(x, y, c = z_gaba , cmap = 'seismic' , s = .1)
    scatter = ax[0,1].scatter(x, y, c = z_vglut1 , cmap = 'seismic' , s = .1)
    scatter = ax[-1,0].scatter(x, y, c = z_vglut2 , cmap = 'seismic' , s = .1)
    scatter = ax[-1,1].scatter(x, y, c = z_nn , cmap = 'seismic' , s = .1)

    # Add titles to subplots with italic text
    ax[0, 0].set_title(GABA_marker, fontstyle='italic')
    ax[0, 1].set_title(Vglut1_marker, fontstyle='italic')
    ax[-1, 0].set_title(Vglut2_marker, fontstyle='italic')
    ax[-1, 1].set_title('Nonneuronal', fontstyle='italic')

    if label_clusters:
        for label in set(labels):
            if label != -1:
                cluster_median = arr_xy[labels == label].median()
                #print (cluster_median)
                ax[0, 0].annotate(text = label, xy=cluster_median, fontsize=8, color='black',
                            ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))  
                ax[0, 1].annotate(text = label, xy=cluster_median, fontsize=8, color='black',
                            ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))  
                ax[-1, 0].annotate(text = label, xy=cluster_median, fontsize=8, color='black',
                            ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))  
                ax[-1, 1].annotate(text = label, xy=cluster_median, fontsize=8, color='black',
                            ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))          


    # Remove x and y ticks from all subplots
    for ax_row in ax:
        for axis in ax_row:
            axis.set_xticks([])
            axis.set_yticks([])

    plt.tight_layout()
    if savefig:
        plt.savefig('./gaba_glut_NN_doublet_marker_expression_' +date+'.png')
    plt.show()


def compute_avg_expr_per_cluster_label(df,meta_data_df):
    '''
    For level 3 analysis comparing amygdala datasets (see cell_comparison_nb.ipynb). For a given gene expression matrix and metadata with 'cluster_label' row, computes average expression of every gene for each cluster label.
    Returns dataframe of n_genes x n_cluster_labels
    '''
    avg_df = pd.DataFrame(index=df.index, columns = np.unique(meta_data_df.loc['cluster_label']))
    lbs = np.unique(meta_data_df.loc['cluster_label'])
    for label in lbs:
        #for each cluster label, store mean expr vector
        avg_df.loc[:,label] = df.loc[:,meta_data_df.loc['cluster_label']==label].mean(axis=1)
    
    return avg_df

def compute_std_expr_per_cluster_label(df,meta_data_df):
    '''
    For level 3 analysis comparing amygdala datasets (see cell_comparison_nb.ipynb). For a given gene expression matrix and metadata with 'cluster_label' row, computes standard deviation expression of every gene for each cluster label.
    Returns dataframe of n_genes x n_cluster_labels
    '''
    std_df = pd.DataFrame(index=df.index, columns = np.unique(meta_data_df.loc['cluster_label']))
    lbs = np.unique(meta_data_df.loc['cluster_label'])
    for label in lbs:
        #for each cluster label, store absolute value standard dev vector
        std_df.loc[:,label] = df.loc[:,meta_data_df.loc['cluster_label']==label].std(axis=1)
    
    return std_df

def drop_clusters(metadata_df,  linkage_cluster_order, cluster_labels = None,cluster_keepers = None, thresh_cells = 50):
    '''returns list cluster ids with 0 markers from enrichment analysis and those with gene specified in gene list. also grabs ids of any missing k/v pairs.'''
    clusters_to_drop = []
    #for k,v in cl_mg_dict.items():
        #if len(v) == 0:
            #print ('here')
            #clusters_to_drop.append(k)
        #after qualitative review, remove cluster 25 containting glut marker Slc17a6 , cluster 3 contating only Ddit4l
        #if gene_list != None:    
            #for g in gene_list:
                #if g in v:
                    #clusters_to_drop.append(k)
    #print (clusters_to_drop)
    #full_rng = np.arange(sorted(cl_mg_dict.keys())[0],sorted(cl_mg_dict.keys())[-1:][0])
    #missing_clusters = np.setdiff1d(full_rng, np.array(sorted(cl_mg_dict.keys())))
    #clusters_to_drop.extend(missing_clusters)
    for c in linkage_cluster_order:
        #print (c)
        if metadata_df.loc[:,metadata_df.loc['cluster_label',:]==c].shape[1] < thresh_cells:
            clusters_to_drop.append(c)
        elif cluster_labels != None:
            if c in cluster_labels:
                clusters_to_drop.append(c)    
    #print (clusters_to_drop)
    #drop any clusters that are in cluster_keepers (to keep them)
    if cluster_keepers != None:
        for c in cluster_keepers:
            if c in clusters_to_drop:
                clusters_to_drop.remove(c)
    return clusters_to_drop, thresh_cells

def filter_heatmap_elements(folder, cell_class, clusters_to_drop,df_marker_log_and_std,meta_data_df_plis, df_post_linkage_intra_sorted,cluster_indices,linkage_cluster_order, save_to_file = False):
    '''using clusters to drop list, filter heatmap elements, return each as [element]_filtered'''
    #linkage_cluster_order_filtered = [x for x in linkage_cluster_order if x not in clusters_to_drop]
    
    #linkage_cluster_order = cl_mg_dict.keys()
    #for c in clusters_to_drop:
    #    cl_mg_dict.pop(c)
    #linkage_cluster_order_filtered_tmp = cl_mg_dict.keys()
    linkage_cluster_order_filtered_tmp = [x for x in linkage_cluster_order if x not in clusters_to_drop]    
    #print ('lco filtered', linkage_cluster_order_filtered)
    #print ('length lco filtered', len(linkage_cluster_order_filtered))
    #only works on first element in gene list, if gene list is longer than 1 element, need to manually add "and gene_list[i]" below..
    #tg_filtered = [x for x in tg if len(x)>0 and gene_list[0] not in x]

    #tg_filtered = list(cl_mg_dict.values())
    #tgfs_filtered = get_tgfs_from_tg(tg_filtered)

    #filtered_index = [item for sublist in tg_filtered for item in sublist]
    df_marker_log_and_std_filtered = df_marker_log_and_std.copy()
    meta_data_df_plis_filtered = meta_data_df_plis.copy()
    #print ('before mask', np.unique(meta_data_df_plis_filtered.loc['cluster_label']))
    #mask = meta_data_df_plis_filtered.loc['cluster_label'].apply(lambda x: x not in clusters_to_drop)
    print (np.all(df_marker_log_and_std_filtered.columns == meta_data_df_plis_filtered.columns))
    mask = meta_data_df_plis_filtered.loc['cluster_label'].apply(lambda x: x in linkage_cluster_order_filtered_tmp)
    meta_data_df_plis_filtered = meta_data_df_plis_filtered.loc[:,mask]
    #print ('after mask', np.unique(meta_data_df_plis_filtered.loc['cluster_label']))
    df_marker_log_and_std_filtered = df_marker_log_and_std_filtered.loc[:,mask]
    print (np.all(df_marker_log_and_std_filtered.columns == meta_data_df_plis_filtered.columns))
    df_post_linkage_intra_sorted_filtered = df_post_linkage_intra_sorted.loc[:,mask]
    change_indices_filtered = get_heatmap_cluster_borders(meta_data_df_plis_filtered)
    print ('change_indices_filtered', change_indices_filtered)
    print (len(change_indices_filtered))
    #lc_ci_dict = dict(zip(linkage_cluster_order,cluster_indices))
    #cluster_indices_filtered_tmp = [v for k,v in lc_ci_dict.items() if k in linkage_cluster_order_filtered_tmp]
    ci_dict = dict(zip(linkage_cluster_order,cluster_indices))
    cluster_indices_filtered = [ci_dict[x] for x in linkage_cluster_order_filtered_tmp]
    
    #print ('len cluster_indices_filtered_tmp 0', len(cluster_indices_filtered_tmp[0]))
    
    #get updated pos
    tmp = 0
    pos_filtered_tmp = [] 
    # for idx, c in enumerate(zip(linkage_cluster_order_filtered,cluster_indices_filtered_tmp)):
    #     append mean of cell indices btwn tmp and tmp + length of cluster
    #     pos_filtered_tmp.append(np.mean(np.arange(tmp,tmp+len(c[1]))))
    #     update pointer
    #     tmp+=len(c[1])
    for i,x in enumerate(change_indices_filtered):
        pos_filtered_tmp.append(np.mean(np.arange(tmp,x)))
        #update pointer
        tmp=x
        #print (tmp)
    
    #print (pos_filtered_tmp)

    #reindex linkage cluster order, cluster labels
    meta_data_df_plis_filtered, linkage_cluster_order_filtered = update_metadata_cluster_labels(linkage_cluster_order_filtered_tmp, meta_data_df_plis_filtered)
    #update keys on cl_mg_dict
    #cl_mg_dict_filtered = dict(zip(linkage_cluster_order_filtered, list(cl_mg_dict.values())))
    #cl_mg_dict_filtered = {int(k): v for k, v in cl_mg_dict_filtered.items()}
    df_marker_log_and_std_col_filtered = pd.DataFrame(data = df_marker_log_and_std_filtered.to_numpy(), 
                                         index = df_marker_log_and_std_filtered.index,
                                        columns = list(meta_data_df_plis_filtered.loc['cluster_label',:]))
    
    #return df_marker_log_and_std_filtered, df_marker_log_and_std_col_filtered, meta_data_df_plis_filtered, pos_filtered_tmp, tg_filtered, tgfs_filtered, linkage_cluster_order_filtered,change_indices_filtered,lc_ci_dict, cluster_indices_filtered_tmp
    #if save_to_file:
        #with open(folder+cell_class+'_cl_mg_filtered_' + today +'.json', "w") as outfile: 
            #json.dump(cl_mg_dict_filtered, outfile)
        #file = cell_class + 'meta_data_df_plis_filtered_' + today
        #meta_data_df_plis_filtered.to_json(folder+file+'.json')
    
    return df_marker_log_and_std_filtered, df_marker_log_and_std_col_filtered, meta_data_df_plis_filtered, pos_filtered_tmp, df_post_linkage_intra_sorted_filtered,linkage_cluster_order_filtered, linkage_cluster_order_filtered_tmp, change_indices_filtered, cluster_indices_filtered  

def compute_fs_waterfall(marker_genes_sorted):
    '''autocomputes optimal fontsize for waterfall gene labeling on heatmap'''
    fs = (len(marker_genes_sorted)-260.5)/(-28.33332)
    fs_w = round(fs,1)
    return fs_w

def update_tsne_params(arr,labels,linkage_cluster_order_og,linkage_cluster_order,linkage_cluster_order_filtered_tmp):
    '''creates conversion dict using original linkage cluster order (og) and linkage cluster order to update labels from tsne arr'''
    conv_dict = dict(zip(linkage_cluster_order_og,linkage_cluster_order))
    arr_updated = arr.copy()
    new_labels = arr_updated['labels'].map(conv_dict)
    arr_updated['labels'] = new_labels
    labels_updated = np.array([conv_dict[x] for x in labels])
    #get filtered version
    mask = arr_updated['labels'].apply(lambda x: x in linkage_cluster_order_filtered_tmp)
    arr_updated_filtered = arr_updated.loc[mask,:]
    labels_updated_filtered = labels_updated[mask]
    return arr_updated_filtered, labels_updated_filtered

def plot_filtered_tsne(arr,labels,cluster_labels,metadata_df,drop_clusters_list,folder, plot_title,savefig=True):
    '''plot filtered tsne plot, note: number metadata columns should match arr shape. use prefiltered metadata.'''
    ids_filtered = np.sort(pd.unique(metadata_df.loc['cluster_label']))
    #arr.insert(2, 'labels',labels)
    arr_sorted = arr.sort_values(by = 'labels')
    print ('arr sortted shape', arr_sorted.shape)
    arr_filtered = arr_sorted.copy()
    arr_filtered = arr_filtered[arr_filtered['labels'].isin(ids_filtered)]
    print ('arr filtered shape', arr_filtered.shape)
    arr_xy = arr.drop('labels', axis = 'columns')
    fig,ax = plt.subplots(figsize = (15,15))
    #ax.scatter(arr_filtered['tsne-1'], arr_filtered['tsne-2'], s = 2)
    for n, grp in arr_filtered.groupby('labels'):
        ax.scatter(x = 'tsne-1',y = 'tsne-2', data=grp, label=n, s = 1)
    '''
    handles, lgd_labels = ax.get_legend_handles_labels()
    lgnd = ax.legend(handles,cluster_labels, loc = 'right', bbox_to_anchor=(1.25, 0.5))
    for handle in lgnd.legend_handles:
        handle.set_sizes([10.0])
    '''
    plt.xticks([])
    plt.yticks([])
    plt.title(plot_title)
    #print (len(set(labels)))
    labels_filtered = set(labels)
    #labels_filtered = [label for label in set(labels) if label not in drop_clusters_list]
    #print (len(set(labels_filtered)))
    for i,cl in enumerate(zip(labels_filtered,cluster_labels)):
        cluster_median = arr_xy[labels == cl[0]].median()
        #print (cluster_median)
        ax.annotate(text = cl[1].split(' ')[1], xy=cluster_median, fontsize=8, color='black',
                            ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))
    '''
    for label in set(labels):
        if label != -1:
            if label not in drop_clusters_list:
                cluster_median = gaba_arr_xy[labels == label].median()
                #print (cluster_median)
                ax.annotate(text = label, xy=cluster_median, fontsize=8, color='black',
                            ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))
    '''
    if savefig:
        plt.savefig(folder + 'filtered_tsne_' + plot_title + '_' + today + '.png', dpi = 1200, bbox_inches='tight')
    plt.show()

def plot_genes_in_cluster(df,meta_data_df,gene_list, c_label = None,ls=False):
    '''Used to check expression of markers within a cluster. Takes in a gene expression dataframe, filtered plis sorted metadata with cluster labels (e.g. 1,2,3...), and speicifed cluster label ,
    plots as simple line graph. If ls == True, assumes raw expression data input, performs log and standerization'''
    if ls:
        status_df = intialize_status_df()
        df_log_std_arr,status_df = log_and_standerdize_df(df,status_df)
        df = pd.DataFrame(data = df_log_std_arr.T, index = df.index, columns=df.columns)
    tmp = df.loc[gene_list,:]
    tmp = tmp.iloc[:,np.where(meta_data_df.loc['cluster_label']==c_label)[0]]
    tmp.T.plot(kind = 'line', xticks = [])
    plt.show()

def plot_rc_tsne(meta_data_plis_filtered_markers,arr_tsne_f, cell_class, folder, m_2_c_dict = None, savefig = False):
    df_tsne_f = pd.DataFrame(index = meta_data_plis_filtered_markers.columns, data=arr_tsne_f, columns=['tsne-1','tsne-2'])
    df_tsne_f.insert(2,'markers', meta_data_plis_filtered_markers.loc['markers',:])
    # Group by 'markers' and calculate the centroid for each group
    centroids = df_tsne_f.groupby('markers')[['tsne-1', 'tsne-2']].mean()

    # Plotting the points
    plt.figure(figsize=(10, 10))
    #plt.scatter(df_tsne_f['tsne-1'], df_tsne_f['tsne-2'], c='blue', label='Points', alpha=0.5)
    for n, grp in df_tsne_f.groupby('markers'):
        #print (n)
        if m_2_c_dict != None:
            plt.scatter(x = 'tsne-1',y = 'tsne-2', data=grp, label=n, s = 1, c = '#' + m_2_c_dict[str(n)])
        else:
            plt.scatter(x = 'tsne-1',y = 'tsne-2', data=grp, label=n, s = 1)
    # Plotting the centroids with labels
    for marker, (x, y) in centroids.iterrows():
        plt.text(x, y, marker, fontsize=10, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Plot aesthetics
    plt.xlabel('tsne-1')
    plt.ylabel('tsne-2')
    plt.title(cell_class + ' filtered tsne (recalc)')
    plt.axis('off')
    #plt.legend(['Data Points'])
    #plt.grid(True)
    if savefig:
        plt.savefig(folder + 'filtered_tsne_rc_'+cell_class+'_'+today+'.pdf')
        np.save(folder + cell_class + '_arr_tsne_f_' + today + '.npy', arr_tsne_f)
    plt.show()

def plot_rc_tsne_marker(meta_data_plis_filtered_markers,arr_tsne_f, df_marker_f, marker_name, cell_class, folder, savefig = False):
    df_tsne_f = pd.DataFrame(index = meta_data_plis_filtered_markers.columns, data=arr_tsne_f, columns=['tsne-1','tsne-2'])
    df_tsne_f.insert(2,'markers', meta_data_plis_filtered_markers.loc['markers',:])
    # Group by 'markers' and calculate the centroid for each group
    centroids = df_tsne_f.groupby('markers')[['tsne-1', 'tsne-2']].mean()

    # Plotting the points
    plt.figure(figsize=(10, 10))
    #plt.scatter(df_tsne_f['tsne-1'], df_tsne_f['tsne-2'], c='blue', label='Points', alpha=0.5)
    #for n, grp in df_tsne_f.groupby('markers'):
        #plt.scatter(x = 'tsne-1',y = 'tsne-2', data=grp, label=n, s = 1)

    z = np.array(df_marker_f.loc[marker_name,:])
    fig, ax = plt.subplots( figsize = (9,6))
    scatter = ax.scatter(df_tsne_f['tsne-1'], df_tsne_f['tsne-2'], c = z , cmap = 'seismic' , s = 1, label = marker_name)
    # Create colorbar
    #sm = ScalarMappable(cmap='seismic')
    #sm.set_array(z)
    #cbar = fig.colorbar(sm)
    #cbar.set_label('Expr')
    cbar = plt.colorbar(
                        plt.cm.ScalarMappable(cmap='seismic'),
                        ax=plt.gca())
    plt.title('Log/Standerdized '+ marker_name + ' Expression')


    # Plotting the centroids with labels
    #for marker, (x, y) in centroids.iterrows():
        #plt.text(x, y, marker, fontsize=10, ha='center', va='center', 
                #bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Plot aesthetics
    plt.xlabel('tsne-1')
    plt.ylabel('tsne-2')
    plt.title(cell_class + ' filtered tsne (recalc) ' + marker_name + ' expr')
    plt.legend()
    #plt.legend(['Data Points'])
    #plt.grid(True)
    if savefig:
        plt.savefig(folder + 'filtered_tsne_rc_marker_' + marker_name + '_' +cell_class+'_'+today+'.pdf')
        #np.save(folder + cell_class + 'arr_tsne_f_' + today + '.npy', arr_tsne_f)
    plt.show()