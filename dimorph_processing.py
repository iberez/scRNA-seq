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

def create_meta_data_df(meta_data, df):
    ''' creates a new meta data df with dim (# meta data features, e.g. serial number, x # cells from big data dataframe) '''
    meta_data_df = pd.DataFrame(index = meta_data.index, columns = df.columns)
    for i,v in enumerate(meta_data.keys()):
        if len(meta_data_df.columns[meta_data_df.columns.str.contains(meta_data.keys()[i])])>0:
            meta_data_df.loc[meta_data_df.index[:],
                            meta_data_df.columns[meta_data_df.columns.str.contains(meta_data.keys()[i])]] = meta_data.loc[meta_data.index[:], meta_data.columns[i]]
    return meta_data_df

def load_data(metadata_file, bigdata_file):
    ''' reads in metadata json, big data (gene expression) feather file, returns dataframe versions for each and a boolean version of the gene expression matrix'''
    meta_data = pd.read_json(metadata_file)
    dimorph_df = pd.read_feather(bigdata_file)
    dimorph_df.set_index('gene', inplace=True)
    meta_data_df = create_meta_data_df(meta_data=meta_data, df = dimorph_df)
    meta_data_df = meta_data_df.fillna('')
    dimorph_df = dimorph_df.loc[:,meta_data_df.loc['Strain'].str.contains('Cntnp')==False]
    meta_data_df = meta_data_df.loc[:,meta_data_df.loc['Strain'].str.contains('Cntnp')==False]
    # create boolean version of the dataframe, where any expression >0 = 1,
    dimorph_df_bool = dimorph_df.mask(dimorph_df>0, other = 1)
    
    return meta_data_df,dimorph_df,dimorph_df_bool

def cell_exclusion(threshold_m,threshold_g, df_bool, meta_data_df, df):
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
    return df_updated, df_bool_updated, meta_data_df_updated

def gene_exclusion(num_cell_lwr_bound, percent_cell_upper_bound, df, df_bool, meta_data_df):
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
    return df_updated, df_bool_updated, meta_data_df_updated

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