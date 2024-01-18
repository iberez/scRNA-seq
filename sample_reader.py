# sample reader to read samples from /data/runs/samples
# and build each sample into a matrix

# Isaac Berez
# 14.01.23

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

def process_meta_data(analysis_dir, dimorph_sample_file, output_folder=None):
    '''process meta data from ods and return sample id list and full meta data dictionary'''
    #read in dimorph sample file
    dimorph_df = read_ods(analysis_dir+dimorph_sample_file)
    #fill in missing data types
    dimorph_df_1 = dimorph_df.rename(columns={"Group":"Group:string", 
                                              "Avesizebp_cDNAlib":"Avesizebp_cDNAlib:float64",
                                              "Date":"Date:string",
                                              "cDNAul":"cDNAul:float64",
                                              "LIbConstructionComment": "LIbConstructionComment:string"
                                             })
    #create lists of meta data header and data types by splitting at colon in each column name
    mdata_header = []
    dtype = []
    for d in dimorph_df_1:
        mdata_header.append(d.split(":")[0])
        dtype.append(d.split(":")[1])
    #get and return sample id
    samples = [x for x in list(dimorph_df['SampleID:string']) if x != None]
    #capitilize to match folder structure
    samples = [x.upper() for x in samples]
    #create meta data dictionary from sample ids
    meta_data_dict = dict.fromkeys(samples)
    #initialize headers for each sample
    mdata_header_dict = dict.fromkeys(mdata_header)
    #populate meta_data_dict with all meta data
    for i in range(len(meta_data_dict.keys())):
        for key,value in zip(mdata_header_dict.keys(),list(dimorph_df_1.loc[i])):
            mdata_header_dict.update({key:value})
        meta_data_dict.update({list(meta_data_dict.keys())[i]:deepcopy(mdata_header_dict)})
    
    return samples, meta_data_dict

def find_latest_sample_file(sample_dir, samples, sample_ind):
    '''finds latest file denoted by YYMMDD in filename'''
    
    path = os.path.join(sample_dir, samples[sample_ind])
    #print ('path: ', path)
    #print ('sample ind: ',samples[sample_ind])
    files = os.listdir(path)
    print ('finding latest sample files from: ', files)

    date_list = []
    #isolate date YYMMDD
    for file in files:
        #print (file)
        if 'out' in file:
            date = re.search("(\d{2}\d{2}\d{2})", file)
        #print((date[0]))
            date_list.append(date[0])
    #print (date_list)
    
    #add 20 in front to make YYYYMMDD, convert to datetime object, then use max() to get latest date
    new_date_list = []
    for d in date_list:
        d = '20' + d
        dt = datetime.strptime(d,'%Y%m%d')
        new_date_list.append(dt)
    #print ('datetime dates:' , new_date_list)
    latest_date = max(new_date_list)
    #print ('latest date: ', latest_date)

    #convert latest date back to old file naming convention 
    latest_date = str(latest_date)
    latest_date_key = str(latest_date[2:-9])
    latest_date_key = latest_date_key[:2]+latest_date_key[3:5]+latest_date_key[6:8]
    #print ('latest date key: ',latest_date_key)

    #match latest date key with file
    for file in files:
        if latest_date_key in file:
            if 'out' in file:
                print ('found latest file: ', file)
                return file

def construct_sample_matrix(sample_dir, samples, sample_ind):
    '''construcs a single combined sample matrix from matrix, gene, feature files'''
    
    os.chdir(sample_dir)
    cwd = os.getcwd()
    #print ('switched to sample dir: ', cwd)
    
    path = os.path.join(cwd,samples[sample_ind])
    
    #check for latest file
    file = find_latest_sample_file(sample_dir,samples,sample_ind)
    
    #for loop would go here to loop over samples
    dir = str((glob.glob(path+
                     '/'+file+
                     '/'+'outs'+
                     '/'+ 'filtered_feature_bc_matrix'))).replace('[','').replace(']','')[1:-1]

    #read matrix
    #print ('reading matrix file from: ', dir)
    m = mmread(dir + '/' + 'matrix.mtx')
    m_arr = m.toarray()

    #read barcodes file and store as cells
    barcodes = pd.read_csv(dir+'/'+'barcodes.tsv', sep='\t',header = None)
    cells = barcodes.iloc[:,0]

    #append the sample id to each cell
    cells = barcodes.iloc[:,0] + '_' + samples[sample_ind]
    cells = cells.values.reshape(cells.shape[0],1)

    #read features file and store as gene labels
    features = pd.read_csv(dir+'/'+'features.tsv', sep='\t', header=None)
    gene_labels = features.iloc[:,1]
    gene_labels = gene_labels.values.reshape(gene_labels.shape[0],1)

    #check for duplicate genes
    #print('unique values in gene labels: ', len(np.unique(gene_labels)))
    #print('total length gene labels: ', len(gene_labels))
    #print('# duplicates: ', len(gene_labels)-len(np.unique(gene_labels)))

    #combine matrix, cells, genes into single dataframe
    sample_df = pd.DataFrame(data = m_arr,
                             index = gene_labels,
                             columns = cells)
    #add duplicate rows together
    #print('sample df size before add duplicate gene rows together: ', sample_df.shape)
    #print('adding duplicate rows...')
    sample_df_summed = sample_df.groupby(level=0).sum()
    print('sample df after adding/removing duplicate rows: ', sample_df_summed.shape)
    return sample_df_summed