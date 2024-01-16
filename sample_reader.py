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
    print ('path: ', path)
    print ('sample ind: ',samples[sample_ind])
    files = os.listdir(path)
    print ('files', files)

    date_list = []
    #isolate date YYMMDD
    for file in files:
        #print (file)
        if 'out' in file:
            date = re.search("(\d{2}\d{2}\d{2})", file)
        #print((date[0]))
            date_list.append(date[0])
    print (date_list)
    
    #add 20 in front to make YYYYMMDD, convert to datetime object, then use max() to get latest date
    new_date_list = []
    for d in date_list:
        d = '20' + d
        dt = datetime.strptime(d,'%Y%m%d')
        new_date_list.append(dt)
    print ('datetime dates:' , new_date_list)
    latest_date = max(new_date_list)
    print ('latest date: ', latest_date)

    #convert latest date back to old file naming convention 
    latest_date = str(latest_date)
    latest_date_key = str(latest_date[2:-9])
    latest_date_key = latest_date_key[:2]+latest_date_key[3:5]+latest_date_key[6:8]
    print ('latest date key: ',latest_date_key)

    #match latest date key with file
    for file in files:
        if latest_date_key in file:
            if 'out' in file:
                print ('latest file: ', file)
                return file