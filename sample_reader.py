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

def process_meta_data(analysis_dir, dimorph_sample_file, output_folder=None):
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

