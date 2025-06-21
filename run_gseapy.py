
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dimorph_processing as dp
import gseapy as gp
import scanpy as sc
import json 
import time
from gseapy import barplot, dotplot, heatmap

def enrich(gene_list_folder,cell_type,group,enrichr_folder,savefig = True):
    '''uses gseapy to run enrichment analysis on gene list.
    saves results to enrichment folder.'''
    #main dir
    main_dir  = enrichr_folder + cell_type + '/'
    os.makedirs(main_dir, exist_ok=True) 
    #read in gene list
    fn = cell_type +'_' + group + '_genes.txt'
    gene_list = []
    with open(gene_list_folder + fn, 'r') as fh:
        for g in fh:
            #use uppercase per Gseapy documentation
            g = g.upper()
            gene_list.append(g[:-1])
    print (f'number of genes:{len(gene_list)}')
    #most relevant libraries to use for enrichment analysis
    library_list = ['Allen_Brain_Atlas_10x_scRNA_2021',
                    'GO_Biological_Process_2025',
                    'MGI_Mammalian_Phenotype_Level_4_2024',
                    'KEGG_2019_Mouse']
    print (library_list)
    #run enrichr
    enr = gp.enrichr(gene_list=gene_list,  
                    gene_sets=library_list, 
                    organism = 'mouse',
                    outdir=main_dir + cell_type + '_enrichr_test_' + group)
    #dot plot
    #top 5 terms of each gene_set, ranked by adjusted p-value
    # categorical scatterplot
    ax = dotplot(enr.results,
    column="Adjusted P-value",
    x='Gene_set', # set x axis, so you could do a multi-sample/library␣
    size=10,
    top_term=5,
    figsize=(3,10),
    title = "enrichr results",
    xticklabels_rot=45, # rotate xtick labels
    show_ring=False, # set to False to revmove outer ring
    marker='o',
    )
    #plt.tight_layout()
    if savefig:
        plt.savefig(main_dir + cell_type + '_enrichr_test_' + group + '/' + cell_type + '_enrichr_dotplot' + '_' + group + '.pdf',bbox_inches='tight')
        plt.close()

    # categorical scatterplot
    ax = barplot(enr.results,
    column="Adjusted P-value",
    group='Gene_set', # set group, so you could do a multi-sample/library␣
    size=10,
    top_term=5,
    figsize=(3,5),
    #color=['darkred', 'darkblue'] # set colors for group
    color = {'Allen_Brain_Atlas_10x_scRNA_2021': 'salmon', 
            'GO_Biological_Process_2025':'darkblue',
            'MGI_Mammalian_Phenotype_Level_4_2024':'lightgreen'}
    )
    if savefig:
        plt.savefig(main_dir + cell_type + '_enrichr_test_' + group + '/' + cell_type + '_enrichr_barplot' + '_' + group + '.pdf',bbox_inches='tight')
        plt.close()

    return None