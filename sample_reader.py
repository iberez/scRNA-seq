# sample reader to read samples from /data/runs/samples
# and build each sample into a matrix

# Isaac Berez
# 14.01.23

from scipy.io import mmread
import os
import glob

sample_dir = '/data/runs/samples/'
dimorph_samples = ['10X54_1', 
                   '10X54_2',
                   '10x98_2',
                   '10x98_3',
                   '10X51_2',
                   '10X51_1',
                   '10X52_1',
                   '10X52_2',
                   '10X51_3',
                   '10X51_4',
                   '10X52_3',
                   '10X52_4',
                   '10X35_1',
                   '10X35_2',
                   '10X38_1',
                   '10X38_2',
                   '10X36_1',
                   '10X36_2',
                   '10X37_1',
                   '10X37_2']

os.chdir(sample_dir)
cwd = os.getcwd()
#print (cwd)
dir = str((glob.glob(cwd+'/'+dimorph_samples[0]+
                 '/'+'*out*'+'/'+'outs'+
                 '/'+'filtered_feature_bc_matrix'))).replace('[','').replace(']','')[1:-1]

#print(cwd)
#print (type(dir))
#print (os.listdir(dir))

m = mmread(dir + '/' + 'matrix.mtx')
m_arr = m.toarray()
print (m_arr)
print (m_arr.shape)