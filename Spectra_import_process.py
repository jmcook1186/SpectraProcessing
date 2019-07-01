# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:59:17 2017

@author: Joseph Cook, University of Sheffield
"""
# Code to import spectra from ASCII text files output by ASD's ViewSpec
# software, extract the values and collate into a pandas dataframe.
# The columns are named according to the original asd filenames minus the
# extensions. The metadata and header info is discarded but can be obtained
# by viewing the original text file.

# Prior to running this code, the asd files have been exported as ascii files 
# using ViewSpec. This saves the individual files as asd.txt. These are then
# renamed in a batch using the command line:

# navigate to folder using cd

# cd /path/folder/
# rename “s/.asd.txt/.txt/g”**-v

# then the files are all saved as .txt and can be read into this script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import numbers

path = "/media/joe/TOSHIBA/BLackandBloom/Field_Data/2018/asd/2018_HCRF_txtfiles/"

filelist = os.listdir(path) # pull all files out of the target directory
DF = pd.DataFrame() # create empty dataframe

for file in filelist:
    try:
        DF[file] = np.squeeze(np.array(pd.read_csv(path+file, header = None))) # for all files in folder, add to DF column
    except:
        print('cannot append {}'.format(file))

# rename each column with the filename minus the extension
filenames = []
for file in filelist:
    file = str(file)
    file = file[:-10]
    filenames.append(file)

#rename dataframe columns according to filenames
filenames = np.transpose(filenames)    
DF.columns = [filenames]

# Average spectra from each site, grouped by equal filenames
DF2 = DF.transpose()
DF2 = DF2.groupby(by=DF2.index, axis=0).apply(lambda g: g.mean() if isinstance(g.iloc[0,0],numbers.Number) else g.iloc[0])
DF2 = DF2.transpose()

DF = DF2.rolling(window=20,center=False).mean()
DF[0:20] = DF2[0:20]
# # drop broken spectra
# DF = DF.drop(['11_8_16_disc_cryoconite2'],axis=1)
# DF = DF.drop(['10_8_16_bumps_2s'],axis=1)
# DF = DF.drop(['19_7_2016_med_alg01'],axis=1)

# correct for 1000nm step and instabilities due to water vapour
for i in DF.columns:

    # calculate correction factor (raises NIR to meet VIS - see Painter 2011)
    corr = DF.loc[650,i] - DF.loc[649,i]
    DF.loc[650:2149,i] = DF.loc[650:2149,i]-corr  

    # interpolate over instabilities at ~1800 nm
    DF.loc[1400:1650,i] = np.nan
    DF[i] = DF[i].interpolate()
    DF.loc[1400:1650,i] = DF.loc[1400:1650,i].rolling(window=50,center=False).mean()
    DF[i] = DF[i].interpolate()

    #interpolate over instabilities at 2500 nm
    DF.loc[2050:2150,i] = np.nan 
    DF[i] = DF[i].interpolate()
    DF.loc[2050:2150,i] = DF.loc[2050:2150,i].rolling(window=50,center=False).mean()
    DF[i] = DF[i].interpolate()

# add wavelength column and plot with wavelength on x-axis
DF['Wavelength'] = np.arange(350,2501,1)
DF.plot(x='Wavelength'],figsize=(15,15),legend=False),plt.ylim(0,1.2)

DF.to_csv('/media/joe/TOSHIBA/BlackandBloom/Field_Data/2018/asd/2018_MASTER_HCRF.csv')



