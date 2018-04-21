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

# >>    rename 's/\.asd$/.txt' *

# then the files are all saved as .txt and can be read into this script


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import numbers

path = "/media/joe/FDB2-2F9B/end_season_spectra/"

filelist = os.listdir(path) # pull all files out of the target directory

# create list of values in each txt file. skip 41 rows to discard header info, define comma delimiter, 
DF = pd.DataFrame()

for file in filelist:   
    DF[file] = pd.read_csv(path+file)

# rename each column with the filename minus the extension
filenames = []
for file in filelist:
    file = str(file)
    file = file[:-10]
    filenames.append(file)

#rename dataframe columns according to filenames
filenames = np.transpose(filenames)    
DF.columns = [filenames]

# Average spectra from each site
DF2 = DF.transpose()
DF2 = DF2.groupby(by=DF2.index, axis=0).apply(lambda g: g.mean() if isinstance(g.iloc[0,0],numbers.Number) else g.iloc[0])
DF = DF2.transpose()

# drop broken spectra
DF = DF.drop(['11_8_16_disc_cryoconite2'],axis=1)
DF = DF.drop(['10_8_16_bumps_2s'],axis=1)
DF['Wavelength'] = np.arange(350,2500,1)

DF.plot(figsize=(15,15)),plt.ylim(0,1)
# correct for 1000nm step 

for i in DF.columns:
    X = np.array(DF['Wavelength'][640:650])
    Y = np.array(DF[i][640:650])
    # calculate linear regression coefficients
    corr = DF[i][650] - DF[i][649]
    DF[i][650:-1] = DF[i][650:-1]-corr
    
    # interpolate over instabilities at ~1800 nm
    DF[i][1480:1600] = np.nan 
    DF[i] = DF[i].interpolate()
    DF[i][1400:1600] = pd.rolling_mean(DF[i][1400:1600],window=30)
    DF[i] = DF[i].interpolate()

    #interpolate over instabilities at 2500 nm
    DF[i][2050:2150] = np.nan 
    DF[i] = DF[i].interpolate()
    DF[i][2050:2150] = pd.rolling_mean(DF[i][2050:2150],window=30)
    DF[i] = DF[i].interpolate()

DF.plot(x=['Wavelength'],figsize=(15,15)),plt.ylim(0,1.2)
DF.to_csv('/media/joe/FDB2-2F9B/2016_end_season_HCRF.csv')