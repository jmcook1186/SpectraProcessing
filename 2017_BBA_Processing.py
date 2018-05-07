# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:59:17 2017

@author: Joseph Cook, University of Sheffield
"""
# This code is used to extract the broadband albedo value of each site from a
# .csv master file generated as per 'Spectra_import_process.py' and averaging out
# replicate measurements for individual sites, then grouping according to
# surface type, finally returning separate dataframes of albedo for each
# surface type (also saved to csv).

# Prior to running this script, the csv file 'BBA.txt' has been produced by 
# following these steps:

# 1. The raw .asd files were renamed to .txt using the linux terminal
# 2. The .txt files were read into pandas and processed according to
# the script 'spectra_import_process.py'
# 3. The downwelling irradiance was used to determine BBA according to the
# equation: BBA = (f(albedo,lambda) * f(incoming,lamda)) / f(incoming,lambda)
# where lambda refers to wavelength, f is the integral over the solar spectrum,
#incoming is the reading from upwards-looking sensor, albedo is spectral albedo


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import numbers

master = pd.read_csv('/media/joe/C38C-3252/BlackandBloom/Field_Data/JC2017/Albedo/BBA.txt')

def create_dataset(master):
    # cycle through txt file, strip out site name and mean all columns where
    # site names match
    for i in np.arange(0,len(master['RowName']),1):
        p = str(master.loc[i,'RowName']) 
        sep = '_A' # split name at str '_A'
        remainder = p.split(sep,1)[0] # keep characters before the separator 
        master.loc[i,'RowName'] = remainder # rename the rows with the amended names
    
    #separate rowname and data into separate arrays
    A = master['RowName']
    B = master['BBA_list']
    
    # transpose data into columns not rows
    BBA = pd.DataFrame(B)
    BBA = BBA.transpose()
    BBA.columns = A # name columns according to names in A

    # Average spectra from each site, grouped by equal filenames
    DF2 = BBA.transpose()
    DF2 = DF2.groupby(by=DF2.index, axis=0).apply(lambda g: g.mean() if isinstance(g.iloc[0,0],numbers.Number) else g.iloc[0])
    BBA = DF2[:-1]

    
    DF.to_csv('/media/joe/C38C-3252/2017_BBAs.csv') # save as csv
    
    return BBA

def split_by_site(BBA):
    
    #identifiers for each site/date collected into surface type groups
    HAsites = ['13_7_SB2','13_7_SB4','13_7_SB6',
    '14_7_S5','14_7_SB1','14_7_SB4','14_7_SB5','14_7_SB10',
    '15_7_SB3',
    '21_7_S3',
    '21_7_SB1','21_7_SB7',
    '22_7_SB4','22_7_SB5','22_7_S3','22_7_S5','22_7_SB5',
    '23_7_SB3','23_7_SB4','23_7_SB5','23_7_S3','23_7_S5',
    '24_7_SB2','24_7_S1',
    '25_7_S1']
        
    LAsites = ['13_7_S2','13_7_S5','13_7_SB1',
    '14_7_S3','14_7_SB2','14_7_SB3','14_7_SB7','14_7_SB9',
    '15_7_S2','15_7_S3','15_7_SB4','15_7_S4','15_7_S5',
    '20_7_SB1','20_7_SB3',
    '21_7_S1','21_7_S5','21_7_SB2','21_7_SB4',
    '22_7_SB1','22_7_SB2','22_7_SB3','22_7_S1',
    '23_7_S1','23_7_S2',
    '24_7_SB2','24_7_S2',
    '25_7_S2','25_7_S4','25_7_S5']
        
    Snowsites = ['13_7_S4',
    '14_7_S4','14_7_SB6','14_7_SB8',
    '17_7_SB1','17_7_SB2','20_7_SB4']
    
    CIsites =['13_7_S1','13_7_S3','13_7_SB3','13_7_SB5',
    '14_7_S1','14_7_S3',
    '15_7_S1','15_7_S4','15_7_SB1','15_7_SB2','15_7_SB5',
    '20_7_SB2',
    '21_7_S2','21_7_S4','21_7_SB3','21_7_SB5','21_7_SB8', '21_7_SB9',
    '22_7_S2','22_7_S4',
    '23_7_SB1','23_7_SB2',
    '23_7_S4']
    
    BBA = BBA.transpose()

    BBA_HA = pd.DataFrame()
    BBA_LA = pd.DataFrame()
    BBA_CI = pd.DataFrame()
    BBA_Snow = pd.DataFrame()

    # Create dataframes for ML algorithm
    for i in HAsites:
        BBA_HA[i] = BBA[i]
    for i in LAsites:
        BBA_LA[i] = BBA[i]
    for i in CIsites:
        BBA_CI[i] = BBA[i]
    for i in Snowsites:
        BBA_Snow[i] = BBA[i]

# NB The following sites have been ignored: 21_7_SB8, 22_7_SB6, 25_7_S3    
    
    BBA_HA = BBA_HA.transpose()
    BBA_LA = BBA_LA.transpose()
    BBA_CI = BBA_CI.transpose()
    BBA_Snow = BBA_Snow.transpose()        
        
    return HAsites, LAsites, CIsites, Snowsites, BBA_HA, BBA_LA, BBA_CI, BBA_Snow


BBA = create_dataset(master)
BBA_HA,BBA_LA,BBA_CI,BBA_Snow = split_by_site(BBA)

