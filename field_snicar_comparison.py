#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:35:33 2018

@author: joe.cook@sheffield.ac.uk

This script is for comparing spectral albedo from field spectroscopy with 
spectral albedo preicted using a radiative transfer model. The mean spectra for
severla surface classes are calculated from a master csv file containing all the
individual spectra for each class. Then the predicted spectra are loaded in from
csv files and plotted together.

The predicted files need to be savd as csv's in the working directory before
running this script. In this implementation these files were generated using the
radiative transfer model BioSNICAR_GO in matlab and exported as csvs.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
plt.style.use('ggplot')

alb_master = pd.read_csv('/home/joe/Code/Albedo_master.csv')
hcrf_master = pd.read_csv('/home/joe/Code/HCRF_master.csv')

HAsites = ['13_7_SB2','13_7_SB4',
'14_7_S5','14_7_SB1','14_7_SB5','14_7_SB10',
'15_7_SB3',
'21_7_S3',
'21_7_SB1','21_7_SB7',
'22_7_SB4','22_7_SB5','22_7_S3','22_7_S5',
'23_7_SB3','23_7_SB4','23_7_SB5','23_7_S3','23_7_S5',
'24_7_SB2','24_7_S1',
'25_7_S1']


LAsites = ['13_7_S2','13_7_S5','13_7_SB1',
'14_7_S2','14_7_S3','14_7_SB2','14_7_SB3','14_7_SB7','14_7_SB9',
'15_7_S2','15_7_S3','15_7_SB4',
'20_7_SB1','20_7_SB3',
'21_7_S1','21_7_S5','21_7_SB2','21_7_SB4',
'22_7_SB1','22_7_SB2','22_7_SB3','22_7_S1',
'23_7_S1','23_7_S2',
'24_7_SB2','24_7_S2',
'25_7_S2','25_7_S4','25_7_S5']


SNsites = ['13_7_S4',
'14_7_S4','14_7_SB6','14_7_SB8',
'17_7_SB1','17_7_SB2']

CIsites =['13_7_S1','13_7_S3','13_7_SB3','13_7_SB5',
'14_7_S1',
'15_7_S1','15_7_S4','15_7_SB1','15_7_SB2','15_7_SB5',
'20_7_SB2',
'21_7_S2','21_7_S4','21_7_SB3','21_7_SB5','21_7_SB8',
'22_7_S2','22_7_S4',
'23_7_SB1','23_7_SB2',
'23_7_S4',
'25_7_S3']

def prepare_plot_spectra(process_spectra=True,plot_spectra = True, savefig = False):
   
    HA_alb = pd.DataFrame()
    LA_alb = pd.DataFrame()
    SN_alb = pd.DataFrame()
    CI_alb = pd.DataFrame()
    
    HA_hcrf = pd.DataFrame()
    LA_hcrf = pd.DataFrame()
    SN_hcrf = pd.DataFrame()
    CI_hcrf = pd.DataFrame()
    
    if process_spectra:
        
        for i in alb_master.columns:

            # interpolate over instabilities at ~1800 nm
            alb_master.loc[1400:1650,i] = np.nan 
            alb_master[i] = alb_master[i].interpolate()
            alb_master.loc[1400:1600,i] = alb_master.loc[1400:1600,i].rolling(window=50,center=False).mean()
            alb_master[i] = alb_master[i].interpolate()
        
            #interpolate over instabilities at 2500 nm
            alb_master.loc[2050:2150,i] = np.nan 
            alb_master[i] = alb_master[i].interpolate()
            alb_master.loc[2050:2150,i] = alb_master.loc[2050:2150,i].rolling(window=50,center=False).mean()
            alb_master[i] = alb_master[i].interpolate()
        
        for i in hcrf_master.columns:
        
            # calculate correction factor (raises NIR to meet VIS - see Painter 2011)
            corr = hcrf_master.loc[651,i] - hcrf_master.loc[650,i]
            hcrf_master.loc[651:2149,i] = hcrf_master.loc[651:2149,i]-corr  

            # interpolate over instabilities at ~1800 nm
            hcrf_master.loc[1400:1650,i] = np.nan 
            hcrf_master[i] = hcrf_master[i].interpolate()
            hcrf_master.loc[1400:1600,i] = hcrf_master.loc[1400:1600,i].rolling(window=50,center=False).mean()
            hcrf_master[i] = hcrf_master[i].interpolate()
        
            #interpolate over instabilities at 2500 nm
            hcrf_master.loc[2050:2150,i] = np.nan 
            hcrf_master[i] = hcrf_master[i].interpolate()
            hcrf_master.loc[2050:2150,i] = hcrf_master.loc[2050:2150,i].rolling(window=50,center=False).mean()
            hcrf_master[i] = hcrf_master[i].interpolate()

    for i in HAsites:
        HA_alb['HA_'+'{}'.format(i)] = alb_master[i]
        HA_hcrf['HA_'+'{}'.format(i)] = hcrf_master[i]   
        
    for i in LAsites:
        LA_alb['LA_'+'{}'.format(i)] = alb_master[i]
        LA_hcrf['LA_'+'{}'.format(i)] = hcrf_master[i]    
   
    for i in CIsites:
        CI_alb['CI_'+'{}'.format(i)] = alb_master[i]
        CI_hcrf['CI_'+'{}'.format(i)] = hcrf_master[i]
    
    LA_field = LA_alb.mean(axis=1)
    HA_field = HA_alb.mean(axis=1)
    CI_field = CI_alb.mean(axis=1)

    # Load SNICAR predicted spectra    
    LA_snicar = pd.read_csv('/home/joe/Desktop/snicar_predict_LA.csv',header= None)
    HA_snicar = pd.read_csv('/home/joe/Desktop/snicar_predict_HA.csv',header= None)
    CI_snicar = pd.read_csv('/home/joe/Desktop/snicar_predict_CI.csv',header=None)
    LA_snicar = np.ravel(np.array(LA_snicar))
    HA_snicar = np.ravel(np.array(HA_snicar))
    CI_snicar = np.ravel(np.array(CI_snicar))


    # interpolate to common wavelength range and resolution
    wavelengths = np.arange(0.305,5,0.01)
    xnew = np.arange(0.35,2.5,0.001)
    f1 = interpolate.interp1d(wavelengths,LA_snicar)
    f2 = interpolate.interp1d(wavelengths,HA_snicar)
    f3 = interpolate.interp1d(wavelengths,CI_snicar)
    LA_snicar = f1(xnew)
    HA_snicar = f2(xnew)
    CI_snicar = f3(xnew)

    if plot_spectra:
        plt.figure(figsize=(10,10))
        plt.plot(xnew,LA_field,'r',label='L$_{bio}$ field')
        plt.plot(xnew,LA_snicar,'r--',label = 'L$_{bio}$ model')
        plt.ylim(0,1),plt.xlim(0.35,1.1),plt.xlabel('Wavelength (micron)',fontsize=22)
        plt.ylabel('Albedo',fontsize=22),plt.legend(loc='best',fontsize=22),plt.grid(None)
        plt.xticks(fontsize=22), plt.yticks(fontsize=22)
        
        plt.plot(xnew,HA_field,'b',label='H$_{bio}$ field')
        plt.plot(xnew,HA_snicar,'b--',label = 'H$_{bio}$ model')
        plt.ylim(0,1),plt.xlim(0.35,1.1),plt.legend(loc='best',fontsize = 22)
        
        plt.plot(xnew,CI_field,'k',label='CI field')
        plt.plot(xnew,CI_snicar,'k--',label = 'CI model')
        plt.ylim(0,1),plt.xlim(0.35,1.1),plt.legend(loc='best',fontsize = 22)    
    
    if savefig:
        plt.savefig('field_snicar_comparison.jpg',dpi=150)

    return LA_snicar, HA_snicar, CI_snicar, LA_field, HA_field, CI_field, xnew


def calculate_errors(xnew,LA_field,LA_snicar,HA_field,HA_snicar,CI_field, CI_snicar,plot_error = True, savefig=False):
    
    LA_field = LA_field.transpose()
    LA_error_spectral = LA_snicar - LA_field
    LA_error = np.sum(abs(LA_error_spectral))

    HA_field = HA_field.transpose()
    HA_error_spectral = HA_snicar - HA_field
    HA_error = np.sum(abs(HA_error_spectral))

    CI_field = CI_field.transpose()
    CI_error_spectral = CI_snicar - CI_field
    CI_error = np.sum(abs(CI_error_spectral))
    
    print('Error for Hbio surface = ',HA_error)
    print('Error for Lbio surface = ',LA_error)
    print('Error for CI surface = ', CI_error)
    
    if plot_error:
        plt.figure(figsize=(10,10))
        plt.plot(xnew,LA_error_spectral,'r'),plt.xlim(0.35,1.1),plt.ylim(-0.2,0.2)
        plt.plot(xnew,HA_error_spectral,'b'),plt.xlim(0.35,1.1),plt.ylim(-0.2,0.2),plt.grid(None)
        plt.plot(xnew,CI_error_spectral,'k'),plt.xlim(0.35,1.1),plt.ylim(-0.2,0.2)
        plt.xlabel('Wavelength (microns)',fontsize=22),plt.ylabel('Absolute Error (dimensionless)',fontsize = 22)
        plt.xticks(fontsize=22),plt.yticks(fontsize=22)
    if savefig:
        plt.savefig('field_snicar_comparison_error.jpg',dpi=150)
        
    return LA_error, LA_error_spectral, HA_error, HA_error_spectral, CI_error, CI_error_spectral



######################## CALL FUNCTIONS ######################################
###############################################################################
LA_snicar, HA_snicar, CI_snicar, LA_field, HA_field, CI_field, xnew = prepare_plot_spectra(process_spectra=True,plot_spectra = True, savefig = True)
calculate_errors(xnew,LA_field,LA_snicar,HA_field,HA_snicar,CI_field,CI_snicar,plot_error = True, savefig = True)