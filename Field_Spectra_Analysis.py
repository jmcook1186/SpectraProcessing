#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 09:05:01 2018

@author: joe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy as sci
import scipy.stats as stats
import statsmodels as sm
import math
plt.style.use('ggplot')

################ IMPORT CSVs FOR ALBEDO AND HCRF DATA #########################
########## DEFINE SITES TO INCLUDE IN EACH IMPURITY LOADING CLASS #############

WL = np.arange(350,2500,1)

alb_master = pd.read_csv('/home/joe/Code/Albedo_master.csv')
hcrf_master = pd.read_csv('/home/joe/Code/HCRF_master.csv')

HAsites = ['13_7_SB2','13_7_SB4',
'14_7_S5','14_7_SB1','14_7_SB5','14_7_SB10',
'15_7_SB3',
'21_7_S3',
'21_7_SB1','21_7_SB7','21_7_SB8',
'22_7_SB4','22_7_SB5','22_7_S3','22_7_S5',
'23_7_SB3','23_7_SB4','23_7_SB5','23_7_S3','23_7_S5',
'24_7_SB2','24_7_S1',
'25_7_S1']


LAsites = ['13_7_S2','13_7_S5','13_7_SB1',
'14_7_S2','14_7_S3','14_7_SB2','14_7_SB3','14_7_SB7','14_7_SB9',
'15_7_S2','15_7_S3','15_7_SB4','15_7_SB1','15_7_SB2',
'20_7_SB1','20_7_SB3','20_7_SB2',
'21_7_S1','21_7_S5','21_7_SB2','21_7_SB4','21_7_SB5',
'22_7_SB1','22_7_SB2','22_7_SB3','22_7_S1',
'23_7_S1','23_7_S2',
'24_7_SB2','24_7_S2',
'25_7_S2','25_7_S4','25_7_S5']


SNsites = ['13_7_S4',
'14_7_S4','14_7_SB6','14_7_SB8',
'17_7_SB1','17_7_SB2']

CIsites =['13_7_S1','13_7_S3','13_7_SB3','13_7_SB5',
'14_7_S1',
'15_7_S1','15_7_S4','15_7_SB5',
'21_7_S2','21_7_S4','21_7_SB3',
'22_7_S2','22_7_S4',
'23_7_SB1','23_7_SB2',
'23_7_S4',
'25_7_S3']


def create_plot_alb_hcrf(process_spectra = True, plots = 2, savefiles = False):
   
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
        
    for i in SNsites:
        SN_alb['SN_'+'{}'.format(i)] = alb_master[i]
        SN_hcrf['SN_'+'{}'.format(i)] = hcrf_master[i]    

    for i in CIsites:
        CI_alb['CI_'+'{}'.format(i)] = alb_master[i]   
        CI_hcrf['CI_'+'{}'.format(i)] = hcrf_master[i]  


    if plots == 1:   
     
        plt.figure(1)
        plt.grid(None)
        plt.xlim(350,2000)
        plt.ylim(0,1)
        plt.plot(WL,HA_alb,color='g',label = 'Heavy Algae')
        plt.plot(WL,LA_alb,color='r',label = 'Light Algae')
        plt.plot(WL,CI_alb,color='b',label = 'Clean Ice')
        plt.plot(WL,SN_alb,color='k', label = 'Clean SN')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Albedo')
        
        plt.figure(2)
        plt.grid(None)
        plt.xlim(350,2000)
        plt.ylim(0,1)
        plt.plot(WL,HA_hcrf,color='g',label = 'Heavy Algae')
        plt.plot(WL,LA_hcrf,color='r',label = 'Light Algae')
        plt.plot(WL,CI_hcrf,color='b',label = 'Clean Ice')
        plt.plot(WL,SN_hcrf,color='k', label = 'Clean SN')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('HCRF')
    
    if plots == 2:
        
        HA_std = HA_alb.std(axis=1)
        HA = HA_alb.mean(axis=1)
        HAavplus = HA + HA_std
        HAavminus = HA - HA_std
        
        HA_hcrf_std = HA_hcrf.std(axis=1)
        HA_hcrf_av = HA_hcrf.mean(axis=1)
        HA_hcrf_avplus = HA_hcrf_av + HA_hcrf_std
        HA_hcrf_avminus = HA_hcrf_av - HA_hcrf_std
        
        LA_std = LA_alb.std(axis=1)
        LA = LA_alb.mean(axis=1)
        LAavplus = LA + LA_std
        LAavminus = LA - LA_std
        
        LA_hcrf_std = LA_hcrf.std(axis=1)
        LA_hcrf_av = LA_hcrf.mean(axis=1)
        LA_hcrf_avplus = LA_hcrf_av + LA_hcrf_std
        LA_hcrf_avminus = LA_hcrf_av - LA_hcrf_std
        
        CI_std = CI_alb.std(axis=1)
        CI = CI_alb.mean(axis=1)
        CIavplus = CI + CI_std
        CIavminus = CI - CI_std
        
        CI_hcrf_std = CI_hcrf.std(axis=1)
        CI_hcrf_av = CI_hcrf.mean(axis=1)
        CI_hcrf_avplus = CI_hcrf_av + CI_hcrf_std
        CI_hcrf_avminus = CI_hcrf_av - CI_hcrf_std
        
        SN_std = SN_alb.std(axis=1)
        SN = SN_alb.mean(axis=1)
        SNavplus = SN + SN_std
        SNavminus = SN - SN_std
        
        SN_hcrf_std = SN_hcrf.std(axis=1)
        SN_hcrf_av = SN_hcrf.mean(axis=1)
        SN_hcrf_avplus = SN_hcrf_av + SN_hcrf_std
        SN_hcrf_avminus = SN_hcrf_av - SN_hcrf_std
        
        
        fig = plt.figure(figsize=(24,12))
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        ax1.set_ylim(0,1),ax1.set_xlim(350,2200)
        ax2.set_ylim(0,1),ax2.set_xlim(350,2200)

        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax2.tick_params(axis='both', which='major', labelsize=22)

        plt.text(-1600,0.95,'dashed line = mean,',fontsize='22')
        plt.text(-1600,0.90, 'shade = 1 $\sigma$',fontsize='22')
        
        ax1.plot(WL,HA,'g--',label='H$_{bio}$')
        ax1.fill_between(WL,HAavplus,HAavminus,facecolor='green',alpha=0.1)
        
        ax1.plot(WL,LA,'r--',label='L$_{bio}$')
        ax1.fill_between(WL,LAavplus,LAavminus,facecolor='red',alpha=0.1)
        
        ax1.plot(WL,CI,'b--',label='CI')
        ax1.fill_between(WL,CIavplus,CIavminus,facecolor='blue',alpha=0.1)
        
        ax1.plot(WL,SN,'k--',label='SN')
        ax1.fill_between(WL,SNavplus,SNavminus,facecolor='black',alpha=0.1)
        ax1.legend(loc='upper right',fontsize=22,facecolor='w')
        
        ax2.plot(WL,HA_hcrf_av,'g--',label='H$_{bio}$')
        ax2.fill_between(WL,HA_hcrf_avplus,HA_hcrf_avminus,facecolor='green',alpha=0.2)
        
        ax2.plot(WL,LA_hcrf_av,'r--',label='L$_{bio}$')
        ax2.fill_between(WL,LA_hcrf_avplus,LA_hcrf_avminus,facecolor='red',alpha=0.2)
        
        ax2.plot(WL,CI_hcrf_av,'b--',label='Clean Ice')
        ax2.fill_between(WL,CI_hcrf_avplus,CI_hcrf_avminus,facecolor='blue',alpha=0.2)
        
        ax2.plot(WL,SN_hcrf_av,'k--',label='Snow')
        ax2.fill_between(WL,SN_hcrf_avplus,SN_hcrf_avminus,facecolor='black',alpha=0.2)
        
        
        ax1.set_ylim(0,1), ax1.set_xlim(350,2000),ax1.set_xlabel('Wavelength (nm)',fontsize='22'),ax1.set_ylabel('Albedo',fontsize='22')
        ax2.set_ylim(0,1), ax2.set_xlim(350,2000),ax2.set_xlabel('Wavelength (nm)',fontsize='22'),ax2.set_ylabel('HCRF',fontsize='22')
        ax2.legend(loc='upper right',fontsize='22', facecolor='w')
        ax1.grid(None),ax2.grid(None)

        ax1.set_facecolor('w'), ax2.set_facecolor('w')

        plt.savefig('/home/joe/Desktop/Albedo_HCRF.jpg',dpi=150)

        plt.show()
    
    if savefiles:

        HA_alb.to_csv('Hbio_albedo.csv')
        LA_alb.to_csv('Lbio_albedo.csv')
        CI_alb.to_csv('CI_albedo.csv')
        SN_alb.to_csv('SN_albedo.csv')
        HA_hcrf.to_csv('Hbio_hcrf.csv')
        LA_hcrf.to_csv('Lbio_hcrf.csv')
        CI_hcrf.to_csv('CI_hcrf.csv')
        SN_hcrf.to_csv('SN_hcrf.csv')                
        
        
    return HA_alb, HA_hcrf, LA_alb, LA_hcrf, CI_alb, CI_hcrf, SN_alb, SN_hcrf



def albedo_hcrf_ANOVA(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf,plots=True,savefiles=False):
    
    # In this part of the script a one-way ANOVA is used to test whether there
# is a significant difference between the means of the four groups, then a
# pairwise t-test with Bonferonni correction is applied to each pair of
# groups per unit wavelength to see which groups differ at which wavelengths.

# First change statistics dataframes into numpy arrays, then loop through each
# wavelength and determine ANOVA p and F statistics. Plot both in separate 
# subplots 

    alb_stat_list = []
    alb_p_list = []
    hcrf_stat_list = []
    hcrf_p_list = []
    
    for i in np.arange(0,2150,1):    
        
        alb_stat,alb_p = stats.mstats.f_oneway(np.array(HA_alb.iloc[i]),np.array(LA_alb.iloc[i]),np.array(CI_alb.iloc[i]),np.array(SN_alb.iloc[i]))
        
        hcrf_stat,hcrf_p = stats.mstats.f_oneway(np.array(HA_hcrf.iloc[i]),np.array(LA_hcrf.iloc[i]),np.array(CI_hcrf.iloc[i]),np.array(SN_hcrf.iloc[i]))
        
        alb_stat_list.append(alb_stat)
        alb_p_list.append(alb_p)
        hcrf_stat_list.append(hcrf_stat)
        hcrf_p_list.append(hcrf_p)
    
    
    if plots:
        
        fig = plt.figure(figsize=(16, 8))
        
        plt.title('ANOVA results for Albedo')
        sub1 = plt.subplot(1, 2, 1)
        sub1.set_xlim(350,1800)
        sub1.set_ylabel('one-way ANOVA p-value',fontsize=22)
        
        sub2 = plt.subplot(1, 2, 2)
        sub2.set_xlim(350,1800)
        sub2.set_ylabel('one-way ANOVA F-statistic',fontsize=22)
        sub1.set_xlabel('Wavelength (nm)',fontsize=22)
        sub2.yaxis.tick_right()
        sub2.yaxis.set_label_position("right")
        sub2.set_xlabel('Wavelength (nm)',fontsize=22)
        
        sub1.plot(WL[0:1450],alb_p_list[0:1450],label='p-value')
        sub1.hlines(0.05,350,1800,color='k',linestyle='dashed')
        
        sub2.plot(WL,alb_stat_list,label='F-statistic')
        sub1.tick_params(axis='both', which='major', labelsize=22)
        sub2.tick_params(axis='both', which='major', labelsize=22)
        
        plt.savefig('/home/joe/Desktop/ANOVA_albedo.jpg',dpi=150)
        
        # HCRF
        fig2 = plt.figure(figsize=(8, 6))
        plt.title('ANOVA results for HCRF')
        sub1 = plt.subplot(1, 2, 1)
        sub1.set_xlim(350,1800)
        sub1.set_ylabel('one-way ANOVA p-value',fontsize=22)
        sub1.set_xlabel('Wavelength (nm)',fontsize=22)
        
        sub2 = plt.subplot(1, 2, 2)
        sub2.set_xlim(350,1800)
        sub2.set_ylabel('one-way ANOVA F-statistic',fontsize=22)
        sub2.yaxis.tick_right()
        sub2.yaxis.set_label_position("right")
        sub1.set_xlabel('Wavelength (nm)')
        sub1.plot(WL[0:1450],hcrf_p_list[0:1450],label='p-value')
        sub1.hlines(0.05,350,1800,color='k',linestyle='dashed')
        sub1.text(1600, -0.006, 'Wavelength (nm)', fontsize=11)
        
        sub2.plot(WL,hcrf_stat_list,label='F-statistic')
        sub1.grid(None),sub2.grid(None)
        sub1.tick_params(axis='both', which='major', labelsize=22)
        sub2.tick_params(axis='both', which='major', labelsize=22)
        
    if savefiles:
        
        np.savetxt("alb_stat_list.csv", alb_stat_list, delimiter=",")
        np.savetxt("alb_p_list.csv", alb_stat_list, delimiter=",")
        np.savetxt("hcrf_stat_list.csv", hcrf_stat_list, delimiter=",")
        np.savetxt("hcrf_p_list.csv", hcrf_p_list, delimiter=",")

        
    return alb_stat_list, alb_p_list, hcrf_stat_list, hcrf_p_list
 
    


def alb_hcrf_posthoc_tests(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf,plots=True):    
   
    # run pairwise t-tests and Bonfferoni correction per unit wavelength.
    # Append t and p per unit wavelength to lists for each test (group x vs group y)
    # append wavelengths where p is significant and t < tcrit to separate lists

    alb_test1_list_psig = []
    alb_test1_list_tsig = []
    alb_test1_list_t = []
    alb_test1_list_p = []
    alb_test2_list_psig = []
    alb_test2_list_tsig = []
    alb_test2_list_p = []
    alb_test2_list_t = []
    alb_test3_list_psig = []
    alb_test3_list_tsig = []
    alb_test3_list_p = []
    alb_test3_list_t = []
    alb_test4_list_psig = []
    alb_test4_list_tsig = []
    alb_test4_list_p = []
    alb_test4_list_t = []
    alb_test5_list_psig = []
    alb_test5_list_tsig = []
    alb_test5_list_p = []
    alb_test5_list_t = []
    alb_test6_list_psig = []
    alb_test6_list_tsig = []
    alb_test6_list_p = []
    alb_test6_list_t = []
    hcrf_test1_list_psig = []
    hcrf_test1_list_tsig = []
    hcrf_test1_list_t = []
    hcrf_test1_list_p = []
    hcrf_test2_list_psig = []
    hcrf_test2_list_tsig = []
    hcrf_test2_list_p = []
    hcrf_test2_list_t = []
    hcrf_test3_list_psig = []
    hcrf_test3_list_tsig = []
    hcrf_test3_list_p = []
    hcrf_test3_list_t = []
    hcrf_test4_list_psig = []
    hcrf_test4_list_tsig = []
    hcrf_test4_list_p = []
    hcrf_test4_list_t = []
    hcrf_test5_list_psig = []
    hcrf_test5_list_tsig = []
    hcrf_test5_list_p = []
    hcrf_test5_list_t = []
    hcrf_test6_list_psig = []
    hcrf_test6_list_tsig = []
    hcrf_test6_list_p = []
    hcrf_test6_list_t = []
    
    for i in np.arange(0,2150,1):   
        t1,p1 = stats.ttest_ind(np.array(HA_alb.loc[i]),np.array(LA_alb.loc[i]),equal_var=False)
        alb_test1_list_t.append(t1)
        alb_test1_list_p.append(p1)
        df1 = len(HA_alb.loc[i])+len(LA_alb.loc[i])-2
        t2,p2 = stats.ttest_ind(np.array(HA_alb.loc[i]),np.array(CI_alb.loc[i]),equal_var=False)
        alb_test2_list_t.append(t2)
        alb_test2_list_p.append(p2)
        df2 = len(HA_alb.loc[i])+len(CI_alb.loc[i])-2
        t3,p3 = stats.ttest_ind(np.array(HA_alb.loc[i]),np.array(SN_alb.loc[i]),equal_var=False)
        alb_test3_list_t.append(t3)
        alb_test3_list_p.append(p3)
        df3 = len(HA_alb.loc[i])+len(SN_alb.loc[i])-2
        t4,p4 = stats.ttest_ind(np.array(LA_alb.loc[i]),np.array(CI_alb.loc[i]),equal_var=False)
        alb_test4_list_t.append(t4)
        alb_test4_list_p.append(p4)
        df4 = len(LA_alb.loc[i])+len(CI_alb.loc[i])-2
        t5,p5 = stats.ttest_ind(np.array(LA_alb.loc[i]),np.array(SN_alb.loc[i]),equal_var=False)
        alb_test5_list_t.append(t5)
        alb_test5_list_p.append(p5)
        df5 = len(LA_alb.loc[i])+len(SN_alb.loc[i])-2
        t6,p6 = stats.ttest_ind(np.array(CI_alb.loc[i]),np.array(SN_alb.loc[i]),equal_var=False)
        alb_test6_list_t.append(t6)
        alb_test6_list_p.append(p6)
        df6 = len(CI_alb.loc[i])+len(SN_alb.loc[i])-2

        t1,p1 = stats.ttest_ind(np.array(HA_hcrf.loc[i]),np.array(LA_hcrf.loc[i]),equal_var=False)
        hcrf_test1_list_t.append(t1)
        hcrf_test1_list_p.append(p1)
        df1 = len(HA_hcrf.loc[i])+len(LA_hcrf.loc[i])-2
        t2,p2 = stats.ttest_ind(np.array(HA_hcrf.loc[i]),np.array(CI_hcrf.loc[i]),equal_var=False)
        hcrf_test2_list_t.append(t2)
        hcrf_test2_list_p.append(p2)
        df2 = len(HA_hcrf.loc[i])+len(CI_hcrf.loc[i])-2
        t3,p3 = stats.ttest_ind(np.array(HA_hcrf.loc[i]),np.array(SN_hcrf.loc[i]),equal_var=False)
        hcrf_test3_list_t.append(t3)
        hcrf_test3_list_p.append(p3)
        df3 = len(HA_hcrf.loc[i])+len(SN_hcrf.loc[i])-2
        t4,p4 = stats.ttest_ind(np.array(LA_hcrf.loc[i]),np.array(CI_hcrf.loc[i]),equal_var=False)
        hcrf_test4_list_t.append(t4)
        hcrf_test4_list_p.append(p4)
        df4 = len(LA_hcrf.loc[i])+len(CI_hcrf.loc[i])-2
        t5,p5 = stats.ttest_ind(np.array(LA_hcrf.loc[i]),np.array(SN_hcrf.loc[i]),equal_var=False)
        hcrf_test5_list_t.append(t5)
        hcrf_test5_list_p.append(p5)
        df5 = len(LA_hcrf.loc[i])+len(SN_hcrf.loc[i])-2
        t6,p6 = stats.ttest_ind(np.array(CI_hcrf.loc[i]),np.array(SN_hcrf.loc[i]),equal_var=False)
        hcrf_test6_list_t.append(t6)
        hcrf_test6_list_p.append(p6)
        df6 = len(CI_hcrf.loc[i])+len(SN_hcrf.loc[i])-2

    
    # t critical values from lookup table using p = 0.05 and df as calculated above.    
        alb_tcrit1 = 2.0076
        alb_tcrit2 = 2.0484
        alb_tcrit3 = 2.0154
        alb_tcrit4 = 2.0345
        alb_tcrit5 = 2.0096
        alb_tcrit6 = 2.0555
        
        hcrf_tcrit1 = 2.0076
        hcrf_tcrit2 = 2.0484
        hcrf_tcrit3 = 2.0154
        hcrf_tcrit4 = 2.0345
        hcrf_tcrit5 = 2.0096
        hcrf_tcrit6 = 2.0555
        
        # multiply p-value by number of comparisons = Bonferoni test for family
        if p1*6 < 0.05:
            alb_test1_list_psig.append(i+350)
        if t1 > 1.943:
            alb_test1_list_tsig.append(i+350)
        if p2*6 < 0.05:
            alb_test2_list_psig.append(i+350)
        if t2 > 1.943:
            alb_test2_list_tsig.append(i+350)
        if p3*6 < 0.05:
            alb_test3_list_psig.append(i+350)
        if t3 > 1.943:
            alb_test3_list_tsig.append(i+350)
        if p4*6 < 0.05:
            alb_test4_list_psig.append(i+350) 
        if t4 > 1.943:
            alb_test4_list_tsig.append(i+350)
        if p5*6 < 0.05:
            alb_test5_list_psig.append(i+350)
        if t5 > 1.943:
            alb_test5_list_tsig.append(i+350)
        if p6*6 < 0.05:
            alb_test6_list_psig.append(i+350) 
        if t6 > 1.943:
            alb_test6_list_tsig.append(i+350)
        
    

        if p1*6 < 0.05:
            hcrf_test1_list_psig.append(i+350)
        if t1 > 1.943:
            hcrf_test1_list_tsig.append(i+350)
        if p2*6 < 0.05:
            hcrf_test2_list_psig.append(i+350)
        if t2 > 1.943:
            hcrf_test2_list_tsig.append(i+350)
        if p3*6 < 0.05:
            hcrf_test3_list_psig.append(i+350)
        if t3 > 1.943:
            hcrf_test3_list_tsig.append(i+350)
        if p4*6 < 0.05:
            hcrf_test4_list_psig.append(i+350) 
        if t4 > 1.943:
            hcrf_test4_list_tsig.append(i+350)
        if p5*6 < 0.05:
            hcrf_test5_list_psig.append(i+350)
        if t5 > 1.943:
            hcrf_test5_list_tsig.append(i+350)
        if p6*6 < 0.05:
            hcrf_test6_list_psig.append(i+350) 
        if t6 > 1.943:
            hcrf_test6_list_tsig.append(i+350)
 
    
    if plots:
        # plot p value and t value per unit wavelemngth for albedo and hcrf
        
        fig = plt.figure(figsize=(10,8))
        sub1 = plt.subplot(2,2,1)
        sub1.plot(WL,alb_test1_list_p,color='g',label='HA vs LA')
        sub1.plot(WL,alb_test2_list_p,color='b',label = 'HA vs Snow')
        sub1.plot(WL,alb_test3_list_p,color='r',label='HA vs CI')
        sub1.plot(WL,alb_test4_list_p,color='k',label='LA vs Snow')
        sub1.plot(WL,alb_test5_list_p,color='y',label = 'LA vs CI')
        sub1.plot(WL,alb_test6_list_p,color='m',label='CI vs Snow')
        sub1.axhline(y=0.05,xmin=0,xmax=2500,linestyle='dashed',color='k')
        sub1.set_xlim(350,1400)
        sub1.set_ylim(0,0.25)
        sub1.set_xlabel('Wavelength')
        sub1.set_ylabel('p-value')
        plt.legend(loc='best')
        
        sub2 = plt.subplot(2,2,2)
        sub2.plot(WL,alb_test1_list_t,color='g',label='HA vs LA')
        sub2.plot(WL,alb_test2_list_t,color='b',label = 'HA vs Snow')
        sub2.plot(WL,alb_test3_list_t,color='r',label='HA vs CI')
        sub2.plot(WL,alb_test4_list_t,color='k',label='LA vs Snow')
        sub2.plot(WL,alb_test5_list_t,color='y',label = 'LA vs CI')
        sub2.plot(WL,alb_test6_list_t,color='m',label='CI vs Snow')
        sub2.axhline(y=alb_tcrit1,xmin=0,xmax=2500,linestyle='dashed',color='g')
        sub2.axhline(y=alb_tcrit2,xmin=0,xmax=2500,linestyle='dashed',color='b')
        sub2.axhline(y=alb_tcrit3,xmin=0,xmax=2500,linestyle='dashed',color='r')
        sub2.axhline(y=alb_tcrit4,xmin=0,xmax=2500,linestyle='dashed',color='k')
        sub2.axhline(y=alb_tcrit5,xmin=0,xmax=2500,linestyle='dashed',color='y')
        sub2.axhline(y=alb_tcrit6,xmin=0,xmax=2500,linestyle='dashed',color='m')
        sub2.set_xlim(350,1400)
        sub2.set_ylim(-15,5)
        sub2.yaxis.tick_right()
        sub2.yaxis.set_label_position("right")
        sub2.set_xlabel('Wavelength')
        sub2.set_ylabel('t-value')
        
        sub3 = plt.subplot(2,2,3)
        sub3.plot(WL,hcrf_test1_list_p,color='g',label='HA vs LA')
        sub3.plot(WL,hcrf_test2_list_p,color='b',label = 'HA vs Snow')
        sub3.plot(WL,hcrf_test3_list_p,color='r',label='HA vs CI')
        sub3.plot(WL,hcrf_test4_list_p,color='k',label='LA vs Snow')
        sub3.plot(WL,hcrf_test5_list_p,color='y',label = 'LA vs CI')
        sub3.plot(WL,hcrf_test6_list_p,color='m',label='CI vs Snow')
        sub3.axhline(y=0.05,xmin=0,xmax=2500,linestyle='dashed',color='k')
        sub3.set_xlim(350,1400)
        sub3.set_ylim(0,0.4)
        sub3.set_xlabel('Wavelength')
        sub3.set_ylabel('p-value')
        
        sub4 = plt.subplot(2,2,4)
        sub4.plot(WL,hcrf_test1_list_t,color='g',label='HA vs LA')
        sub4.plot(WL,hcrf_test2_list_t,color='b',label = 'HA vs Snow')
        sub4.plot(WL,hcrf_test3_list_t,color='r',label='HA vs CI')
        sub4.plot(WL,hcrf_test4_list_t,color='k',label='LA vs Snow')
        sub4.plot(WL,hcrf_test5_list_t,color='y',label = 'LA vs CI')
        sub4.plot(WL,hcrf_test6_list_t,color='m',label='CI vs Snow')
        sub4.axhline(y=hcrf_tcrit1,xmin=0,xmax=2500,linestyle='dashed',color='g')
        sub4.axhline(y=hcrf_tcrit2,xmin=0,xmax=2500,linestyle='dashed',color='b')
        sub4.axhline(y=hcrf_tcrit3,xmin=0,xmax=2500,linestyle='dashed',color='r')
        sub4.axhline(y=hcrf_tcrit4,xmin=0,xmax=2500,linestyle='dashed',color='k')
        sub4.axhline(y=hcrf_tcrit5,xmin=0,xmax=2500,linestyle='dashed',color='y')
        sub4.axhline(y=hcrf_tcrit6,xmin=0,xmax=2500,linestyle='dashed',color='m')
        sub4.set_xlim(350,1400)
        sub4.set_ylim(-15,5)
        sub4.yaxis.tick_right()
        sub4.yaxis.set_label_position("right")
        sub4.set_xlabel('Wavelength')
        sub4.set_ylabel('t-value')
    

        
    return



def red_edge_test(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf):
    
    alb_RE_list_HA = []
    alb_RE_listHA = []
    alb_RE_listLA = []
    alb_RE_listSnow = []
    alb_RE_listCI = []
    hcrf_RE_listHA = []
    hcrf_RE_listLA = []
    hcrf_RE_listSnow = []
    hcrf_RE_listCI = []

    for i in HAsites:
        p = 'HA_{}'.format(i)

        if HA_alb.loc[350,p] > HA_alb.loc[300,p] and HA_alb.loc[150,p] > HA_alb.loc[100,p] and HA_alb.loc[150,p] > HA_alb.loc[250,p]:
            alb_RE_listHA.append(p)

        if HA_hcrf.loc[350,p] > HA_hcrf.loc[300,p] and HA_hcrf.loc[150,p] > HA_hcrf.loc[100,p] and HA_hcrf.loc[150,p] > HA_hcrf.loc[250,p]:
            hcrf_RE_listHA.append(p)
    
    for i in LAsites:
        p = 'LA_{}'.format(i)

        if LA_alb.loc[350,p] > LA_alb.loc[300,p] and LA_alb.loc[150,p] > LA_alb.loc[100,p] and LA_alb.loc[150,p] > LA_alb.loc[250,p]:
            alb_RE_listLA.append(p)

        if LA_hcrf.loc[350,p] > LA_hcrf.loc[300,p] and LA_hcrf.loc[150,p] > LA_hcrf.loc[100,p] and LA_hcrf.loc[150,p] > LA_hcrf.loc[250,p]:
            hcrf_RE_listLA.append(p)
            
    for i in CIsites:
        p = 'CI_{}'.format(i)

        if CI_alb.loc[350,p] > CI_alb.loc[300,p] and CI_alb.loc[150,p] > CI_alb.loc[100,p] and CI_alb.loc[150,p] > CI_alb.loc[250,p]:
            alb_RE_listCI.append(p)

        if CI_hcrf.loc[350,p] > CI_hcrf.loc[300,p] and CI_hcrf.loc[150,p] > CI_hcrf.loc[100,p] and CI_hcrf.loc[150,p] > CI_hcrf.loc[250,p]:
            hcrf_RE_listCI.append(p)

    for i in SNsites:
        p = 'SN_{}'.format(i)

        if SN_alb.loc[350,p] > SN_alb.loc[300,p] and SN_alb.loc[150,p] > SN_alb.loc[100,p] and SN_alb.loc[150,p] > SN_alb.loc[250,p]:
            alb_RE_listSN.append(p)

        if SN_hcrf.loc[350,p] > SN_hcrf.loc[300,p] and SN_hcrf.loc[150,p] > SN_hcrf.loc[100,p] and SN_hcrf.loc[150,p] > SN_hcrf.loc[250,p]:
            hcrf_RE_listSN.append(p)
    
    print('Number of HA sites with red-edge and chll bump in albedo = {}'.format(len(alb_RE_listHA)))
    print('Number of LA sites with red-edge and chll bump in albedo = {}'.format(len(alb_RE_listLA)))
    print('Number of Snow sites with red-edge and chll bump in albedo = {}'.format(len(alb_RE_listSnow)))
    print('Number of CI sites with red-edge and chll bump in albedo = {}'.format(len(alb_RE_listCI)))
    print('Number of HA sites with red-edge and chll bump in hcrf = {}'.format(len(hcrf_RE_listHA)))
    print('Number of LA sites with red-edge and chll bump in hcrf = {}'.format(len(hcrf_RE_listLA)))
    print('Number of Snow sites with red-edge and chll bump in hcrf = {}'.format(len(hcrf_RE_listSnow)))
    print('Number of CI sites with red-edge and chll bump in hcrf = {}'.format(len(hcrf_RE_listCI)))

    return




def derivative_analysis(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf, plots = True, savefiles = False):

    HA_deriv_alb= pd.DataFrame()
    HA_2deriv_alb = pd.DataFrame()
    HA_deriv_hcrf =pd.DataFrame()
    HA_2deriv_hcrf = pd.DataFrame()
    LA_deriv_alb= pd.DataFrame()
    LA_2deriv_alb = pd.DataFrame()
    LA_deriv_hcrf =pd.DataFrame()
    LA_2deriv_hcrf = pd.DataFrame()        
    CI_deriv_alb= pd.DataFrame()
    CI_2deriv_alb = pd.DataFrame()
    CI_deriv_hcrf =pd.DataFrame()
    CI_2deriv_hcrf = pd.DataFrame()
    SN_deriv_alb= pd.DataFrame()
    SN_2deriv_alb = pd.DataFrame()
    SN_deriv_hcrf =pd.DataFrame()
    SN_2deriv_hcrf = pd.DataFrame()
    
    
    for i in HAsites:
        p = 'HA_{}'.format(i)
        
        dv1 = np.gradient(np.array(HA_alb[p]))
        dv2 = np.gradient(dv1)
        dv1_hcrf = np.gradient(np.array(HA_hcrf[p]))
        dv2_hcrf = np.gradient(dv1_hcrf)
        HA_deriv_alb[p] = dv1
        HA_2deriv_alb[p] = dv2
        HA_deriv_hcrf[p] = dv1_hcrf
        HA_2deriv_hcrf[p] = dv2_hcrf

    for i in LAsites:
        p = 'LA_{}'.format(i)
        
        dv1 = np.gradient(np.array(LA_alb[p]))
        dv2 = np.gradient(dv1)
        dv1_hcrf = np.gradient(np.array(LA_hcrf[p]))
        dv2_hcrf = np.gradient(dv1_hcrf)
        LA_deriv_alb[p] = dv1
        LA_2deriv_alb[p] = dv2
        LA_deriv_hcrf[p] = dv1_hcrf
        LA_2deriv_hcrf[p] = dv2_hcrf        
        
    for i in CIsites:
        p = 'CI_{}'.format(i)
        
        dv1 = np.gradient(np.array(CI_alb[p]))
        dv2 = np.gradient(dv1)
        dv1_hcrf = np.gradient(np.array(CI_hcrf[p]))
        dv2_hcrf = np.gradient(dv1_hcrf)
        CI_deriv_alb[p] = dv1
        CI_2deriv_alb[p] = dv2
        CI_deriv_hcrf[p] = dv1_hcrf
        CI_2deriv_hcrf[p] = dv2_hcrf        
        
    for i in SNsites:
        p = 'SN_{}'.format(i)
        
        dv1 = np.gradient(np.array(SN_alb[p]))
        dv2 = np.gradient(dv1)
        dv1_hcrf = np.gradient(np.array(SN_hcrf[p]))
        dv2_hcrf = np.gradient(dv1_hcrf)
        SN_deriv_alb[p] = dv1
        SN_2deriv_alb[p] = dv2
        SN_deriv_hcrf[p] = dv1_hcrf
        SN_2deriv_hcrf[p] = dv2_hcrf    


    # calculate average spectra for each group
    dv1_alb_HA_av = HA_deriv_alb.mean(axis=1)
    dv1_alb_LA_av = LA_deriv_alb.mean(axis=1)
    dv1_alb_CI_av = CI_deriv_alb.mean(axis=1)
    dv1_alb_Snow_av = SN_deriv_alb.mean(axis=1)
    dv2_alb_HA_av = HA_2deriv_alb.mean(axis=1)
    dv2_alb_LA_av = LA_2deriv_alb.mean(axis=1)
    dv2_alb_CI_av = CI_2deriv_alb.mean(axis=1)
    dv2_alb_Snow_av = SN_2deriv_alb.mean(axis=1)
    
    dv1_hcrf_HA_av = HA_deriv_hcrf.mean(axis=1)
    dv1_hcrf_LA_av = LA_deriv_hcrf.mean(axis=1)
    dv1_hcrf_CI_av = CI_deriv_hcrf.mean(axis=1)
    dv1_hcrf_Snow_av = SN_deriv_hcrf.mean(axis=1)
    dv2_hcrf_HA_av = HA_2deriv_hcrf.mean(axis=1)
    dv2_hcrf_LA_av = LA_2deriv_hcrf.mean(axis=1)
    dv2_hcrf_CI_av = CI_2deriv_hcrf.mean(axis=1)
    dv2_hcrf_Snow_av = SN_2deriv_hcrf.mean(axis=1)
    

    if plots:
        fig = plt.figure(figsize=(18, 12))
        sub1 = plt.subplot(4, 1, 1)
        sub1.set_xlim(350,700)
        sub1.set_ylim(-0.001,0.001)
        sub1.set_ylabel('1st Derivative'+'\n'+ 'Albedo',fontsize=22)
        sub1.plot(WL,dv1_alb_HA_av,color='g',label='H$_{bio}$')
        sub1.plot(WL,dv1_alb_LA_av,color='r',label='L$_{bio}$')
        sub1.plot(WL,dv1_alb_CI_av,color='b',alpha=0.2,label='CI')
        sub1.plot(WL,dv1_alb_Snow_av,color='k',alpha=0.5,label='SN')
        sub1.axvspan(680,690,color='g',alpha=0.1)
        sub1.locator_params(nbins=4, axis='y')
        sub1.legend(ncol=4,loc='best',fontsize=22)
        sub1.grid(None)
        
        sub2 = plt.subplot(4, 1, 2)
        sub2.set_xlim(350,700)
        sub2.set_ylabel('2nd Derivative'+'\n'+'Albedo',fontsize=22)
        sub2.set_xlabel('Wavelength (nm)',fontsize=22)
        sub2.set_ylim(-0.0002,0.0002)
        sub2.yaxis.tick_left()
        sub2.yaxis.set_label_position("left")
        sub2.plot(WL,dv2_alb_HA_av,color='g')
        sub2.plot(WL,dv2_alb_LA_av,color='r')
        sub2.plot(WL,dv2_alb_CI_av,color='b',alpha=0.2)
        sub2.plot(WL,dv2_alb_Snow_av,color='k',alpha=0.5)
        sub2.axvspan(680,690,color='g',alpha=0.1)
        sub2.locator_params(nbins=4, axis='y')
        sub2.grid(None)
    
    
        sub1.tick_params(axis='x', which='major', labelsize=0)
        sub1.tick_params(axis='y', which='major', labelsize=22)
        sub2.tick_params(axis='both', which='major', labelsize=22)
        plt.savefig('/home/joe/Desktop/derivative.jpg',dpi=150)
        
        sub3 = plt.subplot(4, 1, 3)
        sub3.set_xlim(350,700)
        sub3.set_ylabel('1st Derivative HCRF')
        sub3.set_ylim(-0.0008,0.001)
        sub3.yaxis.tick_left()
        sub3.yaxis.set_label_position("left")
        sub3.plot(WL,dv1_hcrf_HA_av,color='g')
        sub3.plot(WL,dv1_hcrf_LA_av,color='r')
        sub3.plot(WL,dv1_hcrf_CI_av,color='b',alpha=0.2)
        sub3.plot(WL,dv1_hcrf_Snow_av,color='k',alpha=0.5)
        sub3.axvspan(680,690,color='g',alpha=0.1)
        sub3.locator_params(nbins=4, axis='y')
        
        sub4 = plt.subplot(4, 1, 4)
        sub4.set_xlim(350,700)
        sub4.set_ylabel('2nd Derivative HCRF')
        sub4.set_ylim(-0.0002,0.0002)
        sub4.yaxis.tick_left()
        sub4.yaxis.set_label_position("left")
        sub4.plot(WL,dv2_hcrf_HA_av,color='g')
        sub4.plot(WL,dv2_hcrf_LA_av,color='r')
        sub4.plot(WL,dv2_hcrf_CI_av,color='b',alpha=0.2)
        sub4.plot(WL,dv2_hcrf_Snow_av,color='k',alpha=0.5)
        sub4.axvspan(680,690,color='g',alpha=0.1)
        sub4.locator_params(nbins=4, axis='y')
        
        sub4.text(480, -0.0004, 'Wavelength (nm)', fontsize=11)
        
        if savefiles:
            HA_deriv_alb.to_csv('HA_deriv_alb.csv')
            HA_2deriv_alb.to_csv('HA_2deriv_alb.csv')
            HA_deriv_hcrf.to_csv('HA_deriv_hcrf.csv')
            HA_2deriv_hcrf.to_csv('HA_2deriv_hcrf.csv')
            LA_deriv_alb.to_csv('LA_deriv_alb.csv')
            LA_2deriv_alb.to_csv('LA_2deriv_alb.csv')
            LA_deriv_hcrf.to_csv('LA_deriv_hcrf.csv')
            LA_2deriv_hcrf.to_csv('LA_2deriv_hcrf.csv')        
            CI_deriv_alb.to_csv('CI_deriv_alb.csv')
            CI_2deriv_alb.to_csv('CI_2deriv_alb.csv')
            CI_deriv_hcrf.to_csv('CI_deriv_hcrf.csv')
            CI_2deriv_hcrf.to_csv('CI_2deriv_hcrf.csv')
            SN_deriv_alb.to_csv('SN_deriv_alb.csv')
            SN_2deriv_alb.to_csv('SN_2deriv_alb.csv')
            SN_deriv_hcrf.to_csv('SN_deriv_hcrf.csv')
            SN_2deriv_hcrf.to_csv('SN_2deriv_hcrf')

    return


def calculate_ARF(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf, plots = True, savefiles = False):

    HA_ARF = pd.DataFrame()
    LA_ARF = pd.DataFrame()
    CI_ARF = pd.DataFrame()
    SN_ARF = pd.DataFrame()
    
    
    for i in HAsites:
        p = 'HA_{}'.format(i)
        HA_ARF[p] = HA_hcrf[p] / HA_alb[p]
    for i in LAsites:
        p = 'LA_{}'.format(i)
        LA_ARF[p] = LA_hcrf[p] / LA_alb[p]
    for i in CIsites:
        p = 'CI_{}'.format(i)
        CI_ARF[p] = CI_hcrf[p] / CI_alb[p]
    for i in SNsites:
        p = 'SN_{}'.format(i)
        SN_ARF[p] = SN_hcrf[p] / SN_alb[p]        

    meanARF_HA = HA_ARF.mean(axis=1)
    meanARF_LA = LA_ARF.mean(axis=1)
    meanARF_CI = CI_ARF.mean(axis=1)
    meanARF_SN = SN_ARF.mean(axis=1)
    
    if plots:
        plt.figure(figsize = (10,10))
        plt.plot(WL,meanARF_HA,label='HA'),plt.plot(WL,meanARF_LA,label='LA'),plt.plot(WL,meanARF_CI,label='CI'),plt.plot(WL,meanARF_SN,label='SN'),
        plt.ylim(0,1.2),plt.xlim(300,2200),plt.ylabel('Anisotropic Reflectance Factor'), 
        plt.xlabel('Wavelength (nm)'), plt.legend(loc='best')

    if savefiles:
        HA_ARF.to_csv('HA_ARF.csv')
        LA_ARF.to_csv('LA_ARF.csv')
        CI_ARF.to_csv('CI_ARF.csv')
        SN_ARF.to_csv('SN_ARF.csv')  
        
    return




def absorption_feature_1030(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf, plots = True):

    HA = HA_alb.mean(axis=1)
    LA = LA_alb.mean(axis=1)
    CI = CI_alb.mean(axis=1)
    SN = SN_alb.mean(axis=1)
    
    HA_hcrf = HA_hcrf.mean(axis=1)
    LA_hcrf = LA_hcrf.mean(axis=1)
    CI_hcrf = CI_hcrf.mean(axis=1)
    SN_hcrf = SN_hcrf.mean(axis=1)

    #Calculate the 'continuum' by drawing a straight line between the shoulders of
    # the absorption feature. I have defined the shoulders as 950 and 1035 nm.
    
    RcHA = [HA[600] + (n * (HA[735] - HA[600])/135)  for n in range(135)]
    RcLA = [LA[600] + (n * (LA[735] - LA[600])/135)  for n in range(135)]
    RcCI = [CI[600] + (n * (CI[735] - CI[600])/135)  for n in range(135)]
    RcSN = [SN[600] + (n * (SN[735] - SN[600])/135)  for n in range(135)]
    
    RcHA_hcrf = [HA_hcrf[600] + (n * (HA_hcrf[735] - HA_hcrf[600])/135)  for n in range(135)]
    RcLA_hcrf = [LA_hcrf[600] + (n * (LA_hcrf[735] - LA_hcrf[600])/135)  for n in range(135)]
    RcCI_hcrf = [CI_hcrf[600] + (n * (CI_hcrf[735] - CI_hcrf[600])/135)  for n in range(135)]
    RcSN_hcrf = [SN_hcrf[600] + (n * (SN_hcrf[735] - SN_hcrf[600])/135)  for n in range(135)]
    
    # calculate the depth of the absorption feature by subtracting the real
    # reflectance from the continuum reflectance
    
    feature_depths_HA = RcHA - HA[600:735]
    feature_depths_LA = RcLA - LA[600:735]
    feature_depths_CI = RcCI - CI[600:735]
    feature_depths_SN = RcSN - SN[600:735]
    
    feature_area_HA = sum(feature_depths_HA)
    feature_area_LA = sum(feature_depths_LA)
    feature_area_CI = sum(feature_depths_CI)
    feature_area_SN = sum(feature_depths_SN)
    
    feature_depths_HA_hcrf = RcHA_hcrf - HA_hcrf[600:735]
    feature_depths_LA_hcrf = RcLA_hcrf - LA_hcrf[600:735]
    feature_depths_CI_hcrf = RcCI_hcrf - CI_hcrf[600:735]
    feature_depths_SN_hcrf = RcSN_hcrf - SN_hcrf[600:735]
    
    feature_area_HA_hcrf = sum(feature_depths_HA_hcrf)
    feature_area_LA_hcrf = sum(feature_depths_LA_hcrf)
    feature_area_CI_hcrf = sum(feature_depths_CI_hcrf)
    feature_area_SN_hcrf = sum(feature_depths_SN_hcrf)
    
    # print maximum depths for each ice type
    print('Max feature depth for HA = ', np.max(feature_depths_HA))
    print('Max feature depth for LA = ', np.max(feature_depths_LA))
    print('Max feature depth for CI = ', np.max(feature_depths_CI))
    print('Max feature depth for SN = ', np.max(feature_depths_SN))

    print('Feature area for HA = ', feature_area_HA)
    print('Feature area for LA = ', feature_area_LA)
    print('Feature area for CI = ', feature_area_CI)
    print('Feature area for SN = ', feature_area_SN)   

    print('Max feature depth for HA (HCRF) = ', np.max(feature_depths_HA_hcrf))
    print('Max feature depth for LA (HCRF) = ', np.max(feature_depths_LA_hcrf))
    print('Max feature depth for CI (HCRF) = ', np.max(feature_depths_CI_hcrf))
    print('Max feature depth for SN (HCRF) = ', np.max(feature_depths_SN_hcrf))

    print('Feature area for HA (HCRF) = ', feature_area_HA_hcrf)
    print('Feature area for LA (HCRF) = ', feature_area_LA_hcrf)
    print('Feature area for CI (HCRF) = ', feature_area_CI_hcrf)
    print('Feature area for SN (HCRF) = ', feature_area_SN_hcrf)


# Calculate assymetry of the absorption feature witht he axis of symmetry at
# 1020 nm. Positive values indicate right assymetry, negative values indicate left assymetry
# and zero indicates perfect symmetry around 1020 nm.
    
    HA_L_assym = np.sum(feature_depths_HA[0:int(len(feature_depths_HA)/2)])
    HA_R_assym = np.sum(feature_depths_HA[int(len(feature_depths_HA)/2):-1])
    assymHA = math.log10(HA_R_assym/HA_L_assym)

    LA_L_assym = np.sum(feature_depths_LA[0:int(len(feature_depths_LA)/2)])
    LA_R_assym = np.sum(feature_depths_LA[int(len(feature_depths_LA)/2):-1])
    assymLA = math.log10(LA_R_assym/LA_L_assym)

    CI_L_assym = np.sum(feature_depths_CI[0:int(len(feature_depths_CI)/2)])
    CI_R_assym = np.sum(feature_depths_CI[int(len(feature_depths_CI)/2):-1])
    assymCI = math.log10(CI_R_assym/CI_L_assym)

    SN_L_assym = np.sum(feature_depths_SN[0:int(len(feature_depths_SN)/2)])
    SN_R_assym = np.sum(feature_depths_SN[int(len(feature_depths_SN)/2):-1])
    assymSN = math.log10(SN_R_assym/SN_L_assym)

    HA_L_assym_hcrf = np.sum(feature_depths_HA_hcrf[0:int(len(feature_depths_HA_hcrf)/2)])
    HA_R_assym_hcrf = np.sum(feature_depths_HA_hcrf[int(len(feature_depths_HA_hcrf)/2):-1])
    assymHA_hcrf = math.log10(HA_R_assym_hcrf/HA_L_assym_hcrf)

    LA_L_assym_hcrf = np.sum(feature_depths_LA_hcrf[0:int(len(feature_depths_LA_hcrf)/2)])
    LA_R_assym_hcrf = np.sum(feature_depths_LA_hcrf[int(len(feature_depths_LA_hcrf)/2):-1])
    assymLA_hcrf = math.log10(LA_R_assym_hcrf/LA_L_assym_hcrf)

    CI_L_assym_hcrf = np.sum(feature_depths_CI_hcrf[0:int(len(feature_depths_CI_hcrf)/2)])
    CI_R_assym_hcrf = np.sum(feature_depths_CI_hcrf[int(len(feature_depths_CI_hcrf)/2):-1])
    assymCI_hcrf = math.log10(CI_R_assym_hcrf/CI_L_assym_hcrf)

    SN_L_assym_hcrf = np.sum(feature_depths_SN_hcrf[0:int(len(feature_depths_SN_hcrf)/2)])
    SN_R_assym_hcrf = np.sum(feature_depths_SN_hcrf[int(len(feature_depths_SN_hcrf)/2):-1])
    assymSN_hcrf = math.log10(SN_R_assym_hcrf/SN_L_assym_hcrf)

    
    print('Assymetry of absorption feature HA = ',assymHA)
    print('Assymetry of absorption feature LA = ',assymLA)
    print('Assymetry of absorption feature CI = ',assymCI)
    print('Assymetry of absorption feature SN = ',assymSN)
    
    print('Assymetry of absorption feature HA (HCRF) = ',assymHA_hcrf)
    print('Assymetry of absorption feature LA (HCRF) = ',assymLA_hcrf)
    print('Assymetry of absorption feature CI (HCRF) = ',assymCI_hcrf)
    print('Assymetry of absorption feature SN (HCRF) = ',assymSN_hcrf)


    #Optical grain radius calculayted using equations from biagio di mauro's EGU poster
    # inverted to solve for OGR (USING HCRF DATA ONLY)
    OGR_HA = (2.4 - 0.97*feature_area_HA_hcrf)/(feature_area_HA_hcrf - 22.2)
    OGR_LA = (2.4 - 0.97*feature_area_LA_hcrf)/(feature_area_LA_hcrf - 22.2)
    OGR_CI = (2.4 - 0.97*feature_area_CI_hcrf)/(feature_area_CI_hcrf - 22.2)
    OGR_SN = (2.4 - 0.97*feature_area_SN_hcrf)/(feature_area_SN_hcrf - 22.2)
    
    print('Optical grain radius (microns) HA = ', OGR_HA)
    print('Optical grain radius (microns) LA = ', OGR_LA)
    print('Optical grain radius (microns) CI = ', OGR_CI)
    print('Optical grain radius (microns) SN = ', OGR_SN)

    
    if plots:
    
        fig = plt.figure(figsize=(20,16)),
        sub1=plt.subplot(2,2,1)
        sub1.plot(WL,HA,label = 'Heavy Algae')
        sub1.plot(WL,LA, label = 'Light algae')
        sub1.plot(WL,CI, label = 'Clean Ice')
        sub1.plot(WL,SN, label = 'Clean SN')
        sub1.set_ylim(0,0.8)
        sub1.set_xlim(350,2000)
        sub1.axvline(x = 950,color='k',alpha=0.2,linestyle='dashed')
        sub1.axvline(x=1085,color='k',alpha=0.2,linestyle='dashed')
        sub1.plot(WL[600:735],RcHA,color='b',alpha=0.5,linestyle='dashed')
        sub1.plot(WL[600:735],RcLA,color='g',alpha=0.5,linestyle='dashed')
        sub1.plot(WL[600:735],RcCI,color='r',alpha=0.5,linestyle='dashed')
        sub1.plot(WL[600:735],RcSN,color='c',alpha=0.5,linestyle='dashed')
        sub1.set_ylabel('Albedo')
        plt.grid(None)
        plt.legend(loc='upper right')
        
        sub2 = plt.subplot(2,2,2)
        sub2.plot(WL[600:735],feature_depths_HA,label='Heavy Algae')
        sub2.plot(WL[600:735],feature_depths_LA, label='Light Algae')
        sub2.plot(WL[600:735],feature_depths_CI, label = 'Clean Ice')
        sub2.plot(WL[600:735],feature_depths_SN, label = 'SN')
        sub2.set_ylim(0,0.15)
        sub2.set_xlim(950,1085)
        sub2.set_xlabel('Wavelength (nm)')
        sub2.set_ylabel('Depth of absorption feature')
        plt.grid(None)        
        
        sub3=plt.subplot(2,2,3)
        sub3.plot(WL,HA_hcrf,label = 'Heavy Algae')
        sub3.plot(WL,LA_hcrf, label = 'Light algae')
        sub3.plot(WL,CI_hcrf, label = 'Clean Ice')
        sub3.plot(WL,SN_hcrf, label = 'Clean SN_hcrf')
        sub3.set_ylim(0,0.8)
        sub3.set_xlim(350,2000)
        sub3.axvline(x = 950,color='k',alpha=0.2,linestyle='dashed')
        sub3.axvline(x=1085,color='k',alpha=0.2,linestyle='dashed')
        sub3.plot(WL[600:735],RcHA_hcrf,color='b',alpha=0.5,linestyle='dashed')
        sub3.plot(WL[600:735],RcLA_hcrf,color='g',alpha=0.5,linestyle='dashed')
        sub3.plot(WL[600:735],RcCI_hcrf,color='r',alpha=0.5,linestyle='dashed')
        sub3.plot(WL[600:735],RcSN_hcrf,color='c',alpha=0.5,linestyle='dashed')
        sub3.set_ylabel('HCRF')
        plt.grid(None)
        
        sub4 = plt.subplot(2,2,4)
        sub4.plot(WL[600:735],feature_depths_HA_hcrf,label='Heavy Algae')
        sub4.plot(WL[600:735],feature_depths_LA_hcrf, label='Light Algae')
        sub4.plot(WL[600:735],feature_depths_CI_hcrf, label = 'Clean Ice')
        sub4.plot(WL[600:735],feature_depths_SN_hcrf, label = 'SN_hcrf')
        sub4.set_ylim(0,0.15)
        sub4.set_xlim(950,1085)
        sub4.set_xlabel('Wavelength (nm)')
        sub4.set_ylabel('Depth of absorption feature')
        plt.grid(None)

        
    return 



HA_alb, HA_hcrf, LA_alb, LA_hcrf, CI_alb, CI_hcrf, SN_alb, SN_hcrf = create_plot_alb_hcrf(process_spectra = True, plots = 2, savefiles = False)

#alb_stat_list, alb_p_list, hcrf_stat_list, hcrf_p_list = albedo_hcrf_ANOVA(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf,plots=True,savefiles=False)
#
#alb_hcrf_posthoc_tests(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf,plots=True)
##
#red_edge_test(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf)
##
#derivative_analysis(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf, plots = True, savefiles = False)
##
#calculate_ARF(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf, plots = True, savefiles = False)
##
#absorption_feature_1030(HA_alb,HA_hcrf,LA_alb,LA_hcrf,CI_alb,CI_hcrf,SN_alb,SN_hcrf, plots = True)
#










