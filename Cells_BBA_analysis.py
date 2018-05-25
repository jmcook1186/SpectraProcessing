#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:21:07 2018

@author: joe
"""

from numpy import * 
from scipy.stats.stats import pearsonr
import scipy.stats as stats
import statsmodels as sm

# Import BBA and cell count dataset from csv
DF = pd.read_csv('/home/joe/Code/Log_cells_BBA.csv')
xs = np.array(DF['Log_cells'])
ys = np.array(DF['BBA'])

# Define functions for regression, correlation and analysis of variance

def best_fit(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)*mean(xs)) - mean(xs*xs)))     
    b = mean(ys) - m*mean(xs)
    print('regression line equation = {}x + {}'.format(np.round(m,3),np.round(b,3)))        
    return m, b

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys) * (ys_line - ys))

def coefficient_of_determination(ys,ys_line):
    y_mean_line = [mean(ys) for y in ys]
    squared_error_regr = squared_error(ys, ys_line)
    squared_error_y_mean = squared_error(ys, y_mean_line)
    r_squared = 1 - (squared_error_regr/squared_error_y_mean)
    R,Rp = pearsonr(xs, ys)
    return r_squared, R, Rp
    
def plots(xs,ys,line,R,Rp):

    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10,10))
    plt.scatter(xs,ys,color='b'),plt.plot(xs,line,color='r'),plt.xlabel('Log cells mL$^{-1}$',fontsize=22),
    plt.ylabel('Broadband Albedo (0.35 - 2.5 $\mu$m)',fontsize=22),plt.xticks(fontsize=22),plt.yticks(fontsize=22),
    plt.legend(loc='best'),plt.grid(None),plt.text(4.1,0.72,'$r^2$ = {}'.format(np.round(r_squared,2)),fontsize=22),
    plt.text(1.6,0.68,'Pearson R = {} (p = {})'.format(R,Rp),fontsize=22),plt.savefig('/home/joe/Desktop/Cells_vs_BBA.jpg',facecolor='white',dpi=150)
    return    
    
def ANOVA(DF):
    
    HA_cells = DF.loc[DF['Class'] == 'HA']
    HA_cells = np.array(HA_cells['Cells_mL'])

    LA_cells = DF.loc[DF['Class'] == 'LA']
    LA_cells = np.array(LA_cells['Cells_mL'])
    
    CI_cells = DF.loc[DF['Class'] == 'CI']
    CI_cells = np.array(CI_cells['Cells_mL'])
    
    SN_cells = DF.loc[DF['Class'] == 'SN']
    SN_cells = np.array(SN_cells['Cells_mL'])
    
    F_stat, p = stats.mstats.f_oneway(HA_cells,LA_cells,CI_cells,SN_cells)
    
    print('ANOVA F-stat = {} , ANOVA p = {}'.format(F_stat,p))    
    print('HA Summary: Max = {}, Min = {}, mean = {}, Std = {}'.format(np.max(HA_cells),np.min(HA_cells),np.mean(HA_cells),np.std(HA_cells)))
    print('LA Summary: Max = {}, Min = {}, mean = {}, Std = {}'.format(np.max(LA_cells),np.min(LA_cells),np.mean(LA_cells),np.std(LA_cells)))
    print('CI Summary: Max = {}, Min = {}, mean = {}, Std = {}'.format(np.max(CI_cells),np.min(CI_cells),np.mean(CI_cells),np.std(CI_cells)))
    print('SN Summary: Max = {}, Min = {}, mean = {}, Std = {}'.format(np.max(SN_cells),np.min(SN_cells),np.mean(SN_cells),np.std(SN_cells)))
    
    return F_stat, p



m,b = best_fit(xs,ys)
line = [(m*x)+b for x in xs]
r_squared, R, Rp = coefficient_of_determination(ys,line)
plots(xs,ys,line, R, Rp)



