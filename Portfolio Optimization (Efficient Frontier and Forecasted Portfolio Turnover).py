# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:05:44 2023

@author: banik
"""

import numpy as np
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt
from fredapi import Fred

# Get API key: Fred
fred = Fred(api_key='Enter Your Fred API Key')
rf = fred.get_series('DTB6')[-1]/100
S1 = input("Ticker for stock 1: ").upper()
S2 = input("Ticker for stock 2: ").upper()
S3 = input("Ticker for stock 3: ").upper()
S4 = input("Ticker for stock 4: ").upper()
np.random.seed(123)
Tickers = [S1,S2,S3,S4]
print(f'selected tickers: {Tickers}')
def my(year,frequency,Implied_volatility, months,steps,required_simulation,Average_AUM):
    #1. Collect returns data for 4 (or more) different assets, which have in T observations, where T is 5 years or more.
    df = np.matrix(yf.Tickers(Tickers).history(year,frequency).Close.dropna().pct_change().dropna())
    pd.set_option('max_columns', None)   
    #2. Estimate the sample mean vector μ_T and the sample covariance matrix Σ_T
    n_rows = df.shape[0]
    n_col = df.shape[1]
    row_vector = np.matrix(np.ones(n_rows)).T #creating a row vector (i.e.,n_rows x 1 vector)
    mean_vector = df.T*row_vector/n_rows  #sample mean vector
    cov_matrix = (df-mean_vector.T).T*(df-mean_vector.T)/(n_rows-1) #sample covariance matrix
        
    #3. Calculate the global minimean_vector0m variance (GMV) portfolio weights, 
    #the GMV portfolio’s expected return, and 
    #the GMV portfolio’s standard deviation
    #7. Estimate the GMV's portfolio weights
    col_vector=np.matrix(np.ones(n_col)).T #create a col_vectorit vector (i.e., n_col x 1 vector) with columns
    GMV_w = (pow(cov_matrix,-1)*col_vector)/(col_vector.T*pow(cov_matrix,-1)*col_vector) #GMV portfolio weights
    GMV_r = GMV_w.T*mean_vector
    GMV_SD = np.sqrt(GMV_w.T*cov_matrix*GMV_w)
        
    #4.	Calculate the maximean_vector0m Sharpe ratio (MSR) portfolio weights, 
    #the MSR portfolio’s expected return, and 
    #the MSR portfolio’s standard deviation
    #7. Estimate the MSR's portfolio returns
    MSR_w = (pow(cov_matrix,-1)*mean_vector)/(col_vector.T*pow(cov_matrix,-1)*mean_vector) #MSR portfolio weights
    MSR_r = MSR_w.T*mean_vector #MSR portfolio return
    MSR_SD = np.sqrt(MSR_w.T*cov_matrix*MSR_w) #MSR portfolio standard deviation
    A=col_vector.T*cov_matrix**(-1)*col_vector #coefficient A in characteristic fcol_vectorction
    B=col_vector.T*cov_matrix**(-1)*mean_vector #coefficient B in characteristic fcol_vectorction
    C=mean_vector.T*cov_matrix**(-1)*mean_vector #coefficient C in characteristic fcol_vectorction
    G=A*C-B**2 #coefficient G in characteristic fcol_vectorction
    SD = np.sqrt(1/A) #GMV portfolio's standard deviation
    grid = np.matrix(np.arange(SD,Implied_volatility/np.sqrt(months),(Implied_volatility/np.sqrt(months)-SD)/100)).T
    Cap_r = (-(-2*B)+np.sqrt(pow(-2*B, 2)-4*np.multiply(A,(C-np.multiply(G,np.square(grid))))))/(2*A) #Expected portfolio return in best case scenario
    Floor_r = (-(-2*B)-np.sqrt(pow(-2*B, 2)-4*np.multiply(A,(C-np.multiply(G,np.square(grid))))))/(2*A) #Expected portfolio return in worst case scenario
    #plots the Investment Opportunity Set
    fig, ax = plt.subplots()
    ax.plot(grid*np.sqrt(months),Cap_r*months,grid*np.sqrt(months),Floor_r*months,color='b')
    ax.set_title("Investment Opportunity Set")
    ax.set_xlabel("Annualized standard deviation")
    ax.set_ylabel("Annual expected return")
    plt.show()
        
        
    simulation=np.zeros((required_simulation+1,len(np.square(grid))))
    simulation[0,:] = Cap_r.T
    i=1
    while i<=required_simulation:
            df=np.random.multivariate_normal(np.ravel(mean_vector),cov_matrix,n_rows)
            n_rows=df.shape[0]
            n_col=df.shape[1]
            row_vector=np.matrix(np.ones(n_rows)).T #col_vectorit vector
            mean_vector0=df.T*row_vector/n_rows #mean_vector0 vector
            cov_matrix0=(df-mean_vector0.T).T*(df-mean_vector0.T)/(n_rows-1)
            col_vector=np.matrix(np.ones(n_col)).T
            A=col_vector.T*cov_matrix0**(-1)*col_vector
            B=col_vector.T*cov_matrix0**(-1)*mean_vector0
            C=mean_vector0.T*cov_matrix0**(-1)*mean_vector0
            G=A*C-B**2
            SD=np.sqrt(1/A)
            grid0=np.matrix(np.arange(SD,Implied_volatility/np.sqrt(months),(Implied_volatility/np.sqrt(months)-SD)/required_simulation)).T
            Cap_r0=(-(-2*B)+np.sqrt((-2*B)**2-4*np.multiply(A,(C-np.multiply(G,np.square(grid0))))))/(2*A)
            Floor_r0=(-(-2*B)-np.sqrt((-2*B)**2-4*np.multiply(A,(C-np.multiply(G,np.square(grid0))))))/(2*A)
            simulation[i,:] = Cap_r.T
            plt.plot(grid0*np.sqrt(months),Cap_r0*months,linewidth=0.1)
            i=i+1
    plt.plot(grid0*np.sqrt(months),Cap_r0*months,color='b',linewidth=3)
    plt.title("Investment Opportunity Set")
    plt.xlabel("Annualized standard deviation")
    plt.ylabel("Annual expected return")
    plt.show()
    
    #8.	Plot the 1,000 GMV and MSR (on a separate figure each for the GMV and MSR portfolios) 
    #portfolios’ expected returns and standard deviations in mean-standard deviation space
    # Plot the 1,000 GMV portfolio's expected returns and standard deviations in mean-standard deviation space 
    simulation0 = np.zeros((required_simulation+1,6))
    simulation0[0,0]=GMV_SD
    simulation0[0,1]=GMV_r
    simulation0[0,2]=GMV_w.T[:,0]
    simulation0[0,3]=GMV_w.T[:,1]
    simulation0[0,4]=GMV_w.T[:,2]
    simulation0[0,5]=GMV_w.T[:,3]
    fc_simulation0 = np.zeros((required_simulation+1,5))
    fc_simulation0[0,0] = GMV_w.T[:,0]*Average_AUM
    fc_simulation0[0,1] = GMV_w.T[:,1]*Average_AUM
    fc_simulation0[0,2] = GMV_w.T[:,2]*Average_AUM
    fc_simulation0[0,3] = GMV_w.T[:,3]*Average_AUM
    fc_simulation0[0,4] = fc_simulation0[0,0]+fc_simulation0[0,1]+fc_simulation0[0,2]+fc_simulation0[0,3]
    i=1
    while i<=required_simulation:
            df0=np.random.multivariate_normal(np.ravel(mean_vector),cov_matrix,n_rows)
            n_rows0=df0.shape[0]
            n_col0=df0.shape[1]
            row_vector0=np.matrix(np.ones(n_rows0)).T #col_vectorit vector
            mean_vector0=df0.T*row_vector0/n_rows0 #mean_vector0 vector
            cov_matrix0=(df0-mean_vector0.T).T*(df0-mean_vector0.T)/(n_rows0-1)
            col_vector0=np.matrix(np.ones(n_col0)).T
            cov_matrix0=(df0-mean_vector0.T).T*(df0-mean_vector0.T)/(n_rows0-1)
            GMV_w0 = (pow(cov_matrix0,-1)*col_vector0)/(col_vector0.T*pow(cov_matrix0,-1)*col_vector0)
            GMV_r0 = GMV_w0.T*mean_vector
            GMV_SD0 = np.sqrt(GMV_w0.T*cov_matrix*GMV_w0)
            simulation0[i,0]=GMV_SD0
            simulation0[i,1]=GMV_r0
            simulation0[i,2]=GMV_w0.T[:,0]
            simulation0[i,3]=GMV_w0.T[:,1]
            simulation0[i,4]=GMV_w0.T[:,2]
            simulation0[i,5]=GMV_w0.T[:,3]
            fc_simulation0[i,0] = (GMV_w0.T[:,0]*Average_AUM)
            fc_simulation0[i,1] = (GMV_w0.T[:,1]*Average_AUM)
            fc_simulation0[i,2] = (GMV_w0.T[:,2]*Average_AUM)
            fc_simulation0[i,3] = (GMV_w0.T[:,3]*Average_AUM)
            fc_simulation0[i,4] = fc_simulation0[0,0]+fc_simulation0[0,1]+fc_simulation0[0,2]+fc_simulation0[0,3]
            plt.plot(GMV_SD0*np.sqrt(months),GMV_r0*months,color='green',marker='o', markersize=0.5)
            i = i+1
    #5.	Assume that asset returns conform to a multivariate normal distribution, 
    #with mean and covariance matrix equal to the sample ones, which were estimated in Step 2.
    #6.	Simulate 1,000 independent samples for each asset from the multivariate normal distribution 
    # with mean μ_T and covariance matrix Σ_T, with each draw consisting of T returns.
    #7. '
    print('\n')
    print('Sample mean vector μ_T:')
    print('\n')
    print(mean_vector)
    print('\n')
    print('Sample covariance matrix Σ_T:')
    print('\n')
    print(cov_matrix)
    print('\n')
        
    print('Global minimum variance portfolio weights:')
    print('\n')
    print(round(pd.DataFrame(GMV_w,index=Tickers,columns=["Weight"]),2).T)
    print('\n')
    print(f'Expected return of Global minimum variance portfolio: {round(float(GMV_r*100),4)}%')
    print('\n')
    print(f'Standard deviation of Global minimum variance portfolio: {round(float(GMV_SD*100),4)}%')
    print('\n')
        
    print('Maximum sharpe ratio portfolio weights:')
    print('\n')
    print(round(pd.DataFrame(MSR_w,index=Tickers,columns=["Weight"]),2).T)
    print('\n')
    print(f'Expected return of Maximum sharpe ratio portfolio: {round(float(MSR_r*100),4)}%')
    print('\n')
    print(f'Standard deviation of Maximum sharpe ratio portfolio: {round(float(GMV_SD*100),4)}%')
    print('\n')
        
    print('Assuming the asset returns conform to a multivariance normal distribution - ')
    print('\n')

    print("Portfolio weights in GMV method:")
    print("\n")
    GMV_Weights = pd.DataFrame(GMV_w,index=Tickers,columns=["Weight"])
    GMV_Weights = GMV_Weights.T
    print(np.round(GMV_Weights,2))

    print("\n")
    print("GMV Portfolio Simulation with Standard Deviation, Expected Return, and Weights:")
    print("\n")
    GMV = pd.DataFrame(simulation0[:1000],columns=['Standard Deviation','Expected Return',
                                              f'Weight of {Tickers[0]}',f'Weight of {Tickers[1]}',
                                              f'Weight of {Tickers[2]}',f'Weight of {Tickers[3]}'])
    print(round(GMV,3))
    print("\n")
    print("Summary Statistics for re-sampled GMV portfolio:")
    print("\n")
    print(round(GMV.describe(),3))
    print("\n")
    print("Simulation of GMV Sharpe Ratio:")
    print("\n")
    GMV_Sharpe_ratio = round(pd.DataFrame((simulation0[:1000,1]-rf)/simulation0[:1000,0],columns=['GMV Sharpe Ratio']),3)
    print(GMV_Sharpe_ratio)
    print("\n")
    print("Summary Statistics of GMV Sharpe Ratio")
    sum_stat0 = round(GMV_Sharpe_ratio.describe(),3)
    print(sum_stat0)
    print("\n")
    GMV_allocation = GMV_Weights*Average_AUM
    GMV_allocation.loc[:,'Total'] = GMV_allocation.sum(numeric_only=True,axis=1)
    GMV_allocation = pd.concat([GMV_allocation]*1000,ignore_index=True)
    sim_allocation0 = pd.DataFrame(fc_simulation0[:1000],columns=GMV_allocation.columns)
    GMV_turnover =  GMV_allocation.subtract(sim_allocation0,axis=1)
    GMV_turnover = GMV_turnover.drop(['Total'],axis=1)
    GMV_turnover.loc[:,'Turnover'] = abs(GMV_turnover).sum(numeric_only=True,axis=1)/2
    GMV_turnover['Portfolio turnover ratio'] = GMV_turnover['Turnover']/Average_AUM
    print("Changes in GMV portfolio allocation in simulated portfolios in 1-month forward:")
    print("\n")
    print(round(sim_allocation0,3))
    print("\n")
    print("Average rebalanced GMV portfolio turnover in 1-month forward:")
    print(round(GMV_turnover,3))
    
    plt.plot(GMV_SD*np.sqrt(months),GMV_r*months,color='b',marker='o')
    plt.title("GMV portfolio simulation in mean-standard deviation space")
    plt.xlabel("Annualized standard deviation")
    plt.ylabel("Annual expected return")
    plt.show() 
    
    plt.plot(GMV_turnover.iloc[:,-2],GMV_turnover.iloc[:,-1],color='b')
    plt.title("1-month forward GMV portfolio turnover")
    plt.xlabel("Portfolio turnover")
    plt.ylabel("Portfolio turnover ratio")
    plt.show()
    
    # Plot the 1,000 MSR portfolio's expected returns and standard deviations in mean-standard deviation space 
    simulation1 = np.zeros((required_simulation+1,6))
    simulation1[0,0]=MSR_SD
    simulation1[0,1]=MSR_r
    simulation1[0,2]=MSR_w.T[:,0]
    simulation1[0,3]=MSR_w.T[:,1]
    simulation1[0,4]=MSR_w.T[:,2]
    simulation1[0,5]=MSR_w.T[:,3]
    fc_simulation1 = np.zeros((required_simulation+1,5))
    fc_simulation1[0,0] = MSR_w.T[:,0]*Average_AUM
    fc_simulation1[0,1] = MSR_w.T[:,1]*Average_AUM
    fc_simulation1[0,2] = MSR_w.T[:,2]*Average_AUM
    fc_simulation1[0,3] = MSR_w.T[:,3]*Average_AUM
    fc_simulation1[0,4] = fc_simulation1[0,0]+fc_simulation1[0,1]+fc_simulation1[0,2]+fc_simulation1[0,3]
    i=1
    while i<=required_simulation:
            df1=np.random.multivariate_normal(np.ravel(mean_vector),cov_matrix,n_rows)
            n_rows1=df1.shape[0]
            n_col1=df1.shape[1]
            row_vector1=np.matrix(np.ones(n_rows1)).T #col_vectorit vector
            mean_vector1=df1.T*row_vector1/n_rows1 #mean_vector0 vector
            cov_matrix1=(df1-mean_vector1.T).T*(df1-mean_vector1.T)/(n_rows1-1)
            col_vector1=np.matrix(np.ones(n_col1)).T
            cov_matrix1=(df1-mean_vector1.T).T*(df1-mean_vector1.T)/(n_rows1-1)
            MSR_w1 = (pow(cov_matrix1,-1)*mean_vector1)/(col_vector1.T*pow(cov_matrix1,-1)*mean_vector1)
            MSR_r1 = MSR_w1.T*mean_vector
            MSR_SD1 = np.sqrt(MSR_w1.T*cov_matrix*MSR_w1)
            simulation1[i,0]=MSR_SD1
            simulation1[i,1]=MSR_r1
            simulation1[i,2]=MSR_w1.T[:,0]
            simulation1[i,3]=MSR_w1.T[:,1]
            simulation1[i,4]=MSR_w1.T[:,2]
            simulation1[i,5]=MSR_w1.T[:,3]
            fc_simulation1[i,0] = (MSR_w1.T[:,0]*Average_AUM)
            fc_simulation1[i,1] = (MSR_w1.T[:,1]*Average_AUM)
            fc_simulation1[i,2] = (MSR_w1.T[:,2]*Average_AUM)
            fc_simulation1[i,3] = (MSR_w1.T[:,3]*Average_AUM)
            fc_simulation1[i,4] = fc_simulation1[0,0]+fc_simulation1[0,1]+fc_simulation1[0,2]+fc_simulation1[0,3]
            plt.plot(MSR_SD1*np.sqrt(months),MSR_r1*months,color='green',marker='o', markersize=0.5)
            i=i+1
    # # 11. Make a table with the summary statistics for the re-sampled MSR portfolio simulation[i,0]=MSR_SD1
    print("Portfolio weights in MSR method:")
    print("\n")
    MSR_Weights = pd.DataFrame(MSR_w,index=Tickers,columns=["Weight"])
    MSR_Weights = MSR_Weights.T
    print(np.round(MSR_Weights,2))
    print("\n")
    print("\n")
    print("MSR Portfolio Simulation with Standard Deviation, Expected Return, and Weights:")
    print("\n")
    MSR = pd.DataFrame(simulation1[:1000],columns=['Standard Deviation','Expected Return',
                                              f'Weight of {Tickers[0]}',f'Weight of {Tickers[1]}',
                                              f'Weight of {Tickers[2]}',f'Weight of {Tickers[3]}'])
    print(round(MSR,3))
    print("\n")
    print("Summary Statistics for re-sampled MSR portfolio:")
    print("\n")
    print(round(MSR.describe(),3))
    print("\n")
    print("Simulation of MSR Sharpe Ratio:")
    print("\n")
    rf1 = 0
    MSR_Sharpe_ratio = round(pd.DataFrame((simulation1[:1000,1]-rf1)/simulation1[:1000,0],columns=['MSR Sharpe Ratio']),3)
    print(MSR_Sharpe_ratio)
    print("\n")
    print("Summary Statistics of MSR Sharpe Ratio")
    sum_stat1 = round(MSR_Sharpe_ratio.describe(),3)
    print(sum_stat1)
    print("\n")
    MSR_allocation = MSR_Weights*Average_AUM
    MSR_allocation.loc[:,'Total'] = MSR_allocation.sum(numeric_only=True,axis=1)
    MSR_allocation = pd.concat([MSR_allocation]*1000,ignore_index=True)
    sim_allocation1 = pd.DataFrame(fc_simulation0[:1000],columns=GMV_allocation.columns)
    MSR_turnover =  MSR_allocation.subtract(sim_allocation1,axis=1)
    MSR_turnover = MSR_turnover.drop(['Total'],axis=1)
    MSR_turnover.loc[:,'Turnover'] = abs(MSR_turnover).sum(numeric_only=True,axis=1)/2
    MSR_turnover['Portfolio turnover ratio'] = MSR_turnover['Turnover']/Average_AUM
    print("Changes in MSR portfolio allocation in simulated portfolios in 1-month forward:")
    print("\n")
    print(round(sim_allocation1,3))
    print("\n")
    print("Average rebalanced MSR portfolio turnover in 1-month forward:")
    print(round(MSR_turnover,3))
    
    plt.plot(MSR_SD*np.sqrt(months),MSR_r*months,color='b',marker='o')
    plt.xlim(0.25,0.7)
    plt.ylim(0.25,0.7)
    plt.title("MSR portfolio simulation in mean-standard deviation space")
    plt.xlabel("Annualized standard deviation")
    plt.ylabel("Annual expected return")
    plt.show() 
    
    plt.plot(MSR_turnover.iloc[:,-2],MSR_turnover.iloc[:,-1],color='b')
    plt.title("1-month forward MSR portfolio turnover")
    plt.xlabel("Portfolio turnover")
    plt.ylabel("Portfolio turnover ratio")
    plt.show()
    
#def my(year,frequency,Implied_volatility, months,steps,required_simulation,Average_AUM)
my('5y','1mo',0.5,12,100,1000,1000000)
new_1 = input("Ticker for stock 1: ").upper()
new_2 = input("Ticker for stock 2: ").upper()
new_3 = input("Ticker for stock 3: ").upper()
new_4 = input("Ticker for stock 4: ").upper()
np.random.seed(123)
Tickers = [new_1,new_2,new_3,new_4]
my('5y','1mo',0.5,12,100,1000,1000000)
