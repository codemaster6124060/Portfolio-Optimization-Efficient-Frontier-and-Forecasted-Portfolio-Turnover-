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
fred = Fred(api_key='0bbc318d3ba2efaf9d4e56708954067d')
rf = fred.get_series('DTB6')[-1]/100
S1 = input("Ticker for stock 1: ").upper()
S2 = input("Ticker for stock 2: ").upper()
S3 = input("Ticker for stock 3: ").upper()
S4 = input("Ticker for stock 4: ").upper()
np.random.seed(123)
Tickers = [S1,S2,S3,S4]
print(f'selected tickers: {Tickers}')

def my(year,is_rate,os_rate,frequency,Implied_volatility, months,steps,required_simulation,NAV):
    # year = 5
    # is_rate = 0.75
    # os_rate = 0.25
    # frequency = '1mo'
    # Implied_volatility = 0.5
    # months = 12
    # steps = 100
    # required_simulation = 1000
    # NAV = 1000000
    
    #1. Collect returns data for 4 (or more) different assets, which have in T observations, where T is 5 years or more.
    df = np.matrix(yf.Tickers(Tickers).history(f'{year}y',frequency).Close.dropna().pct_change().dropna())
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
    plt.title("Efficient Frontier of the portfolio")
    plt.xlabel("Annualized standard deviation")
    plt.ylabel("Annual expected return")
    plt.show()
    
    #8.	Plot the 1,000 GMV and MSR (on a separate figure each for the GMV and MSR portfolios) 
    #portfolios’ expected returns and standard deviations in mean-standard deviation space
    # Plot the 1,000 GMV portfolio's expected returns and standard deviations in mean-standard deviation space 
    simulation0 = np.zeros((required_simulation+1,6))
    simulation0[0,0]=GMV_SD
    simulation0[0,1]=GMV_r
    simulation0[0,2:5]=GMV_w.T[:,0:3]
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
            simulation0[i,2:5]=GMV_w0.T[:,0:3]
            plt.plot(GMV_SD0*np.sqrt(months),GMV_r0*months,color='green',marker='o', markersize=0.5)
            i = i+1
    #5.	Assume that asset returns conform to a multivariate normal distribution, 
    #with mean and covariance matrix equal to the sample ones, which were estimated in Step 2.
    #6.	Simulate 1,000 independent samples for each asset from the multivariate normal distribution 
    # with mean μ_T and covariance matrix Σ_T, with each draw consisting of T returns.
    #7. '
    print(df)
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
    
    plt.plot(GMV_SD*np.sqrt(months),GMV_r*months,color='b',marker='o')
    plt.title("GMV portfolio simulation in mean-standard deviation space")
    plt.xlabel("Annualized standard deviation")
    plt.ylabel("Annual expected return")
    plt.show() 
    
    
    # Plot the 1,000 MSR portfolio's expected returns and standard deviations in mean-standard deviation space 
    simulation1 = np.zeros((required_simulation+1,6))
    simulation1[0,0]=MSR_SD
    simulation1[0,1]=MSR_r
    simulation1[0,2:5]=MSR_w.T[:,0:3]
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
            simulation1[i,2:5]=MSR_w1.T[:,0:3]
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
    
    plt.plot(MSR_SD*np.sqrt(months),MSR_r*months,color='b',marker='o')
    plt.xlim(0.25,0.7)
    plt.ylim(0.25,0.7)
    plt.title("MSR portfolio simulation in mean-standard deviation space")
    plt.xlabel("Annualized standard deviation")
    plt.ylabel("Annual expected return")
    plt.show() 
    
    
    #12. Run 1-month forward out of sample test using 75% of your sample 
    #    as the base sample and then the remaining 25% for out-of-sample
    df = np.matrix(yf.Tickers(Tickers).history(f'{year}y',frequency).Close.dropna().pct_change().dropna())
    
    # Get the base-sample data
    is_df = df[:(int(len(df)*is_rate)),:]
    is_rows = is_df.shape[0]
    is_col = is_df.shape[1]
    is_row_vector = np.matrix(np.ones(is_rows)).T #creating a row vector (i.e.,is_rows x 1 vector)
    is_mean_vector = is_df.T*is_row_vector/is_rows  #sample mean vector
    is_cov_matrix = (is_df-is_mean_vector.T).T*(is_df-is_mean_vector.T)/(is_rows-1) #sample covariance matrix
    is_col_vector=np.matrix(np.ones(is_col)).T #create a col_vectorit vector (i.e., is_col x 1 vector) with columns
    #in-sample GMV weights
    is_GMV_w = (pow(is_cov_matrix,-1)*is_col_vector)/(is_col_vector.T*pow(is_cov_matrix,-1)*is_col_vector) #in-sample GMV portfolio weights
    is_GMV_Weights = pd.DataFrame(is_GMV_w,index=Tickers,columns=["Weight"])
    is_GMV_Weights = is_GMV_Weights.T
    #in-sample MSR weights
    is_MSR_w = (pow(is_cov_matrix,-1)*is_mean_vector)/(is_col_vector.T*pow(is_cov_matrix,-1)*is_mean_vector) #in-sample MSR portfolio weights
    is_MSR_Weights = pd.DataFrame(is_MSR_w,index=Tickers,columns=["Weight"])
    is_MSR_Weights = is_MSR_Weights.T
    
    # Get out-of-sample return vector
    os_df = np.matrix(yf.Tickers(Tickers).history(f'{int(year*months*os_rate)}mo',frequency).Close.dropna().pct_change().dropna())
    
    # Generate GMV portfolio turnover for 1-month forward
    os_GMV = np.zeros((len(os_df)+1,6))
    i = 1
    while i<=len(os_df):
        rs_df = df[:len(is_df)+i]
        rs_rows = rs_df.shape[0]
        rs_col = rs_df.shape[1]
        rs_row_vector = np.matrix(np.ones(rs_rows)).T #creating a row vector (i.e.,is_rows x 1 vector)
        rs_mean_vector = rs_df.T*rs_row_vector/rs_rows  #sample mean vector
        rs_cov_matrix = (rs_df-rs_mean_vector.T).T*(rs_df-rs_mean_vector.T)/(rs_rows-1) #sample covariance matrix
        rs_col_vector=np.matrix(np.ones(rs_col)).T #create a col_vectorit vector (i.e., is_col x 1 vector) with columns
        rs_GMV_w = (pow(rs_cov_matrix,-1)*rs_col_vector)/(rs_col_vector.T*pow(rs_cov_matrix,-1)*rs_col_vector) #GMV portfolio weights
        
        os_rows = os_df.shape[0]
        os_col = os_df.shape[1]
        os_row_vector = np.matrix(np.ones(os_rows)).T #creating a row vector (i.e.,is_rows x 1 vector)
        os_mean_vector = os_df.T*os_row_vector/os_rows  #sample mean vector
        
        rs_GMV_r = rs_GMV_w.T*os_mean_vector #GMV portfolio return
        rs_GMV_SD = np.sqrt(rs_GMV_w.T*rs_cov_matrix*rs_GMV_w) #GMV portfolio standard deviation
        os_GMV[i,0] = rs_GMV_r
        os_GMV[i,1] = rs_GMV_SD
        os_GMV[i,2:6] = rs_GMV_w.T
        i = i+1
        
    print("\n")
    print("Changes in out-of-sample GMV portfolio allocation in portfolios in 1-month forward:")
    print("\n")
    os_GMV = pd.DataFrame(os_GMV,columns=['Expected Return','Standard Deviation',
                                              f'Weight of {Tickers[0]}',f'Weight of {Tickers[1]}',
                                              f'Weight of {Tickers[2]}',f'Weight of {Tickers[3]}'])
    os_GMV.index.name = "Months"
    os_GMV = os_GMV.tail(-1)
    print(round(os_GMV,3))
    
    #Get the base sample allocation
    is_GMV_allocation = is_GMV_Weights*NAV
    is_GMV_allocation = pd.concat([is_GMV_allocation]*len(os_df),ignore_index=True)
    #Get the out_of_sample allocation
    os_GMV_allocation = os_GMV.iloc[:, 2:]*NAV
    os_GMV_allocation.columns = is_GMV_allocation.columns
    #rebalancing the portfolio allocation
    os_GMV_turnover = is_GMV_allocation.reset_index(drop=True)-os_GMV_allocation.reset_index(drop=True)
    os_GMV_turnover.loc[:,'Portfolio Turnover (USD)'] = abs(os_GMV_turnover).sum(numeric_only=True,axis=1)/2
    os_GMV_turnover['Portfolio turnover ratio'] = os_GMV_turnover['Portfolio Turnover (USD)']/NAV
    
    print("\n")
    print("Rebalanced GMV portfolio turnover in 1-month forward:")
    print("\n")
    print(round(os_GMV_turnover,2))
    print("\n")
    print("Average rebalanced GMV portfolio turnover in 1-month forward:")
    print("\n")
    print(round(os_GMV_turnover.mean(),2))
    
    # Generate MSR portfolio turnover for 1-month forward
    os_MSR = np.zeros((len(os_df)+1,6))
    i = 1
    while i<=len(os_df):
        rs_df = df[:len(is_df)+i]
        rs_rows = rs_df.shape[0]
        rs_col = rs_df.shape[1]
        rs_row_vector = np.matrix(np.ones(rs_rows)).T #creating a row vector (i.e.,is_rows x 1 vector)
        rs_mean_vector = rs_df.T*rs_row_vector/rs_rows  #sample mean vector
        rs_cov_matrix = (rs_df-rs_mean_vector.T).T*(rs_df-rs_mean_vector.T)/(rs_rows-1) #sample covariance matrix
        rs_col_vector=np.matrix(np.ones(rs_col)).T #create a col_vectorit vector (i.e., is_col x 1 vector) with columns
        rs_MSR_w = (pow(rs_cov_matrix,-1)*rs_mean_vector)/(rs_col_vector.T*pow(rs_cov_matrix,-1)*rs_mean_vector) #MSR portfolio weights
        
        os_rows = os_df.shape[0]
        os_col = os_df.shape[1]
        os_row_vector = np.matrix(np.ones(os_rows)).T #creating a row vector (i.e.,is_rows x 1 vector)
        os_mean_vector = os_df.T*os_row_vector/os_rows  #sample mean vector
        
        rs_MSR_r = rs_MSR_w.T*os_mean_vector #MSR portfolio return
        rs_MSR_SD = np.sqrt(rs_MSR_w.T*rs_cov_matrix*rs_MSR_w) #MSR portfolio standard deviation
        os_MSR[i,0] = rs_MSR_r
        os_MSR[i,1] = rs_MSR_SD
        os_MSR[i,2:6] = rs_MSR_w.T
        i = i+1
    
    print("\n")
    print("Changes in out-of-sample MSR portfolio allocation in portfolios in 1-month forward:")
    print("\n")
    os_MSR = pd.DataFrame(os_MSR,columns=['Expected Return','Standard Deviation',
                                              f'Weight of {Tickers[0]}',f'Weight of {Tickers[1]}',
                                              f'Weight of {Tickers[2]}',f'Weight of {Tickers[3]}'])
    os_MSR.index.name = "Months"
    os_MSR = os_MSR.tail(-1)
    print(round(os_MSR,3))
    
    #Get the base sample allocation
    is_MSR_allocation = is_MSR_Weights*NAV
    is_MSR_allocation = pd.concat([is_MSR_allocation]*len(os_df),ignore_index=True)
    #Get the out_of_sample allocation
    os_MSR_allocation = os_MSR.iloc[:, 2:]*NAV
    os_MSR_allocation.columns = is_MSR_allocation.columns
    #rebalancing the portfolio allocation
    os_MSR_turnover = is_MSR_allocation.reset_index(drop=True)-os_MSR_allocation.reset_index(drop=True)
    os_MSR_turnover.loc[:,'Portfolio Turnover (USD)'] = abs(os_MSR_turnover).sum(numeric_only=True,axis=1)/2
    os_MSR_turnover['Portfolio turnover ratio'] = os_MSR_turnover['Portfolio Turnover (USD)']/NAV
    
    print("\n")
    print("Rebalanced MSR portfolio turnover in 1-month forward:")
    print("\n")
    print(round(os_MSR_turnover,2))
    print("\n")
    print("Average rebalanced MSR portfolio turnover in 1-month forward:")
    print("\n")
    print(round(os_MSR_turnover.mean(),2))



# def my(year,is_rate,os_rate,frequency,Implied_volatility, months,steps,required_simulation,NAV)
my(5,0.75,0.25,'1mo',0.5,12,100,1000,1000000)


# Statistically equivalent portfolio with better portfolio performance and lower turnover
my(5,0.85,0.15,'1mo',0.5,12,100,1000,1000000)
