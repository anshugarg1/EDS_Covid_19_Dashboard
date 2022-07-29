import pandas as pd
import numpy as np
from datetime import datetime
import pandas as pd 
from scipy import optimize
from scipy import integrate
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import dash
dash.__version__
from dash import dcc as dcc
from dash import html as html
from dash.dependencies import Input, Output,State
import plotly.graph_objects as go
import os




# first generate simulated data (number of infected people) using beta and gamma values (default value) using SIR_model_t.
def SIR_model_t(SIR,t,beta,gamma):
    ''' Simple SIR model
        S: susceptible population
        t: time step, mandatory for integral.odeint
        I: infected people
        R: recovered people
        beta: 
        
        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)
    
    '''
    
    S,I,R=SIR
    dS_dt=-beta*S*I/N0          #S*I is the 
    dI_dt=beta*S*I/N0-gamma*I
    dR_dt=gamma*I
    return dS_dt,dI_dt,dR_dt


#second step: find hyperparameters (beta and gamma) parameters by finding a function (using optimize.curve). Here fit_odeint func 
#outputs the integration values for S, I, R variable. optimize.curve uses simulated y data and x data to find the estimated function 
#(by finialising the hyperparameter values). And finally fitting that estimated function on the simulated data function (using fit_odeint). 

# the resulting curve has to be fitted
# free parameters are here beta and gamma
def fit_odeint(x, beta, gamma):
    '''helper function for the integration.'''
    return integrate.odeint(SIR_model_t, (S0, I0, R0), t, args=(beta, gamma))[:,1] # we only would like to get dI




if __name__ == '__main__':
    #data loading
    df_analyse=pd.read_csv('../data/processed/COVID_small_flat_table.csv',sep=';')  
    df_analyse = df_analyse.sort_values('date',ascending=True)

    print(df_analyse.columns)
    c_list = df_analyse.columns[1:]

    SIR_df = pd.DataFrame()
    #N0=10000000  # total population of Germany country
    #approximately 1/8th of original population of each country is considered.

    N0_dict = {}
    N0_dict['Italy'] = 7000000        # original 59M - 7M
    N0_dict['US'] = 41000000         #original 329M - 41M
    N0_dict['Spain'] = 6000000      #original 47M - 6M
    N0_dict['Germany'] = 10000000    #original 80M  - 10M
    N0_dict['Korea, South'] = 7000000   #original 51M  - 7M

    for country in c_list:
        ydata = np.array(df_analyse[country][40:150])
        t=np.arange(len(ydata))    
        N0 = N0_dict[country]
        I0=ydata[0]
        S0=N0-I0
        R0=0
        print('start infected:',I0)
        print('cumulative sum of invected after period',ydata[-1])
        print('Number of days',len(ydata))
        print('N0',N0)

        col_name = country+'_orig'
        print(col_name)
        SIR_df[col_name] = ydata
        popt, pcov = optimize.curve_fit(fit_odeint, t, ydata)
        perr = np.sqrt(np.diag(pcov))

        print('standard deviation errors : ',str(perr), ' start infect:',ydata[0])
        print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])

        # get the final fitted curve / predict the outcome 
        fitted=fit_odeint(t, *popt)
        col_name = country+'_SIR'
        print(col_name)
        SIR_df[col_name] = fitted
        print("------------------------------")

    SIR_df.to_csv('../data/processed/SIR_df.csv')