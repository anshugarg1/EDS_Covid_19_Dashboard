import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)
import pandas as pd

from scipy import signal

def get_doubling_time_via_regression(in_array):
    ''' Use a linear regression to approximate the doubling rate
        Input Parameters:- in_array : pandas.series
        Outputs:- Doubling rate: double, after how many days number of infected people count will double.
    '''
    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1, 1)
    
    assert len(in_array)==3
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_

    return intercept/slope


def savgol_filter(df_input,column='confirmed',window=5):
    ''' Savgol filter for the purpose of smoothing the data without distorting it'''
    ''' Savgol Filter which can be used in groupby apply function (data structure kept)
        Input:-
        df_input : pandas.series - data for each individual contry and state undergoes filtering
        column : name
        window : used data points to calculate the filter result
        Output:-
        df_result: the index of the df_input has to be preserved in result
    '''
    degree=1
    df_result=df_input

    filter_in=df_input[column].fillna(0) # attention with the neutral element here

    result=signal.savgol_filter(np.array(filter_in),
                           window, # window size used for filtering
                           degree)
    df_result[str(column+'_filtered')]=result
    return df_result


def calc_filtered_data(df_input,filter_on='confirmed'):
    '''  Calculate savgol filter and return merged data frame.
        Input:-
        df_input: input data frame 
        filter_on: defines the used column
        Output:-
        df_output: the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Error in calc_filtered_data not all columns in data frame'

    df_output=df_input.copy() # copy otherwise the filter_on column will be overwritten
    pd_filtered_result=df_output[['state','country',filter_on]].groupby(['state','country']).apply(savgol_filter)#.reset_index()
    df_output=pd.merge(df_output,pd_filtered_result[[str(filter_on+'_filtered')]],left_index=True,right_index=True,how='left')
    
    return df_output.copy()


def rolling_reg(df_input,col='confirmed'):
    ''' Rolling Regression to approximate the doubling time. '''
    days_back=3
    result=df_input[col].rolling(
                window=days_back,
                min_periods=days_back).apply(get_doubling_time_via_regression,raw=False)
    return result


def calc_doubling_rate(df_input,filter_on='confirmed'):
    ''' Calculate approximated doubling rate and return merged data frame
        Input:-
        df_input: input data frame 
        filter_on: defines the used column
        Output:-
        df_output: the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Error in calc_doubling_rate not all columns in data frame'

    pd_DR_result= df_input.groupby(['state','country']).apply(rolling_reg,filter_on).reset_index()

    pd_DR_result=pd_DR_result.rename(columns={filter_on:filter_on+'_DR',
                             'level_2':'index'})

    #we do the merge on the index of our big table and on the index column after groupby
    df_output=pd.merge(df_input,pd_DR_result[['index',str(filter_on+'_DR')]],left_index=True,right_on=['index'],how='left')
    df_output=df_output.drop(columns=['index'])
    print(df_output.head(5))
    return df_output



if __name__ == '__main__':
    print('Start: Feature building.')
    pd_JH_data=pd.read_csv('../../data/processed/COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
    #print(pd_JH_data.head(4))
    
    # sorting - we assume the sliding window; going from top to bottom step by step.
    pd_JH_data=pd_JH_data.sort_values('date',ascending=True).copy()
    
    # Index reset to compensate the dropped index during sorting preventing out of order results.
    pd_result_larg=calc_filtered_data(pd_JH_data)
    
    # doubling rate for the non filtered data.
    pd_result_larg=calc_doubling_rate(pd_result_larg)
    
    # doubling rate for filtered data.
    pd_result_larg=calc_doubling_rate(pd_result_larg,'confirmed_filtered')

    # Masking data (lower than 100 doubling rate) with NaN which has  (better visual rendering)
    mask=pd_result_larg['confirmed']>100
    pd_result_larg['confirmed_filtered_DR']=pd_result_larg['confirmed_filtered_DR'].where(mask, other=np.NaN)
    pd_result_larg.to_csv('../../data/processed/COVID_final_set.csv',sep=';',index=False)
    #print(pd_result_larg[pd_result_larg['country']=='Germany'].tail())
    print('Complete: Feature Building.')