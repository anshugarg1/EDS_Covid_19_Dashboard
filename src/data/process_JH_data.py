import pandas as pd
import numpy as np

from datetime import datetime

def store_relational_JH_dataset():
    ''' Transformes the COVID data into a relational data set.'''

    data_path='../../data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw=pd.read_csv(data_path)

    pd_data_base=pd_raw.rename(columns={'Country/Region':'country','Province/State':'state'})

    pd_data_base['state']=pd_data_base['state'].fillna('no')

    pd_data_base=pd_data_base.drop(['Lat','Long'],axis=1)
    #print(pd_data_base.columns)
    #print('-------------')
    #print(pd_data_base.set_index(['state','country']).head(3))
    #print('-------------')
    #print(pd_data_base.set_index(['state','country']).T.head(3))
    #print('-------------')
    #print(pd_data_base.set_index(['state','country']).T.stack(level=[0,1]).head(30))
    #print('-------------')
    #print(pd_data_base.set_index(['state','country']).T.stack(level=[0,1]).reset_index().head(30))
    #print('-------------')
    
    pd_relational_model=pd_data_base.set_index(['state','country']) \
                                .T                              \
                                .stack(level=[0,1])             \
                                .reset_index()                  \
                                .rename(columns={'level_0':'date',
                                                   0:'confirmed'},
                                                  )

    pd_relational_model['date']=pd_relational_model.date.astype('datetime64[ns]')
    print(pd_relational_model.columns)
    print(pd_relational_model.head(5))
    pd_relational_model.to_csv('../../data/processed/COVID_relational_confirmed.csv',sep=';',index=False)
    print(' Number of rows stored: '+str(pd_relational_model.shape[0]))
    print(' Latest date is: '+str(max(pd_relational_model.date)))
    
    
    
if __name__ == '__main__':

    store_relational_JH_dataset()
    
    