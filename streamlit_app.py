#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1st app to be deployed in streamlit.app

Created on Thu Apr  4 16:02:35 2024

@author: user
"""

import sys    # exit()

from typing   import Tuple

import pandas as pd # read_csv(), class DataFrame
import seaborn as sn
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter

import altair as alt
import streamlit as st

from sklearn.preprocessing    import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

def load_data() -> pd.DataFrame:
    data = pd.DataFrame( \
        {
        'Date' : [
                '2024-02-25', '2024-02-26', '2024-02-27', '2024-02-28', '2024-02-29', '2024-03-01', 
                '2024-03-02', '2024-03-03', '2024-03-04', '2024-03-05', '2024-03-06', '2024-03-07', 
                '2024-03-08', '2024-03-09', '2024-03-10', '2024-03-11', '2024-03-12', '2024-03-13', 
                '2024-03-14', '2024-03-15', '2024-03-16', '2024-03-17', '2024-03-18', '2024-03-19', 
                '2024-03-20', '2024-03-21', '2024-03-22', '2024-03-23', '2024-03-24', '2024-03-25', 
                '2024-03-26'
            ],
        'ADAUSDT' : [
                0.61910000, 0.62390000, 0.62870000, 0.65470000, 0.71950000, 0.74160000, 0.72790000, 
                0.77010000, 0.69230000, 0.73490000, 0.74270000, 0.72280000, 0.74160000, 0.71660000, 
                0.77540000, 0.74810000, 0.76380000, 0.75090000, 0.72740000, 0.65910000, 0.68140000, 
                0.66030000, 0.58640000, 0.63960000, 0.63180000, 0.61470000, 0.62320000, 0.64680000, 
                0.65670000, 0.66500000, 0.64210000
            ],
        'BNBUSDT' : [
                401.60000000, 394.60000000, 414.60000000, 399.40000000, 407.40000000, 410.90000000, 
                414.50000000, 418.40000000, 394.10000000, 429.40000000, 474.60000000, 485.80000000, 
                488.30000000, 528.90000000, 523.00000000, 537.50000000, 630.50000000, 603.20000000, 
                632.70000000, 576.40000000, 571.70000000, 555.40000000, 507.70000000, 556.80000000, 
                553.80000000, 553.80000000, 551.90000000, 567.70000000, 587.00000000, 580.40000000, 
                568.80000000
            ],
        'BTCUSDT' : [
                54476.47000000, 57037.34000000, 62432.10000000, 61130.98000000, 62387.90000000, 
                61987.28000000, 63113.97000000, 68245.71000000, 63724.01000000, 66074.04000000, 
                66823.17000000, 68124.19000000, 68313.27000000, 68955.88000000, 72078.10000000, 
                71452.01000000, 73072.41000000, 71388.94000000, 69499.85000000, 65300.63000000, 
                68393.48000000, 67609.99000000, 61937.40000000, 67840.51000000, 65501.27000000, 
                63796.64000000, 63990.01000000, 67209.99000000, 69880.01000000, 69988.00000000, 
                68581.75000000
            ],
        'ETHUSDT' : [
                3175.94000000, 3242.36000000, 3383.10000000, 3340.09000000, 3433.43000000, 
                3421.40000000, 3487.81000000, 3627.76000000, 3553.65000000, 3818.59000000, 
                3868.76000000, 3883.36000000, 3905.21000000, 3878.47000000, 4064.80000000, 
                3979.96000000, 4004.79000000, 3881.70000000, 3742.19000000, 3523.09000000, 
                3644.71000000, 3520.46000000, 3158.64000000, 3516.53000000, 3492.85000000, 
                3336.35000000, 3329.53000000, 3454.98000000, 3590.42000000, 3587.33000000, 
                3485.25000000
            ],
        'LTCUSDT' : [
                71.93000000, 73.97000000, 74.48000000, 79.92000000, 84.87000000, 94.49000000, 
                90.70000000, 88.95000000, 81.93000000, 85.87000000, 88.00000000, 88.32000000, 
                90.67000000, 87.43000000, 103.86000000, 97.52000000, 97.24000000, 94.08000000, 
                89.75000000, 84.12000000, 85.93000000, 86.77000000, 78.45000000, 84.65000000, 
                85.75000000, 83.40000000, 85.26000000, 89.65000000, 90.58000000, 95.77000000, 
                93.77000000
            ],
        'XRPUSDT' : [
                0.55060000, 0.58610000, 0.57500000, 0.58670000, 0.60130000, 0.64420000, 0.62710000, 
                0.64770000, 0.59120000, 0.61210000, 0.62700000, 0.62000000, 0.61970000, 0.60800000, 
                0.72300000, 0.68800000, 0.68910000, 0.66900000, 0.63420000, 0.60310000, 0.61910000, 
                0.64530000, 0.58450000, 0.61070000, 0.64040000, 0.61150000, 0.61670000, 0.63250000, 
                0.64060000, 0.63170000, 0.61260000
            ],
        }
    )
        
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    
    return data

def scale_data(data : pd.DataFrame) -> pd.DataFrame:
    result : Tuple(pd.DataFrame, pd.DataFrame) = ()
    new_data : pd.DataFrame = pd.DataFrame()
    limits = pd.DataFrame()
    
    # TODO get the limits to the variables 
    names = ['Mininum', 'Maximum']
    limits[''] = names
          
    column_list = data.columns.tolist()      
    for column in column_list:
        lim = []
        lim.append(data[column].min())
        lim.append(data[column].max())
        
        limits[column] = lim
    
    limits = limits.set_index('')
    
    # data = data.set_index('Date')

    # TODO scale original data to a numpy array
    min_max_scaler = MinMaxScaler()
    arr_scaled = min_max_scaler.fit_transform(data)

    # TODO recreate the data frame with the scaled data
    # OBS tolist() allocates a new area to avoid conflict w/ old df
    new_data = pd.DataFrame()
    
    new_data['Date'] = data.index.tolist()
    
    for ind in range(len(column_list)):
        new_data[column_list[ind]] = arr_scaled[:, ind]

    result = (new_data, limits)
    
    # Normal function termination
    return result


def main() -> int:
    # TODO get all data
    all_data = load_data()
    
    # TODO extract the symbols from the list
    symbol_list = all_data.columns.tolist()
    symbol_list.remove('Date')
    
    # TODO present the data scaling toggle
    should_scale : bool = st.toggle('Scale the data')

    if should_scale:
        st.write('Data scaling activated')

        # TODO generate a table with data maximum and minimum
        
    else:
        st.write('Data scaling inactive!')
            
    # TODO create a dropbox with all the symbols found & some default
    symbol_sel = st.multiselect(
        "Choose symbols", symbol_list, 
        [ symbol_list[ind] for ind in range(min(2, len(symbol_list)) )] 
    )
        
    if not symbol_sel:
        st.error("Please select at least one symbol.")

    else:
        chosen_list = list(symbol_sel)
        # st.write(f'Chosen: {chosen_list}')
        
        # TODO filter the data to the chosen symbols
        data = pd.DataFrame()
        data['Date'] = all_data['Date'].tolist()
        for chosen in chosen_list:
            data[chosen] = all_data[chosen].tolist()
        
        # print(f'data\n{data}')

        # TODO set the table title 
        title : str = "### Cryptocoin prices (USDT)"
            
        # TODO if data should be scaled
        if should_scale:
            # TODO set index to date to avoid it being used in scaling
            data = data.set_index('Date')
            data, limits = scale_data(data)
            # TODO present the original data limits as a table
            st.write('### Original data limits', limits.sort_index())
            
            title = "### Cryptocoin prices normalized to the interval [0, 1]"
            
        # TODO present the data
        st.write(title, data.sort_index())
                
        # TODO meld the dataframe to be plotted
        # plt_data = pd.melt(data.reset_index(), id_vars=["Date"])
        plt_data = pd.melt(data, id_vars=["Date"])
        # print(f'plt_data:\n{plt_data}')

        # TODO draw the chart of the chosen symbols
        chart = (
            alt.Chart(plt_data)
            .mark_area(opacity=0.3)
            .encode(
                x="Date:T",
                y=alt.Y("value:Q", stack=None),
                color="variable:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)
       
        if len(chosen_list) == 1:
            # TODO treat it like a time series
            
            # BUG make sure `data` is indexed
            data = data.set_index('Date')
            
            # result = seasonal_decompose(data, model='multiplicative')
            result = seasonal_decompose(data, model='additive')
            # figure = result.plot()
            
            # TODO create 4 piled plots: observed, trend, seasonal, resid
            figure, axs = plt.subplots(4)
            
            # TODO set titles
            figure.suptitle('Decomposition of the time series')
            
            # TODO plot observed data in memory
            axs[0].plot(data.index, result.observed)
            axs[0].label_outer()
            axs[0].set_ylabel('Observed')
            
            # TODO plot trend data in memory 
            axs[1].plot(data.index, result.trend)
            axs[1].label_outer()
            axs[1].set_ylabel('Trend')
            
            # TODO plot seasonal data in memory 
            axs[2].plot(data.index, result.seasonal)
            axs[2].label_outer()
            axs[2].set_ylabel('Seasonal')
            
            # TODO plot residual data in memory 
            # Define the date format
            date_form = DateFormatter("%m-%d")
            axs[3].xaxis.set_major_formatter(date_form)
            axs[3].plot(data.index, result.resid)
            axs[3].set_ylabel('Residual')
            
            # TODO send the final image to streamlit
            st.pyplot(figure)

        else:
            # TODO does multivariable statistics
            
            # BUG make sure `data` is indexed
            data = data.set_index('Date')
            
            corr_matrix = data.corr()
            
            corr_title = 'Pearson correlation between cryptocoins'
            should_square : bool = st.toggle('Square the correlations')
            if should_square:
                r2 = corr_matrix.map(lambda x: x * x)
                corr_matrix = r2
                
                # TODO change heatmap titles
                corr_title = 'Squared Pearson correlation between cryptoins'
                
            # TODO set heatmap titles
            fig,ax = plt.subplots() 
            fig.suptitle(corr_title)
            
            sn.heatmap(corr_matrix, annot=True, ax=ax)
            
            st.pyplot(fig)


    # Normal function return
    return 0

if __name__ == '__main__':
    sys.exit(main())