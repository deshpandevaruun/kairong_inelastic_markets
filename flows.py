#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:58:18 2024

@author: varuundeshpande
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 300)
import os
import pickle
from scipy.stats.mstats import winsorize

#https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/quarterly-update/mutual-funds/monthly-returns/
#https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/quarterly-update/mutual-funds/fund-summary/
def file_open(file_name):
    pickle_file = f'{file_name}.pkl'
    csv_file = f'{file_name}.csv'
    
    if os.path.exists(pickle_file):
        try:
            data = pd.read_pickle(pickle_file)
        except (EOFError, pickle.UnpicklingError):
            # If there's an error loading the pickle file, load from CSV and recreate the pickle file
            data = pd.read_csv(csv_file, encoding='unicode_escape')
            data.to_pickle(pickle_file)
    else:
        # Load the dataset from the CSV file and save it as a pickle for future use
        data = pd.read_csv(csv_file, encoding='unicode_escape')
        data.to_pickle(pickle_file)
    
    return data
fund_flows_1_data = file_open('/Users/varuundeshpande/Desktop/Columbia_MSFE/RA_2024/Kairong/InelasticMarkets/fund_flows_1')

fund_flows_2_data = file_open("/Users/varuundeshpande/Desktop/Columbia_MSFE/RA_2024/Kairong/InelasticMarkets/fund_flows_2") 
print('files-read')
fund_flows_1_data['mret'] = fund_flows_1_data['mret'].replace('R', np.nan)
fund_flows_1_data['mret'] = pd.to_numeric(fund_flows_1_data['mret'], errors='coerce')
fund_flows_1_data['mtna'] = pd.to_numeric(fund_flows_1_data['mtna'], errors='coerce')


fund_flows_1_data['caldt'] = pd.to_datetime(fund_flows_1_data['caldt'])
fund_flows_2_data['caldt'] = pd.to_datetime(fund_flows_2_data['caldt'])
fund_flows_1_data = fund_flows_1_data.dropna(subset=['crsp_fundno'])

fund_flows_1_data['crsp_fundno'] = fund_flows_1_data['crsp_fundno'].astype(int)
fund_flows_1_data_post_90 = fund_flows_1_data[fund_flows_1_data['caldt'].dt.year >= 1990]
fund_flows_2_data = fund_flows_2_data[['crsp_fundno', 'caldt', 'crsp_obj_cd', 'index_fund_flag', 'et_flag']]

def fill_obj_cd(group):
    unique_values = group.dropna().unique()
    if len(unique_values) == 1:
        return group.fillna(unique_values[0])
    else:
        return group
#fill up empty entries in obj_cd
fund_flows_2_data['crsp_obj_cd'] = fund_flows_2_data.groupby('crsp_fundno')['crsp_obj_cd'].transform(fill_obj_cd)

print('step 1')
#Get equity and bond funds
equity_identifiers = ['EDC', 'EDY']
bond_identifiers = ['IU', 'IC', 'IG', 'IM']
equity_mutual_funds = fund_flows_2_data[fund_flows_2_data['crsp_obj_cd'].str[0:3].isin(equity_identifiers)]

#exclude index fund
equity_mutual_funds = equity_mutual_funds[equity_mutual_funds['index_fund_flag'] != 'D']
#exclude ETF
equity_mutual_funds = equity_mutual_funds[equity_mutual_funds['et_flag'].isna()]

bond_mutual_funds = fund_flows_2_data[fund_flows_2_data['crsp_obj_cd'].str[0:2].isin(bond_identifiers)]
#exclude index fund
bond_mutual_funds = bond_mutual_funds[bond_mutual_funds['index_fund_flag'] != 'D']
#exclude ETF
bond_mutual_funds = bond_mutual_funds[bond_mutual_funds['et_flag'].isna()]

equity_mutual_funds_fundno = equity_mutual_funds['crsp_fundno'].unique()

bond_mutual_funds_fundno = bond_mutual_funds['crsp_fundno'].unique()

#remove funds with entries less than 2 years
grouped = fund_flows_1_data_post_90.groupby('crsp_fundno')
# Filter groups
fund_flows_1_data_post_90 = grouped.filter(lambda x: x['mret'].notna().sum() >= 24 and x['mtna'].notna().sum() >= 24)


#equity and bond data
equity_mutual_funds_ret = fund_flows_1_data_post_90[fund_flows_1_data_post_90['crsp_fundno'].isin(equity_mutual_funds_fundno)]
bond_mutual_funds_ret = fund_flows_1_data_post_90[fund_flows_1_data_post_90['crsp_fundno'].isin(bond_mutual_funds_fundno)]

equity_mutual_funds_ret['caldt'] = equity_mutual_funds_ret['caldt'] + pd.offsets.MonthEnd(0)
bond_mutual_funds_ret['caldt'] = bond_mutual_funds_ret['caldt'] + pd.offsets.MonthEnd(0)


grouped = equity_mutual_funds_ret.groupby('crsp_fundno')
def add_missing_months(group):
    # Get the minimum and maximum dates in the group
    min_date = group['caldt'].min()
    max_date = group['caldt'].max()
    
    # Create a date range from the overall minimum to maximum date in group
    overall_idx = pd.date_range(start=min_date, end=max_date, freq='M')
    
    # Add missing months to the group by reindexing with the overall date range
    group = group.set_index('caldt').reindex(overall_idx).reset_index()
    
    return group

# Apply the function to each group
result_equity = grouped.apply(add_missing_months).reset_index(drop=True)

grouped_bond = bond_mutual_funds_ret.groupby('crsp_fundno')
result_bond = grouped_bond.apply(add_missing_months).reset_index(drop=True)


def flow(group):
    mtna_shifted = group['mtna'].shift(1)
    mret_shifted = group['mret'].shift(1)
    flow_i = group['mtna']/mtna_shifted  - (1 + group['mret'])
    return flow_i


result_bond['flow'] = result_bond.groupby('crsp_fundno').apply(lambda x: flow(x)).reset_index(level=0, drop=True)
result_equity['flow'] = result_equity.groupby('crsp_fundno').apply(lambda x: flow(x)).reset_index(level=0, drop=True)

print('flows calculated')

result_bond_cleaned = result_bond[~result_bond['flow'].isin([np.inf, -np.inf])]
percentile_1 = result_bond_cleaned['flow'].quantile(0.01)
percentile_99 = result_bond_cleaned['flow'].quantile(0.99)

# Filter the DataFrame
#filtered_result_bond = result_bond_cleaned[(result_bond_cleaned['flow'] >= percentile_1) & (result_bond_cleaned['flow'] <= percentile_99)]

filtered_result_bond = result_bond_cleaned
filtered_result_bond['flow'] = winsorize(filtered_result_bond['flow'], limits=[0.01, 0.01])

result_equity_cleaned = result_equity[~result_equity['flow'].isin([np.inf, -np.inf])]
#result_equity_cleaned['flow_winsorize'] = winsorize(result_equity_cleaned['flow'], limits = [0.01, 0.01])

percentile_1 = result_equity_cleaned['flow'].quantile(0.01)
percentile_99 = result_equity_cleaned['flow'].quantile(0.99)

# Filter the DataFrame
#filtered_result_equity = result_equity_cleaned[(result_equity_cleaned['flow'] >= percentile_1) & (result_equity_cleaned['flow'] <= percentile_99)]

filtered_result_equity = result_equity_cleaned
filtered_result_equity['flow'] = winsorize(filtered_result_equity['flow'], limits=[0.01, 0.01])

filtered_result_equity = filtered_result_equity.rename(columns={'index': 'caldt'})
filtered_result_bond = filtered_result_bond.rename(columns={'index': 'caldt'})

equity_total_tna = filtered_result_equity.groupby('caldt')['mtna'].sum().reset_index()
equity_total_tna = equity_total_tna.rename(columns={'mtna': 'mtna_total'})

bond_total_tna = filtered_result_bond.groupby('caldt')['mtna'].sum().reset_index()
bond_total_tna = bond_total_tna.rename(columns={'mtna': 'mtna_total'})


equity_total = pd.merge(equity_total_tna, filtered_result_equity, on = 'caldt', how = 'inner')
bond_total = pd.merge(bond_total_tna, filtered_result_bond, on = 'caldt', how = 'inner')


equity_total['mtna_weights'] = equity_total['mtna']/equity_total['mtna_total']
bond_total['mtna_weights'] = bond_total['mtna']/bond_total['mtna_total']


equity_total['weights_times_flow'] = equity_total['flow']*equity_total['mtna_weights']
equity_total.dropna(subset = ['weights_times_flow'], inplace = True)


bond_total['weights_times_flow'] = bond_total['flow']*bond_total['mtna_weights']
bond_total.dropna(subset = ['weights_times_flow'], inplace = True)

equity_total_final = equity_total.groupby('caldt')['weights_times_flow'].sum().reset_index()
bond_total_final = bond_total.groupby('caldt')['weights_times_flow'].sum().reset_index()


plt.plot(bond_total_final['caldt'],  bond_total_final['weights_times_flow'])
plt.show()
plt.plot( equity_total_final['caldt'], equity_total_final['weights_times_flow'])
plt.show()

bond_total_final.to_csv('/Users/varuundeshpande/Desktop/Columbia_MSFE/RA_2024/Kairong/InelasticMarkets/bond_flows.csv')
equity_total_final.to_csv('/Users/varuundeshpande/Desktop/Columbia_MSFE/RA_2024/Kairong/InelasticMarkets/equity_flows.csv')


