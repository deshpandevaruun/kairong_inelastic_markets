import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 300)
import os
import pickle
from scipy.stats.mstats import winsorize

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
#https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/quarterly-update/mutual-funds/monthly-returns/
base_path = '/Users/varuundeshpande/Desktop/Columbia_MSFE/RA_2024/Kairong/InelasticMarkets'

monthly_return = file_open('/Users/varuundeshpande/Desktop/Columbia_MSFE/RA_2024/Kairong/InelasticMarkets/fund_flows_1')

print('files read')
fund_summary_ActiveEq = pd.read_csv(os.path.join(base_path, 'fund_summary_bond.csv'))

monthly_return['caldt'] = pd.to_datetime(monthly_return['caldt'])
fund_summary_ActiveEq['caldt'] = pd.to_datetime(fund_summary_ActiveEq['caldt'])
monthly_return['mret'] = monthly_return['mret'].replace('R', np.nan)
monthly_return['mret'] = pd.to_numeric(monthly_return['mret'], errors='coerce')
monthly_return['mtna'] = pd.to_numeric(monthly_return['mtna'], errors='coerce')

monthly_return['year'] = monthly_return['caldt'].dt.year
monthly_return['quarter'] = monthly_return['caldt'].dt.quarter

fund_summary_ActiveEq['year'] = fund_summary_ActiveEq['caldt'].dt.year
fund_summary_ActiveEq['quarter'] = fund_summary_ActiveEq['caldt'].dt.quarter
fund_summary_ActiveEq = fund_summary_ActiveEq.drop('caldt', axis=1)

fund_flows_1_data_post_90 = monthly_return[monthly_return['caldt'].dt.year >= 1991]

equity_mutual_funds_fundno = fund_summary_ActiveEq['crsp_fundno'].unique()
#equity_mutual_funds_ret = fund_flows_1_data_post_90[fund_flows_1_data_post_90['crsp_fundno'].isin(equity_mutual_funds_fundno)]
#bond_mutual_funds_ret = fund_flows_1_data_post_90[fund_flows_1_data_post_90['crsp_fundno'].isin(bond_mutual_funds_fundno)]


equity_mutual_funds_ret = pd.merge(fund_flows_1_data_post_90, fund_summary_ActiveEq, on = ['year','quarter','crsp_fundno'], how = 'inner')
equity_mutual_funds_ret['caldt'] = equity_mutual_funds_ret['caldt'] + pd.offsets.MonthEnd(0)
#bond_mutual_funds_ret['caldt'] = bond_mutual_funds_ret['caldt'] + pd.offsets.MonthEnd(0)


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
print('fmonths added')
# Apply the function to each group
result_equity = grouped.apply(add_missing_months).reset_index(drop=True)

#grouped_bond = bond_mutual_funds_ret.groupby('crsp_fundno')
#result_bond = grouped_bond.apply(add_missing_months).reset_index(drop=True)


def flow(group):
    mtna_shifted = group['mtna'].shift(1)
    
    flow_i = group['mtna'] - (1 + group['mret'])*mtna_shifted
    return flow_i


#result_bond['flow'] = result_bond.groupby('crsp_fundno').apply(lambda x: flow(x)).reset_index(level=0, drop=True)
result_equity['flow'] = result_equity.groupby('crsp_fundno').apply(lambda x: flow(x)).reset_index(level=0, drop=True)
print(result_equity)
result_equity_cleaned = result_equity[~result_equity['flow'].isin([np.inf, -np.inf])]
#result_equity_cleaned['flow_winsorize'] = winsorize(result_equity_cleaned['flow'], limits = [0.01, 0.01])

percentile_1 = result_equity_cleaned['flow'].quantile(0.01)
percentile_99 = result_equity_cleaned['flow'].quantile(0.99)

# Filter the DataFrame
filtered_result_equity = result_equity_cleaned[(result_equity_cleaned['flow'] >= percentile_1) & (result_equity_cleaned['flow'] <= percentile_99)]
#filtered_result_equity = result_equity_cleaned
#filtered_result_equity['flow'] = winsorize(filtered_result_equity['flow'], limits=[0.01, 0.01])


filtered_result_equity = filtered_result_equity.rename(columns={'index': 'caldt'})

equity_total_tna = filtered_result_equity.groupby('caldt')['mtna'].sum().reset_index()
equity_total_tna = equity_total_tna.rename(columns={'mtna': 'mtna_total'})
#equity_total = pd.merge(equity_total_tna, filtered_result_equity, on = 'caldt', how = 'inner')

#equity_total['mtna_weights'] = equity_total['mtna']/equity_total['mtna_total']
#equity_total['weights_times_flow'] = equity_total['flow']*equity_total['mtna_weights']


total_flow = filtered_result_equity.groupby('caldt')['flow'].sum().reset_index()
total_flow = pd.merge(total_flow, equity_total_tna, on = 'caldt', how = 'inner')
total_flow['flow%'] = 100*total_flow['flow']/total_flow['mtna_total']

print(total_flow)
plt.plot(total_flow['caldt'] ,total_flow['flow%'])
#total_flow['caldt'] = pd.to_datetime(total_flow['caldt'])

print(total_flow[total_flow['caldt'].dt.year == 2020])

#print(total_flow[total_flow['caldt'].dt.year == 2003])


