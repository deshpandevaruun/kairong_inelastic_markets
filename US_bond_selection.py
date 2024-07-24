import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 300)
import os
import pickle
from scipy.stats.mstats import winsorize

#https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/quarterly-update/mutual-funds/monthly-returns/
#https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/quarterly-update/mutual-funds/fund-summary/
base_path = '/Users/varuundeshpande/Desktop/Columbia_MSFE/RA_2024/Kairong/InelasticMarkets'

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

fund_summary = file_open("/Users/varuundeshpande/Desktop/Columbia_MSFE/RA_2024/Kairong/InelasticMarkets/fund_flows_2") 


mflink = pd.read_csv(os.path.join(base_path, 'mflinks.csv'))

fund_summary['caldt'] = pd.to_datetime(fund_summary['caldt'])
fund_flows_1_data['caldt'] = pd.to_datetime(fund_flows_1_data['caldt'])

print('The shape of Fund_Summary before eliminating small funds is:', fund_summary.shape)
fund_summary.loc[fund_summary.tna_latest < 0, 'tna_latest'] = np.NaN
fund_summary = fund_summary.loc[(fund_summary['tna_latest'].isnull()) | (fund_summary['tna_latest'] > 15)]
print('The shape of Fund_Summary after eliminating small funds is:', fund_summary.shape)


print('The shape of Fund_Summary before adjusting for incubation bias is:', fund_summary.shape[0])
fund_summary = fund_summary.loc[(fund_summary['first_offer_dt'] <= fund_summary['caldt']) |
                                (fund_summary['first_offer_dt'].isnull())]
print('The shape of Fund_Summary after adjusting for incubation bias is:', fund_summary.shape[0])


mflink.wficn = mflink.wficn.apply(lambda x: int(x))
ids = mflink['crsp_fundno']
print('The number of CRSP_FUNDNOs having multiple WFICN is:', mflink[ids.isin(ids[ids.duplicated()])].shape[0])
print('Dropping duplicates...')
mflink = mflink[['crsp_fundno', 'wficn']]
mflink = mflink.drop_duplicates(subset=['crsp_fundno'], keep='first')
print('Duplicates dropped! Merging to fund summary...')
print('The shape of Fund_Summary before merging WFICN is: ', str(fund_summary.shape))
fund_summary = pd.merge(fund_summary, mflink, on='crsp_fundno', how='left')
print('The shape of Fund_Summary after merging WFICN is: ', str(fund_summary.shape))


print('The number of observations in Fund_Summary before subsetting is:', fund_summary.shape[0])
con1 = [str(x).startswith('IC') for x in fund_summary['crsp_obj_cd']]
#con2 = [str(x).startswith('IG') for x in fund_summary['crsp_obj_cd']]
#con3 = [str(x).startswith('MT') for x in fund_summary['crsp_obj_cd']]
#con = [x or y for x, y in zip(con1, con2)]
fund_summary_US_Active = fund_summary.loc[con1]
print('The number of observations in Fund_Summary after subsetting is:', fund_summary_US_Active.shape[0])


print('The number of observations in Fund_Summary before subsetting is:', fund_summary_US_Active.shape[0])
LIPPER_OBJ_CD = ["A", 'BBB', 'IUT', 'SUS', 'SUT', 'GUS', 'GUT', 'IUG', 'SIU']
con1 = fund_summary_US_Active['lipper_obj_cd'].isin(LIPPER_OBJ_CD)
SI_OBJ_CD = ['GMC', 'SCG', 'AGG', 'GRO', 'GRI', 'ING', 'OPI']
con2 = fund_summary_US_Active['lipper_obj_cd'].isnull() & fund_summary_US_Active['si_obj_cd'].isin(SI_OBJ_CD)
WBRGER_OBJ_CD = ['CHY', 'CBD', 'GOV']
con3 = fund_summary_US_Active['lipper_obj_cd'].isnull() & fund_summary_US_Active['si_obj_cd'].isnull() & \
        fund_summary_US_Active['wbrger_obj_cd'].isin(WBRGER_OBJ_CD)
con4 = fund_summary_US_Active['lipper_obj_cd'].isnull() & fund_summary_US_Active['si_obj_cd'].isnull() & \
           fund_summary_US_Active['wbrger_obj_cd'].isnull() & fund_summary_US_Active['policy'].isin(['Bonds', 'GS'])
fund_summary_US_Active = fund_summary_US_Active.loc[con1 | con2 | con3 ]
drop_list_WBRGER_OBJ_CD = ['BAL', 'IFL']
#con5 = fund_summary_US_Active['wbrger_obj_cd'].isin(drop_list_WBRGER_OBJ_CD)
con6 = fund_summary_US_Active['lipper_obj_cd'].isnull() & fund_summary_US_Active['si_obj_cd'].isnull() & \
           fund_summary_US_Active['wbrger_obj_cd'].isnull() & fund_summary_US_Active['policy'].isnull()
fund_summary_US_Active = fund_summary_US_Active.loc[~con6]
print('The number of observations in Fund_Summary after subsetting is:', fund_summary_US_Active.shape[0])


print('The number of observations in Fund_Summary before subsetting is:', fund_summary_US_Active.shape[0])
#fund_summary_US_Active = fund_summary_US_Active.loc[fund_summary_US_Active['index_fund_flag'] != 'D']
print('The number of observations in Fund_Summary after subsetting is:', fund_summary_US_Active.shape[0])


eliminated_content = ['Index', 'Ind', 'Idx', 'Indx', 'iShares', 'SPDR', 'HOLDRs', 'ETF', 'Exchange-Traded Fund', 'PowerShares', 'StreetTRACKS']
print('The number of observations in Fund_Summary before subsetting is:', fund_summary_US_Active.shape[0])
#for content in eliminated_content:
#    fund_summary_US_Active = fund_summary_US_Active.loc[
#        ~fund_summary_US_Active['fund_name'].str.lower().str.contains(content.lower(), na=False)]
print('The number of observations in Fund_Summary after subsetting is:', fund_summary_US_Active.shape[0])

print('The number of observations in Fund_Summary before subsetting is:', fund_summary_US_Active.shape[0])
#fund_summary_US_Active = fund_summary_US_Active.loc[fund_summary_US_Active['et_flag'] != 'F']
#fund_summary_US_Active = fund_summary_US_Active.loc[fund_summary_US_Active['et_flag'] != 'N']
print('The number of observations in Fund_Summary after subsetting is:', fund_summary_US_Active.shape[0])


print('The shape of Fund_Summary before eliminating observations with no crsp_fundno, fund_name, crsp_cl_grp & wficn is:')
print(fund_summary_US_Active.shape)
con7 = fund_summary_US_Active.fund_name.isnull() & fund_summary_US_Active.crsp_cl_grp.isnull() & \
        fund_summary_US_Active.wficn.isnull()
#fund_summary_US_Active = fund_summary_US_Active[~con7]
print('The shape of Fund_Summary after eliminating observations with no crsp_fundno, fund_name, crsp_cl_grp & wficn is:')
print(fund_summary_US_Active.shape)



print('The shape of Fund_Summary before eliminating Variable Annuity Underlying funds is:', fund_summary_US_Active.shape)
print('Before elimination, the frequency of Variable Annuity (Y) funds vs. the rest (N) is:')
print(fund_summary_US_Active['vau_fund'].value_counts())
#fund_summary_US_Active = fund_summary_US_Active[fund_summary_US_Active.vau_fund != 'Y']
fund_summary_US_Active.drop(['vau_fund'], axis=1, inplace=True)
print('The shape of Fund_Summary after eliminating Variable Annuity Underlying funds is:', fund_summary_US_Active.shape)


print('The shape fund_summary_US_Active before eliminating observations with no caldt is:', fund_summary_US_Active.shape)
#fund_summary_US_Active = fund_summary_US_Active[~fund_summary_US_Active.caldt.isnull()]
print('The shape fund_summary_US_Active after eliminating observations with no caldt is:', fund_summary_US_Active.shape)


print("Saving the cleaned Active Equity dataset to file...")
fund_summary_US_Active.to_csv(os.path.join(base_path, 'fund_summary_bond.csv'), index=False)
print('File saved!')
