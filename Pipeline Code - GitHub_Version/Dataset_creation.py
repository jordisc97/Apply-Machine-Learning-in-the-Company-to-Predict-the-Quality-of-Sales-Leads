import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import seaborn as sns
from matplotlib import pyplot as plt
import time


def dataset_creation():
    date_string = time.strftime("%Y-%m-%d-%Hh")
    
    ## Load
    Leads = pd.read_csv("/Users/solej/OneDrive - HP Inc/Reports/Batch for CBM/Salesforce/Leads.csv",encoding='latin-1')
    Activities = pd.read_csv("/Users/solej/OneDrive - HP Inc/Reports/Batch for CBM/Salesforce/Activities.csv",encoding='latin-1')
    Campaigns = pd.read_csv("/Users/solej/OneDrive - HP Inc/Reports/Batch for CBM/Salesforce/Influenced Campaigns.csv",encoding='latin-1')

    # Print the size of the dataset
    print("Number of rows in Leads:", len(Leads))
    print("Number of rows in Activities:",len(Activities))
    print("Number of rows in Campaigns:",len(Campaigns))


    ## Clean Activities
    Activities['Subject'] = Activities['Subject'].str.lower()
    Activities['Subject'] = Activities['Subject'].str.replace('*', '')
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('attempted'),
                                      "Attempted Calls",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('attemped'),
                                      "Attempted Calls",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('email'),
                                      "Email",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('letter'),
                                      "Email",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('reached'),
                                      "Reached Calls",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('call'),
                                      "Reached Calls",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('reach'),
                                      "Reached Calls",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('comments'),
                                      "Comments",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('mail'),
                                      "Email",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('edm'),
                                      "EDM",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('budget'),
                                      "Budget Change",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('status'),
                                      "Status Change",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('disqualified'),
                                      "Disqualified Lead",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('webhelp'),
                                      "Webhelp Lead",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.contains('re-assignment'),
                                      "Re-assigned Lead",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.startswith('inmail'),
                                      "Reached Calls",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.startswith('inmail'),
                                      "Reached Calls",Activities['Subject'])
    Activities['Subject'] = np.where(Activities['Subject'].str.startswith('attmep'),
                                      "Attempted Calls",Activities['Subject'])


    ## Activities
    ## Selecting just the most common
    Activities = Activities[Activities['Subject'].map(Activities['Subject'].value_counts()) > 300]

    ## OHE of activities
    ohe_activities = pd.get_dummies(Activities['Subject'], prefix='Activ')
    Activities2 = pd.concat([Activities, ohe_activities], axis=1, sort=False).groupby('ResponseId').max()
    Activities2.reset_index(level=0, inplace=True)
    groupby_activ = pd.DataFrame(Activities.groupby('ResponseId').size(),columns=["Activ Touches"])

    ## Merge with Lead table
    Activities3 = Activities2.drop(["HP Lead ID","Subject","First Name","Last Name"],axis=1)
    Leads2 = pd.merge(Leads, Activities3, on='ResponseId', how='left')
    Leads2 = pd.merge(Leads2, groupby_activ, on='ResponseId', how='left')

    list_cols_activ = ["Activ_Attempted Calls","Activ_Budget Change","Activ_Comments","Activ_Disqualified Lead","Activ_EDM",
                       "Activ_Email","Activ_Re-assigned Lead","Activ_Reached Calls","Activ_Status Change",
                       "Activ_closed response","Activ Touches"]
    for col in list_cols_activ:
        Leads2[col] = Leads2[col].replace(np.nan, 0)


    ## Campaigns
    ## OHE of Campaigns
    Campaigns2 = Campaigns.drop(["First Name","Last Name","Start Date","Created Date","Last Modified Date"],axis=1)
    Campaigns2['End Date'] = pd.to_datetime(Campaigns2['End Date'])
    Campaigns2 = Campaigns2.sort_values(by='End Date', ascending=False)
    ohe_campaigntype = pd.get_dummies(Campaigns2['Campaign Type'], prefix='Camp')

    Campaigns3 = pd.concat([Campaigns2, ohe_campaigntype], axis=1, sort=False).groupby('ResponseId').max()
    Campaigns3.reset_index(level=0, inplace=True)
    Campaigns3 = Campaigns3.drop(["Campaign Name","End Date","Campaign Type"],axis=1)
    
    groupby_camp = pd.DataFrame(Campaigns2.groupby('ResponseId').size(),columns=["Camp Touches"])
    Campaigns2.drop_duplicates(subset ="ResponseId", keep="first", inplace = True) 

    Campaigns4 = pd.merge(Campaigns2, groupby_camp, on='ResponseId', how='left')
    Campaigns5 = pd.merge(Campaigns4, Campaigns3, on='ResponseId', how='left')
    
    ## Merge with Lead table
    Leads3 = pd.merge(Leads2, Campaigns5, on='ResponseId', how='left')
    Leads3 = Leads3.rename(columns={"Company / Account": "Company"})

    
    ## Save the files and create the NewData file.
    Leads3.to_excel(f'data\Leads{date_string}.xlsx', index = False)
    Leads3.to_excel(r'data\NewData.xlsx', index = False)
    print('File NewData.xlsx created on', date_string)