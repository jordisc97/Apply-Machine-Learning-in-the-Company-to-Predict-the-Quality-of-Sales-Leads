import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
from xgboost import XGBClassifier, plot_importance
from treeinterpreter import treeinterpreter as ti
from collections import Counter 
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
date_string = time.strftime("%Y-%m-%d-%Hh")
from IPython.display import Image, display_html
from lime.lime_tabular import LimeTabularExplainer


def clean_dataset_ml(train_xgb = False, model_used = 'xgboost_pickle_06_07', number_samples = 200):
    # READ
    leads_df = pd.read_excel('Results_Verticals_Salesforce.xlsx')
    verticalization_df = pd.read_excel('Verticals Linkedin.xlsx')

    # Response Creation Date to datetime
    leads_df['Response Creation Date'] = leads_df['Response Creation Date'].astype('datetime64[ns]')

    # ResponseId to be unique
    leads_df = leads_df.sort_values(by='Response Creation Date', ascending=False)
    leads_df.drop_duplicates(subset ="ResponseId", keep="first", inplace = True) 

    leads_df = leads_df[leads_df.columns.drop(list(leads_df.filter(regex='Unnamed:')))] # drops all cols with unnamed

    # Grouping by different states
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Contacting', 'Work In Progress')
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Interest', 'Work In Progress')

    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Converted', 'Qualified')
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Qualification', 'Qualified')

    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Closed', 'Disqualified')
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('New', 'Unqualified')

    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Sales Nurture', 'Nurture')
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Sales Qualified', 'Nurture')
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('List', 'Nurture')
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Market Development', 'Nurture')


    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Qualified', 'Good')
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Nurture', 'Good')


    # Some initial analytics
    leads_df = leads_df.sort_values(by='Response Creation Date', ascending=True)
#     print("Initially we have:")
#     print("Leads with comments: ", round(len(leads_df['LkdIn_Specialties'].dropna())/len(leads_df)*100,2),'%')
#     print('Verticalized leads:', round(len(leads_df['LkdIn_Industry'].dropna())/len(leads_df)*100,2),'%')

    print('Disqualified Leads:', round(len(leads_df[leads_df['Lead Status']=='Disqualified'])/len(leads_df)*100,2),'%')
    print('Good Leads:', round(len(leads_df[leads_df['Lead Status']=='Good'])/len(leads_df)*100,2),'%')
    print('\nTotal Number of Leads:', round(len(leads_df),2))
    

    # Companies groupby
    leads_df['Company'] = leads_df['Company'].str.lower()
    company_count = leads_df.groupby(['Company']).size().to_frame('count').reset_index()
    company_count = company_count.rename(columns={"Company": "Company_grouped", "count": "Count_account"})
    leads_df = leads_df.merge(company_count, left_on='Company', right_on='Company_grouped', how='left')
    leads_df["Count_account"] = pd.to_numeric(leads_df["Count_account"])


    # Countries to Subregions
    leads_df['Sub Region'] = np.where(leads_df["Country"]=="Germany" ,"Germany", leads_df['Sub Region'])
    leads_df['Sub Region'] = np.where(leads_df["Sub Region"]=="DACH" ,"ACH", leads_df['Sub Region'])


    # Phones
    leads_df["Phone"] = leads_df["Phone"].str.replace(' ', '')
    leads_df["Phone"] = leads_df["Phone"].str.replace('-', '')
    leads_df["Phone"] = leads_df["Phone"].str.replace('–', '')
    leads_df["Phone"] = leads_df["Phone"].str.replace('/', '')
    leads_df["Phone"] = leads_df["Phone"].str.replace('.', '')
    leads_df["Phone"] = leads_df["Phone"].str.replace('(', '')
    leads_df["Phone"] = leads_df["Phone"].str.replace(')', '')
    leads_df["Phone"] = leads_df["Phone"].str.replace('+', '')
    leads_df["Phone"] = leads_df["Phone"].str.replace('^00', '')

    leads_df["Phone"].head(100)
    leads_df['Phone_desc'] = np.where(((leads_df["Phone"].str.startswith('4915')) | 
                                          (leads_df["Phone"].str.startswith('4916')) |
                                          (leads_df["Phone"].str.startswith('4917')) &
                                          (leads_df["Sub Region"]=="Germany")),"Phone",

                                           np.where((((leads_df["Phone"].str.startswith('447')) | 
                                          (leads_df["Phone"].str.startswith('3538'))) &
                                          (leads_df["Sub Region"]=="UK&I")),"Phone",

                                           np.where((((leads_df["Phone"].str.startswith('346')) | 
                                          (leads_df["Phone"].str.startswith('6')) |
                                          (leads_df["Phone"].str.startswith('3519'))) &
                                          (leads_df["Sub Region"]=="IBERIA")),"Phone",

                                           np.where((((leads_df["Phone"].str.startswith('337')) | 
                                          (leads_df["Phone"].str.startswith('336'))) &
                                          (leads_df["Sub Region"]=="France")),"Phone",

                                           np.where((((leads_df["Phone"].str.startswith('393')) | 
                                          (leads_df["Phone"].str.startswith('3567')) |
                                          (leads_df["Phone"].str.startswith('3569'))) &
                                          (leads_df["Sub Region"]=="Italy & Malta")),"Phone",

                                           np.where((((leads_df["Phone"].str.startswith('467')) | 
                                          (leads_df["Phone"].str.startswith('474')) |
                                          (leads_df["Phone"].str.startswith('475')) |
                                          (leads_df["Phone"].str.startswith('479')) |
                                          (leads_df["Phone"].str.startswith('453')) |
                                          (leads_df["Phone"].str.startswith('454')) |
                                          (leads_df["Phone"].str.startswith('455')) |
                                          (leads_df["Phone"].str.startswith('456')) |
                                          (leads_df["Phone"].str.startswith('457')) |
                                          (leads_df["Phone"].str.startswith('458'))) &
                                          (leads_df["Sub Region"]=="NORDICS")),"Phone",

                                           np.where((((leads_df["Phone"].str.startswith('436')) | 
                                          (leads_df["Phone"].str.startswith('417'))) &
                                          (leads_df["Sub Region"]=="ACH")),"Phone",

                                           np.where((((leads_df["Phone"].str.startswith('324')) | 
                                          (leads_df["Phone"].str.startswith('316'))) &
                                          (leads_df["Sub Region"]=="BENELUX")),"Phone",

                                           np.where((((leads_df["Phone"].str.startswith('485')) | 
                                          (leads_df["Phone"].str.startswith('484')) |
                                          (leads_df["Phone"].str.startswith('486')) |
                                          (leads_df["Phone"].str.startswith('487')) |
                                          (leads_df["Phone"].str.startswith('488')) |
                                          (leads_df["Phone"].str.startswith('9725')) |
                                          (leads_df["Phone"].str.startswith('362')) |
                                          (leads_df["Phone"].str.startswith('363')) |
                                          (leads_df["Phone"].str.startswith('365')) |
                                          (leads_df["Phone"].str.startswith('366'))) &
                                          (leads_df["Sub Region"]=="CEE+IL")),"Phone",

                                           np.where(leads_df['Phone'].isnull(),"No Phone",
                                           "Landline"))))))))))


    # 3D Company?
    leads_df["3D_Company?"] = leads_df["Company"].astype(str) + leads_df["LkdIn_Translation"].str.lower().astype(str)
    leads_df["3D_Company?"] = leads_df["3D_Company?"].str.contains('3d', na=False, regex=True)*1


    # Verticals
    ### VERTICALIZATION
    leads_df['HP_and_LnkdIn_Glsdr_Verticals'] = leads_df['Segment'].fillna(leads_df['LkdIn_Industry'])
    leads_df['HP_and_LnkdIn_Glsdr_Verticals'] = leads_df['HP_and_LnkdIn_Glsdr_Verticals'].fillna(leads_df['GlsDr_Industry'])

    verticalization_df = verticalization_df.rename(columns={"LinkedIn_Glsdr_HP_Verticals": "LnkdIn_Glsdr_HP_Standardized"})
    leads_df = leads_df.merge(verticalization_df, left_on='HP_and_LnkdIn_Glsdr_Verticals', 
                              right_on='LinkedIn_Glsdr_Industry', how='left')


    # Join Company Size
    leads_df['LnkdIn_and_Glsdr_Employees'] = leads_df['LkdIn_Company_size'].fillna(leads_df['GlsDr_Company size'])
    leads_df['LnkdIn_and_Glsdr_Employees'].unique()


    # Lead Number
    leads_df['HP Lead ID'] = leads_df['HP Lead ID'].str.replace('LEAD-', '')
    leads_df["HP Lead ID"] = pd.to_numeric(leads_df["HP Lead ID"])
    leads_df['ResponseId'] = leads_df['ResponseId'].str.replace('3DMCR-R', '')
    leads_df["ResponseId"] = pd.to_numeric(leads_df["ResponseId"])


    # Have phone or Email?
    leads_df["Phone_or_Email"] = leads_df["Phone"]
    leads_df['Phone_or_Email'] = leads_df['Phone_or_Email'].fillna(leads_df['Email'])
    leads_df['Phone_or_Email'] = np.where(leads_df['Phone_or_Email'].isnull(), 0, 1)


    # Have LinkedIn or Glassdoor?
    leads_df["Lnkdin_or_Glsdr"] = leads_df["LkdIn_Web_Name"]
    leads_df['Lnkdin_or_Glsdr'] = leads_df['Lnkdin_or_Glsdr'].fillna(leads_df['GlsDr_Company size'])
    leads_df['Lnkdin_or_Glsdr'] = np.where(leads_df['Lnkdin_or_Glsdr'].isnull(), 0, 1)


    # Income clearning
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('Desconocido/No aplicable por año', '0')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('Desconocido/No aplicable', '0')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('De 10 a 25\xa0millones\xa0(EUR) por año', '25000000')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('De 1000 a 2000\xa0millones\xa0(EUR) por año', '2000000000')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('De 500 a 1000\xa0millones\xa0(EUR) por año', '1000000000')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('De 5000 a 10\xa0000\xa0millones\xa0(EUR) por año', '10000000000')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('De 50 a 100\xa0millones\xa0(EUR) por año', '25000000')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('De 100 a 500\xa0millones\xa0(EUR) por año', '500000000')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('De 1 a 5\xa0millones\xa0(EUR) por año', '5000000')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('De 25 a 50\xa0millones\xa0(EUR) por año', '50000000')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('Menos de 1\xa0millón\xa0(EUR) por año', '1000000')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('De 2000 a 5000\xa0millones\xa0(EUR) por año', '5000000000')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('Más de 10\xa0000\xa0millones\xa0(EUR) por año', '100000000000')
    leads_df["GlsDr_Income"] = leads_df["GlsDr_Income"].replace('De 5 a 10\xa0millones\xa0(EUR) por año', '100000000')
    leads_df["GlsDr_Income"].fillna(0, inplace=True)
    leads_df["GlsDr_Income"] = pd.to_numeric(leads_df["GlsDr_Income"])


    # Company size
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].str.replace('�',' ')

    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('10,001+ employees', '15000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('5,001-10,000 employees', '10000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('1,001-5,000 employees', '5000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('501-1,000 employees', '1000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('201-500 employees', '500')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('51-200 employees', '200')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('11-50 employees', '50')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('2-10 employees', '10')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('0-1 employees', '1')

    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('Más de 10\xa0000\xa0empleados', '15000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('M s de 10 000 empleados', '15000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 5001 a 10\xa0000\xa0empleados', '10000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 5001 a 10 000 empleados', '10000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 1001 a 5000\xa0empleados', '5000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 1001 a 5000\xa0empleados', '5000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 1001 a 5000 empleados', '5000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 501 a 1000\xa0empleados', '1000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 501 a 1000 empleados', '1000')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 201 a 500 empleados', '500')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 201 a 500\xa0empleados', '500')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 51 a 200 empleados', '200')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 51 a 200 empleados', '200')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 51 a 200\xa0empleados', '200')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 1 a 50\xa0empleados', '50')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('De 1 a 50 empleados', '50')
    leads_df["LnkdIn_and_Glsdr_Employees"] = leads_df["LnkdIn_and_Glsdr_Employees"].replace('Desconocido', '0')

    leads_df["LnkdIn_and_Glsdr_Employees"].fillna(0, inplace=True)

    leads_df["LnkdIn_and_Glsdr_Employees"] = pd.to_numeric(leads_df["LnkdIn_and_Glsdr_Employees"])


    # Words and their appearences
    leads_df['LkdIn_Translation'] = leads_df['LkdIn_Translation'].str.lower()
    leads_df['LkdIn_Translation'] = leads_df['LkdIn_Translation'].str.replace('-', '')
    leads_df['LkdIn_Translation'] = leads_df['LkdIn_Translation'].str.replace('/', ',')
    leads_df['LkdIn_Translation'] = leads_df['LkdIn_Translation'].str.replace(' and ', ',')

    leads_df["LkdIn_Translation"] = leads_df["LkdIn_Translation"].str.split(",", n = 5, expand = True) 


    # Products occurences
    corpus = leads_df["LkdIn_Translation"].dropna().tolist()
    corpus = [item.strip() for item in corpus]

    def highest_occurrence_words(corpus):  
        Counters = Counter(corpus) 
        most_occur = Counters.most_common()
        return(most_occur)

    most_occur = highest_occurrence_words(corpus)

    # Replace the list of product of each lead by one of the 75 most used words if they are in the corpus of the lead.
    for word in most_occur[:75]:
        leads_df.loc[leads_df['LkdIn_Translation'].str.contains(word[0], na=False), 'LkdIn_Translation'] = word[0]


    # Generate Fiscal Years
    leads_df['Month'] = leads_df['Response Creation Date'].dt.month
    leads_df['Year'] = leads_df['Response Creation Date'].dt.year

    leads_df["Quarter"] = np.where(leads_df["Month"]==1,1,"")
    leads_df["Quarter"] = np.where(np.logical_and(leads_df["Month"]>=2, leads_df["Month"]<=4), 2,
                                  np.where(np.logical_and(leads_df["Month"]>=5, leads_df["Month"]<=7), 3,
                                    np.where(np.logical_and(leads_df["Month"]>=8, leads_df["Month"]<=10), 4, 1)
                                          ))
    leads_df["Fiscal_Year"] = np.where(np.logical_and(leads_df["Month"]>=1, leads_df["Month"]<=10), 
                                       leads_df['Year'], leads_df['Year']+1)

    leads_df['FY Correct'] = pd.to_numeric(leads_df["Fiscal_Year"].map(str)+leads_df["Quarter"].map(str))


    # Lead Status
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Disqualified', '0')
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Good', '1')
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Unqualified', '2')
    leads_df["Lead Status"] = leads_df["Lead Status"].replace('Work In Progress', '2')

    leads_df["Lead Status"] = pd.to_numeric(leads_df["Lead Status"])
    leads_df.fillna(0, inplace=True)


    # Split dataset
    leads_df['Response Creation Date'] = leads_df['Response Creation Date'].map(str)
    leads_df['Phone'] = leads_df['Phone'].map(str)

    leads_df = leads_df.sort_values(by='Response Creation Date', ascending=True) # Used for the hashing

    y = leads_df['Lead Status']
    X = leads_df.drop(['Opportunity Name','Opportunity Sales Stage', 'Opportunity Closed As Won/Lost', 'HP Lead ID',
                       'Opportunity ClosedWon', 'Opportunity Owner', 'Created Month', 'Last Modified', 'LinkedIn_Glsdr_Industry',
                       'Lead Status', 'Status Reason', 'Quantity', 'Lead Qualifier', 'Accept Lead', 'Activ_Disqualified Lead',
                       'Opportunity CloseDate', 'Opportunity Created Date', 'Nurture Reason','Willingness to Buy',
                       'Nurture Type', 'Lead Source', 'Rating', 'First Name', 'Middle Name', 'Last Name', 'Activ_closed response',
                       'Estimated Budget','Primary Campaign','Lead Accepted Date', 'Industry', 'Sub Industry', 'LkdIn_Industry',
                       'LkdIn_Company_size','GlsDr_Company size', 'GlsDr_Industry', 'Company_grouped', 'Activ_Re-assigned Lead',
                       'Job Role','Activ_Budget Change','Activ_Comments','Activ_Status Change','Job Function', 'Lead Close Reasons']
                      , axis=1)

    # OTHER FIELDS            'Activ_Attempted Calls','Activ_Budget Change','Activ_Comments','Activ_EDM','Activ_Email','Activ_Reached Calls','Activ_Status Change','Activ Touches'


    leads_obj = X.select_dtypes(include='object')
    leads_num = X.select_dtypes(exclude='object')


    # Hashing
    leads_obj['LkdIn_Web_Name'] = leads_obj['LkdIn_Web_Name'].map(str)  

    for i in leads_obj.columns:
        leads_obj[i] = leads_obj[i].map(str)
        le = preprocessing.LabelEncoder()
        le.fit(leads_obj[i])
        leads_obj[i] = le.transform(leads_obj[i])

    X_data = pd.concat([leads_obj,leads_num], axis=1, sort=False)


    # Select the labels to predict "New" leads
    selected_y_train = np.where(y < 2)[0]
    selected_y_predict = np.where(y == 2)[0]


    X_data_predict = X_data.iloc[selected_y_predict]
    y = y.iloc[selected_y_train]
    X_data_train = X_data.iloc[selected_y_train]
    print("X_data_train:", X_data_train.shape)
    print("X_data_predict:",X_data_predict.shape)



    # Split in train, test and validation
    X_train, X_test, y_train, y_test = train_test_split(X_data_train, y, test_size=0.3, random_state=42) # 70% training and 30% test
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42) # 66% test and 33% validation


    def feature_ranking(scaled_features_df,importances,std):
        # Print the feature ranking
        print("Feature ranking:")
        labels = []
        indices = np.argsort(importances)[::-1]

        for f in range(scaled_features_df.shape[1]):
            print('{num}. Feature {ind} ({imp}) - {name}'
                  .format(num=f + 1, ind = indices[f], imp = np.round_(importances[indices[f]],5), 
                          name = scaled_features_df.columns[indices[f]]))
            labels.append(scaled_features_df.columns[indices[f]])

        # Plot the feature importances of the forest
        plt.figure(figsize=(15,8))
        ax = plt.gca()
        plt.title("Feature importances")
        plt.bar(range(scaled_features_df.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, scaled_features_df.shape[1]])
        ax.set_xticklabels(labels, rotation=90)
        plt.show()

    def feature_importances_custom(model): 
        print("Using Model:" , model)
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                     axis=0)
        feature_ranking(X_train,importances,std)


    # Making the score
    scoring = {'accuracy': make_scorer(accuracy_score), 'AUC': 'roc_auc',}

    Xframes = [X_train, X_test]
    yframes = [y_train, y_test]

    X_result_train = pd.concat(Xframes)
    y_result_train = pd.concat(yframes)

    # FOR EXPLAINABILITY, transform data to martices
    X_result_matrix = X_result_train.values
    y_result_matrix = y_result_train.values
    X_val_matrix  = X_val.values
    y_val_matrix  = y_val.values
    
    
    # If retrain is true:
   
    if train_xgb:
        # Fit Train and Test and predict over Validation
        print("\nStarting the training of XGBoost with the validation set")
        print("\nSize of new Train set:", X_result_train.shape)
        print("Size of Validation set:", X_val.shape)
        
        # A parameter grid for XGBoost
        xgb_param = {  
                'objective':['binary:logistic'],
                'learning_rate': [0.05, 0.1, 0.2],
        #         'max_depth': [5, 10],
        #         'min_child_weight': [1, 5, 10],
        #         'subsample': [0.6, 0.8, 1.0],
                'n_estimators': [2500],
                'scale_pos_weight':[2, 5]
                }

        xgb_grid_val = GridSearchCV(XGBClassifier(), xgb_param, cv=5,
                            n_jobs = 10, scoring=scoring, refit='accuracy', verbose=True)
        xgb_grid_val.fit(X_result_matrix, y_result_matrix)


        "Start Predicting"
        pred_results = xgb_grid_val.predict(X_val_matrix)
        pred_proba_results = xgb_grid_val.predict_proba(X_val_matrix)[:, 1]

        acc_xgbCV = metrics.accuracy_score(y_val, pred_results)
        auc_xgbCV = metrics.roc_auc_score(y_val, pred_proba_results)

        print("Accuracy : %.5g" % metrics.accuracy_score(y_val, pred_results))
        print("AUC: %f" % metrics.roc_auc_score(y_val, pred_proba_results))
        print("F1 Score: %f" % metrics.f1_score(y_val, pred_results ))

        print("\nAccuracy if all bad leads:",round(1-y_result_train.sum()/len(y_result_train),5))
        print("Accuracy improved:",round((metrics.accuracy_score(y_val, pred_results )-(1-y_result_train.sum()/len(y_result_train))),5))
        print("\n")


        ### FEATURE IMPORTANCE
        feat_imp = xgb_grid_val.best_estimator_.feature_importances_
        feat = X_result_train.columns.tolist()
        res_df = pd.DataFrame({'Features': feat, 'Importance': feat_imp}).sort_values(by='Importance', ascending=False)
        res_df.plot('Features', 'Importance', kind='barh', title='Feature Importances',figsize=(15,12))
        plt.xlabel('Feature Importance Score')
        plt.gca().invert_yaxis()
        plt.show()


        from sklearn.metrics import confusion_matrix
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(y_val, pred_results)
        cm[:,[0, 1]] = cm[:,[1, 0]]
        cm[[0, 1],:] = cm[[1, 0],:]
        a = cm[0][1]
        cm[0][1] = cm[1][0]
        cm[1][0]= a
        sns.heatmap(cm.astype('float') / cm.sum(axis=0) [np.newaxis,:], annot=True, cmap=plt.cm.Blues, 
            xticklabels=["Good","Bad"], yticklabels=["Good","Bad"])
        plt.title("Normalized confusion matrix", fontsize=18)
        plt.ylabel("Predicted label")
        plt.xlabel("True label")
        plt.show()


        # Pickle the model
        acc_string = str(round(metrics.accuracy_score(y_val, pred_results),5))
        acc_string = acc_string.replace(".", "_")
        pickle.dump(xgb_grid_val, open(f"pickle/xgboost_pickle_{date_string}_{acc_string}", "wb"))
        # Use the latest model just created
        model_used = "xgboost_pickle_" + date_string + "_" + acc_string


    # load
    print("Using model:", model_used)
    xgb_model_loaded = pickle.load(open(f"pickle/{model_used}", "rb"))

    # FOR EXPLAINABILITY
    X_train_matrix = X_train.values
    y_train_matrix = y_train.values
    X_test_matrix  = X_test.values
    y_test_matrix  = y_test.values

    # Explain every entry

    def strip_html(htmldoc, strip_tags = ['html','meta','head','body'], outfile=None, verbose=False):
        """Strip out HTML boilerplate tags but perserve inner content

        Only will strip out the first occurrence of each tag, if multiple occurrences
        are desired, function must be modified.

        Args:
            htmldoc : str 
                HTML markup to process
            strip_tags : list[str]
                list of tags to be stripped out, including any attribute information
            outfile : str, optional (default: None)
                filename to output stripped html, if None parsed string is returned
            verbose : boolean (default: False)
                if True, prints removed tags and filepath
        """

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(htmldoc)

        for tag in strip_tags:
            rmtag = soup.find(tag)
            if rmtag is not None:
                rmtag.unwrap()
                if verbose: print(tag,'tags removed')

        stripped = soup.prettify()
        if outfile is not None:
            with open(outfile, 'w', encoding='utf-8') as f:
                f.write(stripped)
            if verbose: 
                print(f'file saved to: {outfile}')
        else:
            return stripped


    # Initialize LIME
    limeparams = dict( training_data = X_result_train.values, training_labels = y_result_train.values, feature_names = list(X_result_train.columns), class_names = ['Disqualified','Good'])
    lte = LimeTabularExplainer(**limeparams)


    # Loop to get the explainability
    from tqdm.notebook import tqdm
    print("\nCalculating the explainability:")
    explanations = []
    for i in tqdm(range(0,len(X_data_predict))):
        lte_expl = lte.explain_instance(X_data_predict.iloc[i], xgb_model_loaded.predict_proba, num_features=5, num_samples=number_samples)
        explanations.append(lte_expl.as_list())


    # Predict the probability of good lead for all dataset
    y_pred_train = xgb_model_loaded.predict_proba(X_data_train.values)
    y_pred_new = xgb_model_loaded.predict_proba(X_data_predict.values)


    # Create the dataset of scores and explainability for every lead
    result_new_score = pd.DataFrame(np.vstack((X_data_predict["ResponseId"], y_pred_new[:, 1])).T, 
                          columns=["ResponseId","Pred Prob"])  #1 is good

    result_new_score['Explainability'] = pd.Series(explanations, index=result_new_score.index)
    result_new_score['ResponseId'] = result_new_score['ResponseId'].astype(int)
    result_new_score[['Explain_1', 'Explain_2', 'Explain_3', 'Explain_4', 'Explain_5']] = pd.DataFrame(result_new_score['Explainability'].tolist(), index=result_new_score.index)

    result_train_score = pd.DataFrame(np.vstack((X_data_train["ResponseId"], y_pred_train[:, 1])).T, columns=["ResponseId","Pred Prob"])  #1 is good

    result_all_score = pd.concat([result_new_score,result_train_score], sort=False)
    result_all_score['ResponseId'] = '3DMCR-R' + result_all_score['ResponseId'].astype(str)


    # Save it all in excel
    date_string = time.strftime("%Y-%m-%d-%Hh")
    result_all_score.to_excel("Lead_Scores_Machine_Learning.xlsx")
    result_all_score.to_excel(f'predictions/Lead_Scores_Machine_Learning{date_string}.xlsx')
    print("Lead_Scores_Machine_Learning excel and its backups saved correctly at:", date_string)