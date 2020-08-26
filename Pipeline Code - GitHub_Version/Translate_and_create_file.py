import os, random, sys, time
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.common.exceptions import NoSuchElementException
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import numpy as np
import copy
from langdetect import detect
from googletrans import Translator
from Detect_New_Entries import merge_new_entires, detect_new_entires
import time


# DETECT LANGUAGE
def detect_language(lkdin_df, column = "Specialties"):
    languages = []
    for speciality in lkdin_df[column]:
        if pd.isnull(speciality):
            languages.append("NoText")
        else:
            string = str(speciality)
            languages.append(detect(string))
    return languages


# TRANSLATE
def translate_list(list_specialities, columns="Specialties"):
    translatedList = []
    for speciality in list_specialities[columns]:
        temporal_place = []
        ################################# ORIGINAL STRING
        temporal_place.append(speciality)

        ################################# REINITIALIZE THE API
        translator = Translator()
        time.sleep(round(random.uniform(1, 2),2))

        try:
            ################################# TRANSLATE
            temporal_place.append(translator.translate(speciality, dest='en').text)
        except Exception as e:
            print(str(e))
            temporal_place.append("")
            continue

        translatedList.append(temporal_place)
        print(len(translatedList),"-",round(len(translatedList)/len(list_specialities)*100,2),"% completed.")
    return translatedList


def create_file_ml():
    date_string = time.strftime("%Y-%m-%d-%Hh")
    
    # GENERATE INPUT FILE FOR MACHINE LEARNING
    ################################# READ
    print("LinkedIn Scraping is being translated")
    lkdin_df = pd.read_excel('Linkedin_Scraping_Translated.xlsx')  # File from LinkedIn  Linkedin_Scraping_Translated
    df_to_transl = lkdin_df[lkdin_df["Translated?"]!=1]
    print("New Leads:",len(df_to_transl))


    ################################ DETECT LANGUAGE
    df_to_transl["Specialties"] = df_to_transl["Specialties"].replace('.', np.nan)
    languages = detect_language(df_to_transl, "Specialties")
    df_to_transl['Languages'] = np.array(languages)

    ################################# TRANSLATE COMENTS
    list_specialities = df_to_transl[df_to_transl["Languages"]!= "en"] # NOT ENGLISH
    list_specialities = list_specialities[list_specialities['Languages']!="NoText"] # NOT NAN
    print("Leads that need to be translated:", len(list_specialities))
    translatedList = translate_list(list_specialities)

    ################################# JOIN THE TRANSLATED PART TO THE ORIGINAL DATAFRAME
    df_to_transl = df_to_transl.drop(["Translation"], axis=1)
    df_translation = pd.DataFrame(translatedList, columns=["Original","Translation"])
    df_to_transl2 = df_to_transl.merge(df_translation, left_on='Specialties', right_on='Original', how='left')
    df_to_transl2['Translation'] = df_to_transl2['Translation'].fillna(df_to_transl2['Specialties'])
    df_to_transl3 = df_to_transl2.drop(columns="Original")
    df_to_transl3["Translated?"] = 1

    ################################# MERGE DATAFRAMES AGAIN
    already_transl = lkdin_df[lkdin_df["Translated?"]==1]
    lkdin_df3 = already_transl.append(df_to_transl3)

    lkdin_df3 = lkdin_df3[lkdin_df3.columns.drop(list(lkdin_df3.filter(regex='Unnamed:')))] # drops all cols with unnamed

    lkdin_df3.to_excel("Linkedin_Scraping_Translated.xlsx")
    lkdin_df3.to_excel(f'backups_scrape/Linkedin_Scraping_Translated{date_string}.xlsx')


    ################################# TRANSLATE GLASSDOOR
    print("\nGlassdoor Scraping is being translated")
    revenue_df = pd.read_excel('Glassdoor_Scraping_Revenue.xlsx')       # File from scrape_glassdoor
    df_to_transl_glsdr = revenue_df[revenue_df["Translated?"]!=1]
    print("New Leads:",len(df_to_transl_glsdr))

    ################################# TRANSLATE SECTORS
    unique_sectors = pd.DataFrame(df_to_transl_glsdr['Sector'].unique(), columns=["Sectors"])
    print("Leads that need to be translated:", len(unique_sectors))
    translatedList_glsdr = translate_list(unique_sectors, "Sectors")
    ################################# JOIN THE TRANSLATED PART TO THE ORIGINAL DATAFRAME
    df_to_transl_glsdr = df_to_transl_glsdr.drop(["Translation_sector"], axis=1)
    df_translation_glsdr = pd.DataFrame(translatedList_glsdr, columns=["Original","Translation"])
    df_to_transl2_glsdr = df_to_transl_glsdr.merge(df_translation_glsdr, left_on='Sector', right_on='Original', how='left')
    df_to_transl3_glsdr = df_to_transl2_glsdr.drop(columns="Original")
    df_to_transl3_glsdr["Translated?"] = 1

    ################################# MERGE DATAFRAMES AGAIN
    already_transl = revenue_df[revenue_df["Translated?"]==1]
    glsdr_df3 = already_transl.append(df_to_transl3_glsdr)

    glsdr_df3 = glsdr_df3[glsdr_df3.columns.drop(list(glsdr_df3.filter(regex='Unnamed:')))] # drops all cols with unnamed
    glsdr_df3 = glsdr_df3.drop(["Translation"], axis=1)

    # glsdr_df3 = glsdr_df3.rename(columns={"Translation": "Translation_sector"})

    glsdr_df3.to_excel("Glassdoor_Scraping_Revenue.xlsx")
    glsdr_df3.to_excel(f'backups_scrape/Glassdoor_Scraping_Revenue{date_string}.xlsx')



    ################################# MERGE
    print("\nMerging the files...")
    revenue_df = glsdr_df3      # File from scrape_glassdoor
    original_leads = pd.read_excel('data/NewData.xlsx')
    original_leads['Company'] = original_leads['Company'].str.lower()

    revenue_df = revenue_df[['Name','Tamaño','Ingresos', 'Translation_sector']]
    revenue_df = revenue_df.rename(columns={"Name": "Company",
                                        "Tamaño": "GlsDr_Company size",
                                        "Ingresos": "GlsDr_Income",
                                        "Translation_sector": "GlsDr_Industry"})
    revenue_df['Company'] = revenue_df['Company'].str.lower()

    # MAKE SURE THE DATA HAS THE SAME FIELDS
    lkdin_df3 = lkdin_df3[['Name','Web_Name','Industry','Company size','Headquarters','Specialties', 'Translation']]
    lkdin_df3 = lkdin_df3.rename(columns={"Name": "Company",
                                        "Industry": "LkdIn_Industry",
                                        "Web_Name": "LkdIn_Web_Name",
                                        "Company size": "LkdIn_Company_size",
                                        "Headquarters": "LkdIn_Headquarters",
                                        "Specialties": "LkdIn_Specialties",
                                         "Translation" : "LkdIn_Translation"})
    lkdin_df3['Company'] = lkdin_df3['Company'].str.lower()

    # MERGE
    result = lkdin_df3.merge(original_leads, left_on='Company', right_on='Company', how='right')
    result = result.merge(revenue_df, left_on='Company', right_on='Company', how='left')

    ### SAVE THE DATA TO Results_Verticals_Salesforce
    result = result[result.columns.drop(list(result.filter(regex='Unnamed:')))] # drops all cols with unnamed
    print("Saving the results on:", date_string)
    result.to_excel("Results_Verticals_Salesforce.xlsx")
    result.to_excel(f'backups_scrape/Results_Verticals_Salesforce{date_string}.xlsx')




