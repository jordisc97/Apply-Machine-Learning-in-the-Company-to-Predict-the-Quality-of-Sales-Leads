%%time 
import os, random, sys, time, datetime
from Dataset_creation import dataset_creation
from Linkedin_Scraping import linkedin_scraping
from Glassdoor_Scraping import glassdoor_scraping
from Translate_and_create_file import create_file_ml
from Predict_Qualified_Salesforce import clean_dataset_ml
from datetime import timedelta
import schedule
import time

import warnings
warnings.filterwarnings("ignore")

def pipeline_ml():
    dataset_creation()
    linkedin_scraping()
    glassdoor_scraping()
    create_file_ml()
    clean_dataset_ml(train_xgb = False, model_used = 'xgboost_pickle_2020-08-06-10h_0_94749', number_samples = 200)
    
schedule.every().day.at("00:01").do(pipeline_ml)

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute
    
