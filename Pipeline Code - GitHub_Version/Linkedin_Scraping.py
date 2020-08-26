path = "data"
filename = "NewData"

import os, random, sys, time
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.common.exceptions import NoSuchElementException
from Detect_New_Entries import detect_new_entires, merge_new_entires
import time
from webdriver_manager.chrome import ChromeDriverManager


def linkedin_scraping(batch = 150):
    # Scrapes the new data from linkedin 
    # and saves it to the Excel Scraping Translated.xlsx

    ### READ DATA
    ##############################################################
    leads_df = pd.read_excel('Linkedin_Scraping_Translated.xlsx') 
    original_leads = pd.read_excel(f'{path}/{filename}.xlsx')

    ### LEADS MISSING - NOT CASE SENSITIVE
    ##############################################################
    original_leads = original_leads.rename({'Company / Account':'Company'}, axis=1)

    scraped_comp = pd.DataFrame(leads_df['Name'].str.lower().unique(),columns=["Comp"]).dropna()
    orig_leads = pd.DataFrame(original_leads['Company'].str.lower().unique(),columns=["Comp"]).dropna()

    to_scrape = orig_leads.merge(scraped_comp,indicator = True, how='left').loc[lambda x : x['_merge']!='both']
    new_leads = to_scrape['Comp']
    print("New Leads in Linkedin Scraping Translated:", len(new_leads))

    def login_account():
        ### LOGIN
        ##############################################################
        browser = webdriver.Chrome('driver/chromedriver83.exe')
        browser.get('https://www.linkedin.com/login')

        import random
        session = round(random.uniform(1, 6))

        file = open(f'config/config_linkedin_{session}.txt')
        lines = file.readlines()
        username = lines[0]
        password = lines[1]

        elementID = browser.find_element_by_id('username')
        elementID.send_keys(username)

        elementID = browser.find_element_by_id('password')
        elementID.send_keys(password)
        elementID.submit()
        return browser
    
    def save_the_data():
        ## Transform data from scrapping to DataFrame
        ##############################################################
        grid_header = ['Name', 'Web_Name', 'Website','Phone','Industry','Company size','Headquarters','Type','Founded','Specialties']

        dataframe = []
        for k in range(0,batch):
            j = 0
            i = 0
            lista = []

            while i< len(grid_header):
                try:
                    if grid_header[i] == companies_list[0+k*2][j]:
                        field = companies_list[1+k*2][j]
                    else:
                        field = None
                        j -= 1
                    lista.append(field)
                    j +=1

                except:
                    lista.append(None)
                i += 1    
            dataframe.append(lista)

        df = pd.DataFrame.from_records(dataframe,columns=grid_header)
        df["Phone"] = df["Phone"].str.split('\n').str[0]
        df["Name"] = df["Name"].replace(" an "," & ", regex=True)


        ### Save the data
        ##############################################################

        if len(companies)!= 0:  
            if round(len(df['Industry'].dropna())/len(df),3)<0.2:
                linkedin_scraping()
            else:
                prev_file = pd.read_excel("Linkedin_Scraping_Translated.xlsx")
                date_string = time.strftime("%Y-%m-%d-%Hh")

                out_grid_header = ['Name', 'Web_Name', 'Website','Phone','Industry','Company size',
                                   'Headquarters','Type','Founded','Specialties','Translated?','Translation']

                mix_file = prev_file.append(df)
                mix_file = mix_file.drop(columns=['Unnamed: 0'])
                mix_file = mix_file[out_grid_header]
                mix_file.to_excel("Linkedin_Scraping_Translated.xlsx")
                mix_file.to_excel(f'backups_scrape/Linkedin_Scraping_Translated{date_string}.xlsx')

                print("Linkedin Scraping Translated and its backups saved correctly at:",date_string)
                browser.close()
        else:
            print('No new Responses')

            
    # PREPARE THE LIST
    ##############################################################
    company_list = new_leads
    company_list = company_list.dropna() 
    company_list = company_list.drop_duplicates()
    company_list = company_list.tolist()
    company_list = [w.replace(' & ',' an ') for w in company_list]
    
    companies = company_list


    ### LOOP
    ##############################################################

    companies_list = []
    companies_count = 0

    ### IF THERE ARE NO LEADS TO SCRAPE, THE WEBDIRVER WON'T OPEN
    if len(companies)!= 0:
        browser = login_account()
        
    ### SCRAPE EACH COMPANY
    for company in companies:
        browser.get('https://www.linkedin.com/')
        comp_search = "https://www.linkedin.com/search/results/companies/?keywords="
        company = company.replace("about/", "")
        fullLink_search = comp_search + company
        browser.get(fullLink_search)
        time.sleep(round(random.uniform(6, 9),2))

        error = False

        headers_list = []
        comp_data_list = []

        comp_data_list.append(company)
        headers_list.append('Name')

        # If it does not find the company catch.
        try:
    #         browser.find_element_by_xpath("//h3[@class='app-aware-link ember-view']").click()
            venue = browser.find_element_by_xpath('//a[@class="app-aware-link ember-view"]')
            venue.click()
            time.sleep(round(random.uniform(1, 2),2))

        except NoSuchElementException:  #spelling error making this code not work as expected
            companies_list.append(headers_list)
            companies_list.append(comp_data_list)
            error = True

        time.sleep(round(random.uniform(5, 9),2))

        if error == False:
            current_page = browser.current_url

            if "about/" in current_page:
                pass
            else:
                fullLink = current_page + "about/"
                browser.get(fullLink)
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(round(random.uniform(5, 7),2))

            src = browser.page_source
            soup = BeautifulSoup(src, 'lxml')

            try:
                comp_data_list.append(browser.find_element_by_xpath("//h1[@class='org-top-card-summary__title t-24 t-black truncate']").text)
                headers_list.append('Web_Name')
            except NoSuchElementException:  
                pass 


            # Chech if there is anything in the about page
            try:
                company_div = soup.find('div', {'class': 'org-grid__core-rail--no-margin-left'})
                comp = company_div.find_all('p')
                comp_desc = comp[0]
            except IndexError:  #spelling error making this code not work as expected
                companies_list.append(headers_list)
                companies_list.append(comp_data_list)
                error = True    

            if error == False:    
                comp_desc = comp[0].get_text().strip() 
                header = company_div.find_all('dt')
                comp_data = company_div.find_all('dd')

                j = 0         
                for i in range(len(header)):
                    headers_list.append(header[i].get_text().strip())
                    comp_data_list.append(comp_data[j].get_text().strip())
                    if header[i].get_text().strip()=="Company size" and len(header) < len(comp_data):
                        j += 1  
                    j += 1

                companies_list.append(headers_list)
                companies_list.append(comp_data_list)

        time.sleep(round(random.uniform(2, 5),2))    
        companies_count = companies_count+1
        # Progress bar
        print('{}: Scraping: {}% completed.'.format(companies_count,format(companies_count/len(companies)*100, '.2f')), end="\r")
        
        if companies_count % batch == 0:
            # Every change of user, the data is stored
            save_the_data()
            companies_list = []
            browser = login_account()

        

    browser.close()






