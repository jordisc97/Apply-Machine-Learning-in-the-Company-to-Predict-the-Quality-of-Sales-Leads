import os, random, sys, time
# from urllib.parse import urlparse
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.common.exceptions import NoSuchElementException
# from Detect_New_Entries import detect_new_entires, merge_new_entires
import time

# Scrapes the new data from glassdoor
# and saves it to the Scrapping revenue.xlsx
from selenium import webdriver
#     browser = webdriver.Chrome(executable_path=r'C:\path\to\chromedriver.exe')
#     browser.get('http://google.com/')


def glassdoor_scraping(batch=200,path = "data", filename = "NewData"):

    ### LEADS MISSING
    ##############################################################
    leads_df = pd.read_excel('Glassdoor_Scraping_Revenue.xlsx') #Glassdoor_Scraping_Revenue.xlsx
    original_leads = pd.read_excel(f'{path}/{filename}.xlsx')

    ### LEADS MISSING - NOT CASE SENSITIVE
    ##############################################################
    original_leads = original_leads.rename({'Company / Account':'Company'}, axis=1)
    scraped_comp = pd.DataFrame(leads_df['Name'].str.lower().unique(),columns=["Comp"]).dropna()
    orig_leads = pd.DataFrame(original_leads['Company'].str.lower().unique(),columns=["Comp"]).dropna()

    to_scrape = orig_leads.merge(scraped_comp,indicator = True, how='left').loc[lambda x : x['_merge']!='both']
    new_leads = to_scrape['Comp']
    #     new_leads = list(set(scraped_comp) - set(orig_leads))
    print("New Leads in Glassdoor Scraping Revenue:", len(new_leads))


    ### LOGIN
    ##############################################################
    browser = webdriver.Chrome('driver/chromedriver83.exe')
    browser.get('https://www.glassdoor.es/profile/login_input.htm?userOriginHook=HEADER_SIGNIN_LINK')

    file = open('CONFIG/config_glassdoor.txt')
    lines = file.readlines()
    username = lines[0]
    password = lines[1]

    time.sleep(round(random.uniform(1, 2),2))

    ##############################################################
    elementID = browser.find_element_by_id('userEmail')
    elementID.send_keys(username)
    time.sleep(round(random.uniform(1, 1.3),2))

    elementID = browser.find_element_by_id('userPassword')
    elementID.send_keys(password)
    time.sleep(round(random.uniform(1, 2),2))

    elementID.submit()

    # READ
    ##############################################################
    company_list = new_leads
    company_list = company_list.dropna() 
    company_list = company_list.drop_duplicates()
    company_list = company_list.tolist()
    companies = company_list
#     print("New companies:", len(companies))

    def save_data():
        grid_header = ['Name','Sitio web', 'Sede', 'Tamaño', 'Fundada en', 'Tipo', 'Sector', 'Ingresos']
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
        df = df[df["Sitio web"] != 'global.mandg.com']


        ### Save the data every change of user
        ##############################################################
        prev_file = pd.read_excel("Glassdoor_Scraping_Revenue.xlsx")
        date_string = time.strftime("%Y-%m-%d-%Hh")

        out_grid_header = ['Name','Sitio web', 'Sede', 'Tamaño', 'Fundada en', 'Tipo', 'Sector', 
                           'Ingresos','Translated?','Translation_sector']

        mix_file = prev_file.append(df)
        mix_file = mix_file.drop(columns=['Unnamed: 0'])
        mix_file = mix_file[out_grid_header]

        mix_file.to_excel("Glassdoor_Scraping_Revenue.xlsx")
        mix_file.to_excel(f'backups_scrape/Glassdoor_Scraping_Revenue{date_string}.xlsx')
        print("Glassdoor Scraping Revenue and its backups saved correctly on", date_string)
        

        
    ### LOOP
    ##############################################################
    companies_list = []
    companies_count = 0

    browser.get('https://www.glassdoor.es/Opiniones/barcelona-hp-opiniones-SRCH_IL.0,9_IM1015_KE10,12.htm')

    for company in companies:
        error = False
        inside_page = False
        headers_list = []
        comp_data_list = []

        headers_list.append('Name')
        comp_data_list.append(company)
        browser.get('https://www.glassdoor.es/Opiniones/barcelona-hp-opiniones-SRCH_IL.0,9_IM1015_KE10,12.htm')

        time.sleep(round(random.uniform(3, 4),2))

        try: 
            elementID = browser.find_element_by_id('sc.keyword')
        except: ## ERROR WHEN IT GETS OUT OF THE OPINIONS SEARCH
            browser.get('https://www.glassdoor.es/Opiniones/barcelona-hp-opiniones-SRCH_IL.0,9_IM1015_KE10,12.htm')
            error = True
        elementID = browser.find_element_by_id('sc.keyword')
        elementID.send_keys(Keys.CONTROL, 'a')
        elementID.send_keys(Keys.BACKSPACE)
        elementID.send_keys(company)
        elementID.submit()


        time.sleep(round(random.uniform(5, 7),2))

        ### SOMETIMES IT ENTERS THE PAGE DIRECTLY
        try:
            browser.find_element_by_xpath("//span[@class='sqLogo tighten lgSqLogo logoOverlay']//img").click()
            inside_page = True
        except:
            pass

        # IF COMPANY NOT FOUND, CATCH.
        if inside_page == False:
            try:
                browser.find_element_by_xpath("//body[contains(@class,'_initOk noTouch tablet mobileFF')]/div[contains(@class,'pageContentWrapper')]/div[@id='PageContent']/div[@id='PageBodyContents']/div[contains(@class,'pageInsideContent cf')]/div[@id='EI-Srch']/div[@id='SearchResults']/div[@id='ReviewSearchResults']/article[@id='MainCol']/div[contains(@class,'module')]/div[2]/div[1]/div[2]/div[1]").click()  
            except NoSuchElementException:  #spelling error making this code not work as expected
                error = True

        time.sleep(round(random.uniform(5, 6),2))    

        if error==False:    
            src = browser.page_source
            soup = BeautifulSoup(src, 'lxml')

            try:
                company_div = soup.find('div', {'class': 'info flexbox row col-hh'})
                header = company_div.find_all('label')
                info = company_div.find_all('span')
            except:
                pass

            for i in range(0,len(header)):
                headers_list.append(header[i].get_text().strip())
                comp_data_list.append(info[i].get_text().strip())

        companies_list.append(headers_list)
        companies_list.append(comp_data_list)

        companies_count = companies_count+1
        print('{}: Scraping: {}% completed.'.format(companies_count,format(companies_count/len(companies)*100, '.2f')), end="\r")
        if companies_count % batch == 0:
            # Save the data every change of user
            save_data()
            companies_list = []
            time.sleep(round(random.uniform(65, 90),2))
#         print('{}: Scraping: {}% completed - {}'.format(companies_count,round(companies_count/len(companies)*100,2),company), end="\r")
        

    
    browser.close()
    