from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import os
import concurrent.futures
from selenium.webdriver.chrome.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import imdb

os.makedirs('movie_reviews',exist_ok=True)

def scrape_data(revs):
    
    # for i in tqdm(range(1,10)):
    
    try:
        if_spoiler = revs.find_element(By.CLASS_NAME,"spoiler-warning")
        spolier_btn = revs.find_element(By.CLASS_NAME,"ipl-expander").click()
        contents = revs.find_element(By.XPATH,"//div[contains(@class, 'text show-more__control')]").text
    except NoSuchElementException:
        contents = revs.find_element(By.CLASS_NAME,"content").text
        if contents == "":
            contents = revs.find_element(By.CLASS_NAME,"text show-more__control clickable").text

    
    try:
        title = revs.find_element(By.CLASS_NAME,"title").text.strip()
    except NoSuchElementException:
        title=  ""

    try:
        rating = revs.find_element(By.CLASS_NAME,"rating-other-user-rating").text.split("/")[0]
    except NoSuchElementException:
        rating= ""
    re.sub('\n',' ',contents)
    re.sub('\t',' ',contents)
    contents.replace("//","")
    date = revs.find_element(By.CLASS_NAME,"review-date").text
    return date,contents,rating,title
    
    

# if __name__ == '__main__':
    # movie_link = 'https://www.imdb.com/title/tt2906216/reviews/?ref_=tt_ql_2'
    # movie_link = 'https://www.imdb.com/title/tt10366206/reviews/?ref_=tt_ql_2'
def main_scraper(movie_name:str,save_name:str):
    ia = imdb.Cinemagoer()
    movies = ia.search_movie(movie_name)
    movie_id = movies[0].movieID
    movie_link = f'https://www.imdb.com/title/tt{movie_id}/reviews/?ref_=tt_ql_2'
    driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()))
    driver.get(movie_link)
    driver.maximize_window()

    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight-250);')
        try:
            load_button = driver.find_element(By.CLASS_NAME,'ipl-load-more__button')
            load_button.click()
            time.sleep(1) 
        except:
            break

    driver.execute_script('window.scrollTo(0, 100);')

    rev_containers = driver.find_elements(By.CLASS_NAME,"review-container")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(scrape_data,rev_containers)
    reviews_date = []
    reviews_comment = []
    reviews_rating = []
    reviews_title = []
    for result in results:
        date,contents,rating,title = result
        reviews_date.append(date)

        reviews_comment.append(contents)
        reviews_rating.append(rating)
        reviews_title.append(title)

        # driver.quit()
    df = pd.DataFrame(columns=['Date','Title','Review','Rating'])

    df['Date'] = reviews_date
    df['Title'] = reviews_title
    df['Review'] = reviews_comment
    df['Rating'] = reviews_rating

    
    # print(df)
    df.to_csv(f'movie_reviews/{save_name}.csv',index=False)
