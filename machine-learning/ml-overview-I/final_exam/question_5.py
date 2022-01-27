# Python script to scrape an article given the url of the article and store the extracted text in a file
# Url: https://www.yelp.ca/biz/pai-northern-thai-kitchen-toronto-5?osq=Restaurants

import os
import requests
import re
import sys
import urllib.request as url
# Code here - Import BeautifulSoup library
from bs4 import BeautifulSoup
# Code ends here

# function to get the html source text of the website
global url_string

# Code here - Ask the user to input "Enter url of a medium article: " and collect it in url
url_string = input("Enter url of a the restaurant: ")
# Code ends here

# Code here - Call get method in requests object, pass url and collect it in res
source = url.urlopen(url_string)
# Code ends here

soup = BeautifulSoup(source, 'html.parser')

# Restaurant name 
mains = soup.find_all("div", {"class": " businessName__09f24__3Ml2X display--inline-block__09f24__3SvIn border-color--default__09f24__3Epto"})
bus_name = mains[0].find("a").text
print("Restaurant name:" + bus_name)

# How many reviews are there in total
ratings = mains[0].find("div", {"class": " arrange-unit__373c0__2u2cR arrange-unit-fill__373c0__3cIO5 border-color--default__373c0__2s5dW nowrap__373c0__AzEKB"}).div.get('aria-label')
print("Restaurant ratings:" + ratings)


# function to remove all the html tags and replace some with specific strings
def clean(text):
    rep = {"<br>": "\n", "<br/>": "\n", "<li>":  "\n"}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    text = re.sub('\<(.*?)\>', '', text)
    return text


def collect_text(soup):
	text = f'url: {url}\n\n'
	para_text = soup.find_all('p')
	print(f"paragraphs text = \n {para_text}")
	for para in para_text:
		text += f"{para.text}\n\n"
	return text

# function to savegit file in the current directory
def save_file(text):
	if not os.path.exists('./scraped_articles'):
		os.mkdir('./final_exam/scraped_articles')
	#name = url.split("/")[-1]
	name = "yelp_scraped"
	fname = f'{name}.txt'
	
	# Code here - write a file using with (2 lines)
	with open(fname, 'w') as final_file:
		final_file.write(text) 
	# Code ends here

	print(f'File saved in directory {fname}')


if __name__ == '__main__':
	text = collect_text(get_page())
	save_file(text)
	# Instructions to Run this python code
	# Give url as https://www.yelp.ca/biz/pai-northern-thai-kitchen-toronto-5?osq=Restaurants