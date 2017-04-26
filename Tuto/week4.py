# Note - this code must run in Python 2.x and you must download
# http://www.pythonlearn.com/code/BeautifulSoup.py
# Into the same folder as this program

import urllib.request
from bs4 import *

url = 'http://python-data.dr-chuck.net/comments_349785.html'
#url = 'http://python-data.dr-chuck.net/comments_42.html'

html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, "html.parser")

tags = soup.find_all('span')
sum = 0
for tag in tags:
    sum += int(tag.string)

print(sum)
