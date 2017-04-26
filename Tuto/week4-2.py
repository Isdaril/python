import urllib.request
from bs4 import *

def FindNewUrl(i, urlstring):
    #this function returns the i-th link found in the page urlstring
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, "html.parser")
    tags = soup.find_all('a')
    print(tags[i-1].string)
    return tags[i-1].get('href', None)

url = 'http://python-data.dr-chuck.net/known_by_Leone.html'
#url = 'http://python-data.dr-chuck.net/known_by_Fikret.html'
position = 18
repeat = 7

for i in range(repeat):
    url = FindNewUrl(position,url)
