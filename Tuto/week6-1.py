import urllib.request
import json

serviceurl = 'http://python-data.dr-chuck.net/comments_349786.json'
#serviceurl = 'http://python-data.dr-chuck.net/comments_42.json'


print('Retrieving', serviceurl)
data = urllib.request.urlopen(serviceurl).read()
print('Retrieved',len(data),'characters')

info = json.loads(data)
sum = 0
for count in info['comments']:
    sum += count['count']

print('sum: ',sum)
