import urllib.request
import xml.etree.ElementTree as ET

serviceurl = 'http://python-data.dr-chuck.net/comments_349782.xml'
#serviceurl = 'http://python-data.dr-chuck.net/comments_42.xml'


print('Retrieving', serviceurl)
data = urllib.request.urlopen(serviceurl).read()
print('Retrieved',len(data),'characters')

tree = ET.fromstring(data)
results = tree.findall('.//count')
sum = 0
for count in results:
    sum += int(count.text)

print('sum: ',sum)
