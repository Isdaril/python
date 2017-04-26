import urllib.request as urlrequest
import urllib.parse as urlparse
import json

# serviceurl = 'http://maps.googleapis.com/maps/api/geocode/json?'
serviceurl = 'http://python-data.dr-chuck.net/geojson?'


address = 'Universidad Tecnologica Boliviana'
#address = 'South Federal University'
f = open('place_id.txt', 'w')

url = serviceurl + urlparse.urlencode({'sensor':'false', 'address': address})
print('Retrieving', url)
uh = urlrequest.urlopen(url)
data = uh.read()
print('Retrieved',len(data),'characters')

js = json.loads(data)
place_id = js['results'][0]['place_id']
print(place_id)
f.write(place_id)
