import urllib.request as rq


urlname = 'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1920&bih=969&q=fish&oq=fish&gs_l=img.3..0l10.3633.4042.0.4265.5.5.0.0.0.0.97.309.4.4.0....0...1ac.1.64.img..1.4.307.0.SckLjh-Q35c#imgrc=bCfIXr2kvXulpM:'
filename = 'D:\Documents\Cours\Python\Fishy\Fishs\image1.jpg'
f = open(filename,'wb')
f.write(rq.urlopen(urlname).read())
f.close()
