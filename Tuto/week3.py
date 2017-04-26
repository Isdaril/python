import socket

mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysock.connect(('data.pr4e.org', 80))
message = 'GET http://data.pr4e.org/intro-short.txt HTTP/1.0\n\n'
mysock.send(str.encode(message))
f = open('httpexfile.txt', 'w')

while True:
    data = mysock.recv(512)
    if ( len(data) < 1 ) :
        break
    f.write(data.decode("UTF-8"));

mysock.close()
