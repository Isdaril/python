import math

class Calculator:
    
    def __init__(self):
       self.alreadyFound = dict() 
    def findPrimeCount(self,n):
        result = 1
        if n in self.alreadyFound:
            return self.alreadyFound[n]
        if n == 1:
            self.alreadyFound[1] = 0
            return 0;
        lim = math.floor(math.sqrt(n))
        #print("sqrt of",n,"is:",lim)
        for i in range(2,lim+2):
            if n % i == 0:
                #print(i,"is a divisor")
                result = self.findPrimeCount(n/i) + 1
                break
        self.alreadyFound[n] = result
        return result

T = int(input().strip())
calc = Calculator()
for i in range(T):
    N = int(input().strip())
    res = 0
    array = [int(x) for x in input().split()]
    for k in array:
        a = calc.findPrimeCount(k)
        print(k, a)
        res ^= a
    if res == 0:
        print("2")
    else:
        print("1")