import re

def SumNumbersInText(Str) :
    hand = open(Str)
    numbers = re.findall('[0-9]+', hand.read())
    result = sum(int(i) for i in numbers)
    return result

#def SumNumbersInTextWithOneLineOfCode(Str):
#    sum([int(i) for i in re.findall('[0-9]+', open(Str).read())])

sum = SumNumbersInText('regex_sum_349780.txt')
print(sum)
#sum1 = SumNumbersInTextWithOneLineOfCode('regex_sum_349780.txt')
#print(sum)
