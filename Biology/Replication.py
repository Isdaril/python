def remove_duplicates(Items):
    ItemsNoDuplicates = []
    for i in range(len(Items)):
        flag = 0
        for j in range(len(ItemsNoDuplicates)):
            if ItemsNoDuplicates[j] == Items[i]:
                flag = 1
                break
        if flag == 0:
            ItemsNoDuplicates.append(Items[i])
    return ItemsNoDuplicates

def FrequentWords(Text, k):
    FrequentPatterns = []
    Count = CountDict(Text, k)
    m = max(Count.values())
    for i in Count:
        if Count[i] == m:
            FrequentPatterns.append(Text[i:i+k])
    FrequentPatternsNoDuplicates = remove_duplicates(FrequentPatterns)
    return FrequentPatternsNoDuplicates

def CountDict(Text, k):
    Count = {}
    for i in range(len(Text)-k+1):
        Pattern = Text[i:i+k]
        Count[i] = PatternCount(Pattern, Text)
    return Count

def reverse(Text):
    rev = ''
    for i in range(len(Text)):
        rev += Text[len(Text)-i-1]
    return rev

def complement(char):
    comp = ''
    if char == 'C':
        comp = 'G'
    elif char == 'G':
        comp = 'C'
    elif char == 'T':
        comp = 'A'
    elif char == 'A':
        comp = 'T'
    return comp

def ReverseComplement(Text):
    rev = ''
    for i in range(len(Text)):
        rev += complement(Text[i])
    return(reverse(rev))

def SymbolArray(Genome, symbol):
    array = {}
    n = len(Genome)
    ExtendedGenome = Genome + Genome[0:n//2]
    for i in range(n):
        array[i] = PatternCount(symbol, ExtendedGenome[i:i+(n//2)])
    return array

def FasterSymbolArray(Genome, symbol):
    array = {}
    n = len(Genome)
    ExtendedGenome = Genome + Genome[0:n//2]
    array[0] = PatternCount(symbol, Genome[0:n//2])
    for i in range(1, n):
        array[i] = array[i-1]
        if ExtendedGenome[i-1] == symbol:
            array[i] = array[i]-1
        if ExtendedGenome[i+(n//2)-1] == symbol:
            array[i] = array[i]+1
    return array

def Skew(Genome):
    skew = {} #initializing the dictionary
    n = len(Genome)
    skew[0] = 0
    for i in range(1, n+1):
        skew[i] = skew[i-1]
        if Genome[i-1] == 'C':
            skew[i] -= 1
        if Genome[i-1] == 'G':
            skew[i] += 1
    return skew

def MinimumSkew(Genome):
    positions = [] # output variable
    sk = Skew(Genome)
    minimum = min(sk.values())
    for k,v in sk.items():
        if v == minimum:
            positions.append(k)
    return positions

def PatternCount(Pattern, Text):
    count = 0
    for i in range(len(Text)-len(Pattern)+1):
        if Text[i:i+len(Pattern)] == Pattern:
            count = count+1
    return count

def PatternMatching(Pattern, Genome):
    positions = []
    for i in range(len(Genome)-len(Pattern)+1):
        if Genome[i:i+len(Pattern)] == Pattern:
            positions.append(i)
    return positions

def HammingDistance(p, q):
    n = len(p)
    distance = 0
    for i in range(n):
        if (p[i] != q[i]):
            distance += 1
    return distance

def ApproximatePatternMatching(Pattern, Text, d):
    positions = [] # initializing list of positions
    for i in range(len(Text)-len(Pattern)+1):
        if HammingDistance(Text[i:i+len(Pattern)],Pattern)<=d:
            positions.append(i)
    return positions

def ApproximatePatternCount(Pattern, Text, d):
    count = 0 # initialize count variable
    for i in range(len(Text)-len(Pattern)+1):
        if HammingDistance(Text[i:i+len(Pattern)],Pattern)<=d:
            count += 1
    return count

p = 'TGACCCGTTATGCTCGAGTTCGGTCAGAGCGTCATTGCGAGTAGTCGTTTGCTTTCTCAAACTCC'
q = 'GAGCGATTAAGCGTGACAGCCCCAGGGAACCCACAAAACGTGATCGCAGTCCATCCGATCATACA'
Pattern = 'GAGG'
Genome = 'GATACACTTCCCGAGTAGGTACTG'
mini = MinimumSkew(Genome)
pos = HammingDistance(p, q)
print(pos)
