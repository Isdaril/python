import math
import random

def Count(Motifs):
    count = {}
    n = len(Motifs[0])
    k = len(Motifs)
    for symbol in "ACGT":
        count[symbol] = []
        for i in range(n):
             count[symbol].append(0)

    for i in range(n):
        for j in range(k):
            symbol = Motifs[j][i]
            count[symbol][i] += 1
    return count

def Profile(Motifs):
    n = len(Motifs[0])
    k = len(Motifs)
    profile = Count(Motifs)
    for symbol in profile.keys():
        for i in range(n):
            profile[symbol][i] = profile[symbol][i]/k
    return profile

#def ProfileWithPseudocounts(Motifs):
#    n = len(Motifs[0])
#    k = len(Motifs)
#    profile = CountWithPseudocounts(Motifs)
#    for symbol in profile.keys():
#        for i in range(n):
#            profile[symbol][i] = profile[symbol][i]/(k+4)
#    return profile

def GreedyMotifSearch(Dna, k, t):
    BestMotifs = []
    for i in range(0, t):
        BestMotifs.append(Dna[i][0:k])
    n = len(Dna[0])
    for i in range(n-k+1):
        Motifs = []
        Motifs.append(Dna[0][i:i+k])
        for j in range(1, t):
            P = Profile(Motifs[0:j])
            Motifs.append(ProfileMostProbablePattern(Dna[j], k, P))
        if Score(Motifs) < Score(BestMotifs):
            BestMotifs = Motifs
    return BestMotifs

def Entropy(Motifs):
    n = len(Motifs[0])
    k = len(Motifs)
    profile = Profile(Motifs)
    entropy = 0
    for symbol in profile.keys():
        for i in range(n):
            proba = profile[symbol][i]
            if proba != 0:
                entropy -= proba * math.log(proba,2)
    return entropy

def GreedyMotifSearchWithPseudocounts(Dna, k, t):
    BestMotifs = []
    for i in range(0, t):
        BestMotifs.append(Dna[i][0:k])
    n = len(Dna[0])
    for i in range(n-k+1):
        Motifs = []
        Motifs.append(Dna[0][i:i+k])
        for j in range(1, t):
            P = ProfileWithPseudocounts(Motifs[0:j])
            Motifs.append(ProfileMostProbablePattern(Dna[j], k, P))
        if Score(Motifs) < Score(BestMotifs):
            BestMotifs = Motifs
    return BestMotifs

#Motifs = ['AACGTA', 'CCCGTT','CACCTT','GGATTA','TTCCGG']
#Motifs = ['TCGGGGGTTTTT', 'CCGGTGACTTAC','ACGGGGATTTTC','TTGGGGACTTTT','AAGGGGACTTCC','TTGGGGACTTCC','TCGGGGATTCAT','TCGGGGATTCCT','TAGGGGAACTAC','TCGGGTATAACC']
#Text = 'ACCTGTTTATTGCCTAAGTTCCGAACAAACCCAATATAGCCCGAGGGCCT'
#Dna = ['GGCGTTCAGGCA', 'AAGAATCAGTCA', 'CAAGGAGTTCGC', 'CACGTCAATCAC', 'CAATAATATTCG']
#Dna = ["GCGCCCCGCCCGGACAGCCATGCGCTAACCCTGGCTTCGATGGCGCCGGCTCAGTTAGGGCCGGAAGTCCCCAATGTGGCAGACCTTTCGCCCCTGGCGGACGAATGACCCCAGTGGCCGGGACTTCAGGCCCTATCGGAGGGCTCCGGCGCGGTGGTCGGATTTGTCTGTGGAGGTTACACCCCAATCGCAAGGATGCATTATGACCAGCGAGCTGAGCCTGGTCGCCACTGGAAAGGGGAGCAACATC", "CCGATCGGCATCACTATCGGTCCTGCGGCCGCCCATAGCGCTATATCCGGCTGGTGAAATCAATTGACAACCTTCGACTTTGAGGTGGCCTACGGCGAGGACAAGCCAGGCAAGCCAGCTGCCTCAACGCGCGCCAGTACGGGTCCATCGACCCGCGGCCCACGGGTCAAACGACCCTAGTGTTCGCTACGACGTGGTCGTACCTTCGGCAGCAGATCAGCAATAGCACCCCGACTCGAGGAGGATCCCG", "ACCGTCGATGTGCCCGGTCGCGCCGCGTCCACCTCGGTCATCGACCCCACGATGAGGACGCCATCGGCCGCGACCAAGCCCCGTGAAACTCTGACGGCGTGCTGGCCGGGCTGCGGCACCTGATCACCTTAGGGCACTTGGGCCACCACAACGGGCCGCCGGTCTCGACAGTGGCCACCACCACACAGGTGACTTCCGGCGGGACGTAAGTCCCTAACGCGTCGTTCCGCACGCGGTTAGCTTTGCTGCC", "GGGTCAGGTATATTTATCGCACACTTGGGCACATGACACACAAGCGCCAGAATCCCGGACCGAACCGAGCACCGTGGGTGGGCAGCCTCCATACAGCGATGACCTGATCGATCATCGGCCAGGGCGCCGGGCTTCCAACCGTGGCCGTCTCAGTACCCAGCCTCATTGACCCTTCGACGCATCCACTGCGCGTAAGTCGGCTCAACCCTTTCAAACCGCTGGATTACCGACCGCAGAAAGGGGGCAGGAC", "GTAGGTCAAACCGGGTGTACATACCCGCTCAATCGCCCAGCACTTCGGGCAGATCACCGGGTTTCCCCGGTATCACCAATACTGCCACCAAACACAGCAGGCGGGAAGGGGCGAAAGTCCCTTATCCGACAATAAAACTTCGCTTGTTCGACGCCCGGTTCACCCGATATGCACGGCGCCCAGCCATTCGTGACCGACGTCCCCAGCCCCAAGGCCGAACGACCCTAGGAGCCACGAGCAATTCACAGCG", "CCGCTGGCGACGCTGTTCGCCGGCAGCGTGCGTGACGACTTCGAGCTGCCCGACTACACCTGGTGACCACCGCCGACGGGCACCTCTCCGCCAGGTAGGCACGGTTTGTCGCCGGCAATGTGACCTTTGGGCGCGGTCTTGAGGACCTTCGGCCCCACCCACGAGGCCGCCGCCGGCCGATCGTATGACGTGCAATGTACGCCATAGGGTGCGTGTTACGGCGATTACCTGAAGGCGGCGGTGGTCCGGA", "GGCCAACTGCACCGCGCTCTTGATGACATCGGTGGTCACCATGGTGTCCGGCATGATCAACCTCCGCTGTTCGATATCACCCCGATCTTTCTGAACGGCGGTTGGCAGACAACAGGGTCAATGGTCCCCAAGTGGATCACCGACGGGCGCGGACAAATGGCCCGCGCTTCGGGGACTTCTGTCCCTAGCCCTGGCCACGATGGGCTGGTCGGATCAAAGGCATCCGTTTCCATCGATTAGGAGGCATCAA", "GTACATGTCCAGAGCGAGCCTCAGCTTCTGCGCAGCGACGGAAACTGCCACACTCAAAGCCTACTGGGCGCACGTGTGGCAACGAGTCGATCCACACGAAATGCCGCCGTTGGGCCGCGGACTAGCCGAATTTTCCGGGTGGTGACACAGCCCACATTTGGCATGGGACTTTCGGCCCTGTCCGCGTCCGTGTCGGCCAGACAAGCTTTGGGCATTGGCCACAATCGGGCCACAATCGAAAGCCGAGCAG", "GGCAGCTGTCGGCAACTGTAAGCCATTTCTGGGACTTTGCTGTGAAAAGCTGGGCGATGGTTGTGGACCTGGACGAGCCACCCGTGCGATAGGTGAGATTCATTCTCGCCCTGACGGGTTGCGTCTGTCATCGGTCGATAAGGACTAACGGCCCTCAGGTGGGGACCAACGCCCCTGGGAGATAGCGGTCCCCGCCAGTAACGTACCGCTGAACCGACGGGATGTATCCGCCCCAGCGAAGGAGACGGCG", "TCAGCACCATGACCGCCTGGCCACCAATCGCCCGTAACAAGCGGGACGTCCGCGACGACGCGTGCGCTAGCGCCGTGGCGGTGACAACGACCAGATATGGTCCGAGCACGCGGGCGAACCTCGTGTTCTGGCCTCGGCCAGTTGTGTAGAGCTCATCGCTGTCATCGAGCGATATCCGACCACTGATCCAAGTCGGGGGCTCTGGGGACCGAAGTCCCCGGGCTCGGAGCTATCGGACCTCACGATCACC"]
#k = 3
#t = 5
#Profile = {"A" : [0.4, 0.3, 0, 0.1, 0, 0.9], "C" : [0.2, 0.3, 0, 0.4, 0, 0.1], "G" : [0.1, 0.3, 1, 0.1, 0.5, 0], "T" : [0.3, 0.1, 0, 0.4, 0.5, 0]}

#################
# WEEK 4 : PART 2
#################
def Consensus(Motifs):
    n = len(Motifs[0])
    count = CountWithPseudocounts(Motifs)
    consensus = ""
    for i in range(n):
        m = 0
        frequentSymbol = ""
        for symbol in "ACGT":
            if count[symbol][i] > m:
                m = count[symbol][i]
                frequentSymbol = symbol
        consensus += frequentSymbol
    return consensus

def Score(Motifs):
    n = len(Motifs[0])
    k = len(Motifs)
    consensus = Consensus(Motifs)
    score = 0
    for i in range(k):
        for j in range(n):
            if Motifs[i][j] != consensus[j]:
                score +=1
    return score

def CountWithPseudocounts(Motifs):
    count = {}
    n = len(Motifs[0])
    k = len(Motifs)
    for symbol in "ACGT":
        count[symbol] = []
        for i in range(n):
             count[symbol].append(1)

    for i in range(n):
        for j in range(k):
            symbol = Motifs[j][i]
            count[symbol][i] += 1
    return count

def ProfileWithPseudocounts(Motifs):
    profile = {}
    n = len(Motifs[0])
    k = len(Motifs)
    for symbol in "ACGT":
        profile[symbol] = []
        for i in range(n):
             profile[symbol].append(1/(k+4))

    for i in range(n):
        for j in range(k):
            symbol = Motifs[j][i]
            profile[symbol][i] += 1/(k+4)
    return profile

def Pr(Text, Profile):
    n = len(Text)
    proba = 1
    for i in range(n):
        proba = proba * Profile[Text[i]][i]
    return proba

def ProfileMostProbablePattern(Text, k, Profile):
    n = len(Text)
    proba = 0
    pattern = Text[0:k]
    for i in range(n-k+1):
        tmpPr = Pr(Text[i:i+k], Profile)
        if tmpPr > proba:
            proba = tmpPr
            pattern = Text[i:i+k]
    return pattern

def Motifs(Profile, Dna):
    motifs = []
    t = len(Dna)
    k = len(Profile['A'])
    for i in range(t):
        motifs.append(ProfileMostProbablePattern(Dna[i], k, Profile))
    return motifs

def RandomMotifs(Dna, k, t):
    motifs = []
    for i in range(t):
        rand = random.randint(0, len(Dna[i])-k)
        motifs.append(Dna[i][rand:rand+k])
    return motifs

def RandomizedMotifSearch(Dna, k, t):
    motifs = RandomMotifs(Dna, k, t)
    BestMotifs = motifs
    while True:
        Profile = ProfileWithPseudocounts(motifs)
        motifs = Motifs(Profile, Dna)
        if Score(motifs) < Score(BestMotifs):
            BestMotifs = motifs
        else:
            return BestMotifs

def Normalize(Probabilities):
    proba = Probabilities
    sumProba = sum(Probabilities.values())
    for i in Probabilities.keys():
        proba[i] = Probabilities[i]/sumProba
    return proba

def WeightedDie(Probabilities):
    kmer = ''
    rand = random.uniform(0, 1)
    concatenatedSum = 0
    for i in Probabilities.keys():
        concatenatedSum += Probabilities[i]
        if rand <= concatenatedSum:
            kmer = i
            break
    return kmer

def ProfileGeneratedString(Text, Profile, k):
    n = len(Text)
    probabilities = {}
    for i in range(0,n-k+1):
        probabilities[Text[i:i+k]] = Pr(Text[i:i+k], Profile)
    probabilities = Normalize(probabilities)
    return WeightedDie(probabilities)

def GibbsSampler(Dna, k, t, N):
    motifs = RandomMotifs(Dna, k, t)
    BestMotifs = motifs
    for i in range(1,N):
        rand = random.randint(0,t-1)
        del motifs[rand]
        profile = ProfileWithPseudocounts(motifs)
        kmer = ProfileGeneratedString(Dna[rand], profile, k)
        motifs.insert(rand, kmer)
        if Score(motifs) < Score(BestMotifs):
            BestMotifs = motifs
    return BestMotifs

#Profile = {"A" : [0.8, 0, 0, 0.2], "C" : [0, 0.6, 0.2, 0], "G" : [0.2, 0.2, 0.8, 0], "T" : [0, 0.2, 0, 0.8]}
#Profile = {'A': [0.5, 0.1], 'C': [0.3, 0.2], 'G': [0.2, 0.4], 'T': [0.0, 0.3]}
#Probabilities = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}
#Text = 'AAACCCAAACCC'
#Dna = ['TTACCTTAAC', 'GATGTCTGTC', 'ACGGCGTTAG', 'CCCTAACGAG', 'CGTCAGAGGT']
#Dna = ['CGCCCCTCTCGGGGGTGTTCAGTAAACGGCCA', 'GGGCGAGGTATGTGTAAGTGCCAAGGTGCCAG', 'TAGTACCGAGACCGAAAGAAGTATACAGGCGT', 'TAGATCAAGTTTCAGGTGCACGTCGGTGAACC', 'AATCCACCAGCTCCACGTGCAATGTTGGCCTA']
#Dna = ["GCGCCCCGCCCGGACAGCCATGCGCTAACCCTGGCTTCGATGGCGCCGGCTCAGTTAGGGCCGGAAGTCCCCAATGTGGCAGACCTTTCGCCCCTGGCGGACGAATGACCCCAGTGGCCGGGACTTCAGGCCCTATCGGAGGGCTCCGGCGCGGTGGTCGGATTTGTCTGTGGAGGTTACACCCCAATCGCAAGGATGCATTATGACCAGCGAGCTGAGCCTGGTCGCCACTGGAAAGGGGAGCAACATC", "CCGATCGGCATCACTATCGGTCCTGCGGCCGCCCATAGCGCTATATCCGGCTGGTGAAATCAATTGACAACCTTCGACTTTGAGGTGGCCTACGGCGAGGACAAGCCAGGCAAGCCAGCTGCCTCAACGCGCGCCAGTACGGGTCCATCGACCCGCGGCCCACGGGTCAAACGACCCTAGTGTTCGCTACGACGTGGTCGTACCTTCGGCAGCAGATCAGCAATAGCACCCCGACTCGAGGAGGATCCCG", "ACCGTCGATGTGCCCGGTCGCGCCGCGTCCACCTCGGTCATCGACCCCACGATGAGGACGCCATCGGCCGCGACCAAGCCCCGTGAAACTCTGACGGCGTGCTGGCCGGGCTGCGGCACCTGATCACCTTAGGGCACTTGGGCCACCACAACGGGCCGCCGGTCTCGACAGTGGCCACCACCACACAGGTGACTTCCGGCGGGACGTAAGTCCCTAACGCGTCGTTCCGCACGCGGTTAGCTTTGCTGCC", "GGGTCAGGTATATTTATCGCACACTTGGGCACATGACACACAAGCGCCAGAATCCCGGACCGAACCGAGCACCGTGGGTGGGCAGCCTCCATACAGCGATGACCTGATCGATCATCGGCCAGGGCGCCGGGCTTCCAACCGTGGCCGTCTCAGTACCCAGCCTCATTGACCCTTCGACGCATCCACTGCGCGTAAGTCGGCTCAACCCTTTCAAACCGCTGGATTACCGACCGCAGAAAGGGGGCAGGAC", "GTAGGTCAAACCGGGTGTACATACCCGCTCAATCGCCCAGCACTTCGGGCAGATCACCGGGTTTCCCCGGTATCACCAATACTGCCACCAAACACAGCAGGCGGGAAGGGGCGAAAGTCCCTTATCCGACAATAAAACTTCGCTTGTTCGACGCCCGGTTCACCCGATATGCACGGCGCCCAGCCATTCGTGACCGACGTCCCCAGCCCCAAGGCCGAACGACCCTAGGAGCCACGAGCAATTCACAGCG", "CCGCTGGCGACGCTGTTCGCCGGCAGCGTGCGTGACGACTTCGAGCTGCCCGACTACACCTGGTGACCACCGCCGACGGGCACCTCTCCGCCAGGTAGGCACGGTTTGTCGCCGGCAATGTGACCTTTGGGCGCGGTCTTGAGGACCTTCGGCCCCACCCACGAGGCCGCCGCCGGCCGATCGTATGACGTGCAATGTACGCCATAGGGTGCGTGTTACGGCGATTACCTGAAGGCGGCGGTGGTCCGGA", "GGCCAACTGCACCGCGCTCTTGATGACATCGGTGGTCACCATGGTGTCCGGCATGATCAACCTCCGCTGTTCGATATCACCCCGATCTTTCTGAACGGCGGTTGGCAGACAACAGGGTCAATGGTCCCCAAGTGGATCACCGACGGGCGCGGACAAATGGCCCGCGCTTCGGGGACTTCTGTCCCTAGCCCTGGCCACGATGGGCTGGTCGGATCAAAGGCATCCGTTTCCATCGATTAGGAGGCATCAA", "GTACATGTCCAGAGCGAGCCTCAGCTTCTGCGCAGCGACGGAAACTGCCACACTCAAAGCCTACTGGGCGCACGTGTGGCAACGAGTCGATCCACACGAAATGCCGCCGTTGGGCCGCGGACTAGCCGAATTTTCCGGGTGGTGACACAGCCCACATTTGGCATGGGACTTTCGGCCCTGTCCGCGTCCGTGTCGGCCAGACAAGCTTTGGGCATTGGCCACAATCGGGCCACAATCGAAAGCCGAGCAG", "GGCAGCTGTCGGCAACTGTAAGCCATTTCTGGGACTTTGCTGTGAAAAGCTGGGCGATGGTTGTGGACCTGGACGAGCCACCCGTGCGATAGGTGAGATTCATTCTCGCCCTGACGGGTTGCGTCTGTCATCGGTCGATAAGGACTAACGGCCCTCAGGTGGGGACCAACGCCCCTGGGAGATAGCGGTCCCCGCCAGTAACGTACCGCTGAACCGACGGGATGTATCCGCCCCAGCGAAGGAGACGGCG", "TCAGCACCATGACCGCCTGGCCACCAATCGCCCGTAACAAGCGGGACGTCCGCGACGACGCGTGCGCTAGCGCCGTGGCGGTGACAACGACCAGATATGGTCCGAGCACGCGGGCGAACCTCGTGTTCTGGCCTCGGCCAGTTGTGTAGAGCTCATCGCTGTCATCGAGCGATATCCGACCACTGATCCAAGTCGGGGGCTCTGGGGACCGAAGTCCCCGGGCTCGGAGCTATCGGACCTCACGATCACC"]
k = 15
t = 10
N = 100

n = 20
def RepeatRandomizedSearch(Dna, k, N, t, n):
    motifs = GibbsSampler(Dna, k, t, N)
    BestMotifs = motifs
    for i in range(1,n):
        motifs = GibbsSampler(Dna, k, t, N)
        if Score(motifs) < Score(BestMotifs):
            BestMotifs = motifs
    return BestMotifs

#BestMotifs = RepeatRandomizedSearch(Dna, k, N, t, n)
#print(BestMotifs)
#print(Score(BestMotifs))

#####
#TEST
#####
Dna = ['ATGAGGTC', 'GCCCTAGA', 'AAATAGAT', 'TTGTGCTA']
kmers = ['GTC', 'CCC', 'ATA', 'GCT']
NewMotifs = Motifs(Profile(kmers), Dna)
#print(NewMotifs)
Probabilities = [0.15, 0.6, 0.225, 0.225, 0.3]
proba = []
Sum = sum(Probabilities)
for i in Probabilities:
    proba.append(i/Sum)
print(proba)
