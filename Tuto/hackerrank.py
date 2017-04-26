import sys

def ParseInput(inp):
    lines = inp.splitlines()
    N = int(lines[0])
    strings = lines[1:N+1]
    Q = int(lines[N+1])
    queries = lines[N+2:N+Q+2]
    return((N,strings,Q,queries))

def CountQueriesinStrings(strings,queries):
    results = []
    for query in queries:
        count = 0
        #print(query)
        for string in strings:
            if query == string:
                count+=1
            #print(string)
            #print(count)
        results.append(count)
    return(results)

inp = sys.stdin.read()
(N,strings,Q,queries) = ParseInput(inp)
results = CountQueriesinStrings(strings,queries)
for res in results:
    print(res)
