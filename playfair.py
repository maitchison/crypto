import random
from difflib import SequenceMatcher

""" todo:
proper annealing algorithm
clean up how crib / no crib works
look at increasing performance

look into just printing the best score so far
"""

import math
import numpy as np
import time

# just try random permutations
ALPHA_MINUS_J = "abcdefghiklmnopqrstuvwxyz"

def get_pairs(s):
    pairs = []
    last = None
    for c in s:
        if last == None:
            last = c
        else:
            if last == c:
                pairs.append((last, 'x'))
            else:
                pairs.append((last,c))
                last = None
    if last != None:
        pairs.append((last,'x'))
    return pairs


def txt(pair):
    return pair[0]+pair[1]

def get(grid,x,y):
    x = (x + 5) % 5
    y = (y + 5) % 5
    return grid[(x,y)]

def lookup(grid, pair):
    x1, y1 = grid[pair[0]]
    x2, y2 = grid[pair[1]]
    if x1 == x2:
        return (get(grid, x1, y1+1),get(grid, x1, y2+1))
    if y1 == y2:
        return (get(grid, x1+1, y1),get(grid, x2+1, y1))
    return (grid[(x1,y2)], grid[(x2,y1)])


def reverse_lookup(grid, pair):
    x1, y1 = grid[pair[0]]
    x2, y2 = grid[pair[1]]
    if x1 == x2:
        return (get(grid, x1, y1-1),get(grid, x1, y2-1))
    if y1 == y2:
        return (get(grid, x1-1, y1),get(grid, x2-1, y1))
    return (grid[(x2,y1)],grid[(x1,y2)])


def tidy(s):
    s = s.replace("j", "i")
    s = s.replace(" ", "")
    return s


def get_grid(key):
    grid = {}
    for x in range(0, 5):
        for y in range(0, 5):
            grid[(x, y)] = key[x + y * 5]
            grid[key[x + y * 5]] = (x, y)
    return grid

def encrypt(message, key):
    message = tidy(message)
    grid = get_grid(key)
    pairs = get_pairs(message)
    result = ""
    for pair in pairs:
        result += txt(lookup(grid, pair))
    return result


def decrypt(encoded_text, key):
    grid = get_grid(key)
    pairs = get_pairs(encoded_text)
    result = ""
    for pair in pairs:
        result += txt(reverse_lookup(grid, pair))
    return result

def make_key(input):
    input = tidy(input)
    key = ""
    for i in range(25):
        #try next letter
        while len(input) > 0:
            c = input[0]
            input = input[1:]
            if c not in key:
                key += c
                break
        else:
            # while failed, we need to generate the next netter
            for c in ALPHA_MINUS_J:
                if c not in key:
                    key += c
                    break
    return key

def show_key(key):
    for i in range(5):
        print(key[i*5:i*5+5])

def match_strings_single(a,b):
    result = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            result += 1
    return result

# match pair at a time
def match_strings_paired(a,b):
    result = 0
    for i in range(len(a)//2):
        if a[i*2] == b[i*2] and a[i*2+1] == b[i*2+1]:
            result += 2
    return result

def match_strings(a,b, paired = False):
    if paired:
        return match_strings_paired(a, b)
    else:
        return match_strings_single(a, b)

""" Find the fitness of given text with a specified floating pad. """
def get_fitness(text, crib, crib_pos = None):

    PAIRED_MATCH = True

    if PAIRED_MATCH and crib_pos and crib_pos % 2 == 1:
        # remove first character
        crib_pos += 1
        crib = crib[1:]

    n = len(crib)

    fitness = 0
    if crib_pos:
        return match_strings(text[crib_pos:crib_pos+n],crib, PAIRED_MATCH)

    """ sliding window... quite slow :( """
    for i in range(len(text)-n):
        a = text[i:i+n]
        b = crib
        if PAIRED_MATCH and i % 2 == 1:
            # remove first character so pairs line up
            a = a[1:]
            b = b[1:]

        fitness = max(fitness, match_strings(a, b, PAIRED_MATCH))
    return fitness


def get_matching(a,b):
    s = ""
    for i in range(len(a)):
        if a[i] == b[i]:
            s += a[i]
        else:
            s += "-"
    return s

""" Check how 'english' this sentance is by looking up all combinations in an english dictionary. """
def get_englishness(s):
    score = 0
    for word_len in range(3,9):
        for i in range(len(s)-word_len):
            if s[i:i+word_len] in dict:
                score += dict[s[i:i+word_len]]
    return score / 10


def mutate(x):
    result = x.copy()
    a = random.randint(0, 24)
    b = random.randint(0, 24)
    result[a] = x[b]
    result[b] = x[a]
    return result


def annealing_attack(encrpyted_text, crib = None, crib_pos = None, CRIB_WEIGHT = 1.0, ENGLISH_WEIGHT = 1.0):
    print("Beggining annealing attack...")

    if not crib:
        CRIB_WEIGHT = 0.0

    if crib_pos and crib_pos < 0:
        crib_pos = len(encrpyted_text) - len(crib) - 1

    # 4 works very well with crib...
    if CRIB_WEIGHT > 0:
        BASE_T = 0.6
    else:
        BASE_T = 0.6

    x = list(ALPHA_MINUS_J)
    random.shuffle(x)
    B = None
    e = None


    best_text = ""
    best_solution = ""
    accept = 0
    reject = 0
    time_since_last_best = 0

    start_time = time.time()

    abs_best = ""
    abs_best_e = 0

    DECAY_EVERY = 1000

    NO_PROGRESS_THRESHOLD = 40000

    """
    if crib and not crib_pos:
        # takes longer to solve these ones
        NO_PROGRESS_THRESHOLD = 60000
        #DECAY_EVERY = 1200
        BASE_T = 0.4
        ENGLISH_WEIGHT *= 0.1 # it's harder to learn the position of the crib, so we don't want the 'englishness' taking over...
    """

    # apply english rules only after we get a crib with this many correct places.
    ENGLISH_THRESHOLD = (len(crib) - crib.count("-")) * 0.5

    T = BASE_T

    max_t = int(1e8)

    for t in range(1, max_t):

        if t % DECAY_EVERY == 0:
            T = T * 0.99

        x_prime = mutate(x)
        # occasional chance to mutate twice.
        if random.randint(1,10) == 1:
            x_prime = mutate(x_prime)

        decoded = decrypt(encrpyted_text, "".join(x_prime))

        crib_score = get_fitness(decoded, crib, crib_pos) if CRIB_WEIGHT > 0 else 0
        english_score = get_englishness(decoded) if (crib_score > ENGLISH_THRESHOLD) & (ENGLISH_WEIGHT > 0) else 0
        e_prime = CRIB_WEIGHT * crib_score + ENGLISH_WEIGHT * english_score
        e = e if e else e_prime
        B = B if B else e_prime

        delta_e = e - e_prime

        if delta_e <= 0:
            x = x_prime
            e = e_prime
            reject += 1
        else:
            h = math.exp(-delta_e / T)
            U = np.random.rand()
            if U < h:
                x = x_prime
                e = e_prime
                accept += 1
            else:
                reject += 1
        if e > B:
            B = e
            best_solution = x
            best_text = decoded
            time_since_last_best = 0
            if e > abs_best_e:
                abs_best_e = e
                abs_best = decoded

        if t % 5000 == 0:
            accept_percent = accept / (accept+reject) * 100
            print("{0:.2f}% {4:.5f} {1:.2f}/{3:.2f} {2} ({5:.2f}%) {6}".format((t / max_t) * 100, e, abs_best, B, T, accept_percent, get_matching(best_text, solution)))
            accept = 0
            reject = 0


        if decoded == solution:
            print("{0:.2f}% {4:.5f} {1:.2f}/{3:.2f} {2} ({5:.2f}%) {6}".format((t / max_t) * 100, e, abs_best, B, T, accept_percent, get_matching(best_text, solution)))
            print("Success.")
            current_time = time.time()
            print("took {0:.1f} minutes".format((current_time-start_time) / 60))
            show_key(best_solution)
            return

        if time_since_last_best > NO_PROGRESS_THRESHOLD:
            print("Reset.")
            x = list(ALPHA_MINUS_J)
            random.shuffle(x)
            B = None
            e = None
            time_since_last_best = 0
            T = BASE_T

        time_since_last_best += 1

    print("Best solution:")
    show_key(best_solution)


key = make_key("manchester street")
print("The key:")
show_key(key)

print("Loading dictionary")
f = open("20k.txt")
dict = {}
for line in f:
    word = line.strip().lower().replace("j","i")
    dict[word] = len(word) / 4

dict["stop"] = 5
#dict["norskhydro"] = 10
#dict["norsk"] = 5
#dict["hydro"] = 5

print(" - read {0} words.".format(len(dict)))

solution = "mosturgentstopallmembersofgliderteamkilxledstopincontactwithnorskhydroinformantstopredpenguinfrenzystopdonotsendfolxlowupteamuntilligivecoordinatesandtimeforsafelandingzoneendx"

message = "".join([
    "FVLYP IPGLU LYPQH FFSDE MDHEV OKNCB GEPSM FNCKY"
    ,"GSSBU PURKI UFOHH QZRYS FUHEL CXSAP BUOVA EIFYL"
    ,"UPWED SWGFK ZBFGE GUIHL UPQEU FPUBD KBOVK YFTZP"
    ,"QUMRB OLUHN NHNRW MAQPA BCFIP SMHKB UHEDO VHEMO"
    ,"SGIFB CFKVU GBBGK C"]).lower().replace(" ","")

key = "TUOPQYZVWXHIARCSBMEDLNFGK".lower()
print(" "*18,message)
print(" "*18,get_matching(solution,decrypt(message, key)))

random.seed = 1234
np.random.seed(1234)

# solves trivially
#annealing_attack(message,"stopredpenguinfrenzystop",79)

# takes a bit of time, but will work.
annealing_attack(message,"stopredpenguinfrenzystop")

# we can pad if we want
#annealing_attack(message,"stop----enguinfrenzystop",79)

# this is too hard to solve...
#annealing_attack(message,"end",-1)

#annealing_attack(message)


