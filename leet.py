from typing import *
from itertools import *

w = "sabbas"
n=len(w)
m = "#" + "#".join(w) + "#"
z = len(m)
p = [0] * z  
c = r = 0    
for k in range(z):
    if k < r:
        p[k] = min(r - k, p[2 * c - k])
    while k + p[k] + 1 < z and k - p[k] - 1 >= 0 and m[k + p[k] + 1] == m[k - p[k] - 1]:
        p[k] += 1
    if k + p[k] > r:
        c, r = k, k + p[k]
st, en = [], []
for k in range(1,z-1): 
    if p[k]==k:
        st.append(k-1)    
    if p[k]==z-1-k:
        en.append(k-n)


print(st, en)