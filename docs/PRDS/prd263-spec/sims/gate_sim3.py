# Revised gate: escalation and stable-row guard use SAME-DAY passes only (no calibration oracle).
import random
from math import comb
random.seed(23)
def sign_p(w,l):
    n=w+l
    return 1.0 if n==0 else sum(comb(n,i) for i in range(w,n+1))/2**n
def arm(p,k): return sum(random.random()<p for _ in range(k))
def trial(base,cand,safety,mode):
    W=L=0
    for i,(pb,pc) in enumerate(zip(base,cand)):
        b=arm(pb,3); c=arm(pc,3); nb=nc=3
        if (b>=2)!=(c>=2) or b in (1,2) or c in (1,2):
            b+=arm(pb,2); c+=arm(pc,2); nb=nc=5
        bm=b*2>nb; cm=c*2>nc
        if cm and not bm: W+=1
        if bm and not cm:
            L+=1
            if i in safety: return False
        # stable-row guard: base passed every same-day pass, candidate failed any
        if b==nb and c<nc:
            if 3-arm(pc,3)>=2: return False
    if mode=='improve': return sign_p(W,L)<=0.05
    return (L-W)<=1
def rate(base,cand,safety,mode,n):
    return sum(trial(base,cand,safety,mode) for _ in range(n))/n
def today(): return [0.995]*57+[0.01]*2+[0.5]*6
S=set(range(13))
print("65 rows, IMPROVE null:", rate(today(),today(),S,'improve',20000))
for f in (5,8):
    c=today()
    for j in range(f): c[57+j]=0.9
    print(f" 65 rows fixes={f}:", rate(today(),c,S,'improve',6000))
print("65 rows NON-INFERIOR equal:", rate(today(),today(),S,'ni',8000))
for pb in (0.0,0.3,0.6):
    out=[]
    for m in (1,2,3,4):
        c=today()
        for j in range(m): c[13+j]=pb
        out.append(round(rate(today(),c,S,'ni',6000),3))
    print(" NI p_break",pb,out)
def v2(): return [0.995]*25+[0.05]*8+[0.5]*5
print("38-row v2 held-out, IMPROVE null:", rate(v2(),v2(),S,'improve',20000))
for ff in (4,6,8):
    c=v2()
    for j in range(ff): c[25+j]=0.9
    print(f" v2 fixes fail={ff}:", rate(v2(),c,S,'improve',6000))
