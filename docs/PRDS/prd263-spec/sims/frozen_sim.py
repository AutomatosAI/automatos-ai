import random, math
from math import comb
from statistics import NormalDist
import sys


def report(*parts):
    """Write one line of results to stdout (the repo's lint bans print)."""
    sys.stdout.write(" ".join(str(p) for p in parts) + "\n")

N=NormalDist()
def pz(x_cur,n_cur,x_ref,n_ref):
    pc=x_cur/n_cur; pr=x_ref/n_ref; pp=(x_cur+x_ref)/(n_cur+n_ref)
    se=math.sqrt(pp*(1-pp)*(1/n_cur+1/n_ref)) if 0<pp<1 else 0
    if se==0: return 1.0
    return 1-N.cdf((pr-pc)/se)
def flag(x_cur,n_cur,x_ref,n_ref):
    if n_cur<20 or n_ref<20: return False
    pc=x_cur/n_cur; pr=x_ref/n_ref
    if pr-pc<0.15 or pr==0 or (pr-pc)/pr<0.20: return False
    return pz(x_cur,n_cur,x_ref,n_ref)<0.05
def strong(x_cur,n_cur,x_ref,n_ref):
    return flag(x_cur,n_cur,x_ref,n_ref) and pz(x_cur,n_cur,x_ref,n_ref)<0.001
random.seed(7)
report("--- FROZEN reference = 4 clean weeks before change (n=80); post-change weeks n=20")
report("rollback = two consecutive flags vs frozen ref, or one strong flag, within 4 post weeks")
for p0,p1 in ((0.8,0.8),(0.9,0.9),(0.8,0.5),(0.8,0.6),(0.8,0.65)):
    T=100000; rb=0; f=[0]*4; rb2=0
    for _ in range(T):
        ref=sum(random.random()<p0 for _ in range(80))
        w=[sum(random.random()<p1 for _ in range(20)) for _ in range(4)]
        fl=[flag(x,20,ref,80) for x in w]
        st=[strong(x,20,ref,80) for x in w]
        for i,x in enumerate(fl): f[i]+=x
        trig=any(st) or any(fl[i] and fl[i+1] for i in range(3))
        rb+=trig
        rb2+= (fl[0] and fl[1]) or st[0] or st[1]
    report(p0,"->",p1," P(flag) wk1-4",[round(x/T,3) for x in f]," P(rollback in 4 wks)",round(rb/T,4)," P(rollback by wk2)",round(rb2/T,4))
report("--- old rule: max of relative deltas >10%, success down OR error_rate(=1-success) up, same n both weeks")
def pmf(n,k,p): return comb(n,k)*p**k*(1-p)**(n-k)
def old2(p,n):
    t=0
    for a in range(n+1):
        for b in range(n+1):
            sa,sb=a/n,b/n; ea,eb=1-sa,1-sb
            s_down = sb>0 and (sb-sa)/sb*100>10
            e_up = (ea-eb)/eb*100>10 if eb>0 else (ea>0)
            if s_down or e_up: t+=pmf(n,a,p)*pmf(n,b,p)
    return t
for n in (10,20): report("p=.8 n=",n,round(old2(.8,n),3))
