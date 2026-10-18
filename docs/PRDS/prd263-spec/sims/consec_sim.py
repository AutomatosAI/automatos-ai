import random, math
from statistics import NormalDist
N=NormalDist()
def flag(x_cur,n_cur,x_ref,n_ref):
    if n_cur<20 or n_ref<20: return False
    pc=x_cur/n_cur; pr=x_ref/n_ref
    if pr-pc<0.15: return False
    if pr==0 or (pr-pc)/pr<0.20: return False
    pp=(x_cur+x_ref)/(n_cur+n_ref)
    se=math.sqrt(pp*(1-pp)*(1/n_cur+1/n_ref))
    if se==0: return False
    z=(pr-pc)/se
    return 1-N.cdf(z)<0.05
def exact_single(p,nc=20,nr=80):
    from math import comb
    pc=[comb(nc,k)*p**k*(1-p)**(nc-k) for k in range(nc+1)]
    pr=[comb(nr,k)*p**k*(1-p)**(nr-k) for k in range(nr+1)]
    return sum(pc[a]*pr[b] for a in range(nc+1) for b in range(nr+1) if flag(a,nc,b,nr))
random.seed(1)
for p in (0.8,0.9):
    print("single exact p=",p, round(exact_single(p),4))
    T=200000; both=0; first=0
    for _ in range(T):
        w=[sum(random.random()<p for _ in range(20)) for _ in range(6)]
        f1=flag(w[4],20,sum(w[0:4]),80)
        f2=flag(w[5],20,sum(w[1:5]),80)
        first+=f1; both+=f1 and f2
    print(" sim single",first/T," consecutive",both/T," indep approx",(first/T)**2)
print("--- sustained step after 4 clean weeks; ref = trailing 4 weeks")
for p0,p1 in ((0.8,0.5),(0.8,0.6)):
    T=100000; f=[0]*4; reg1=0; reg2=0; regpair=0
    for _ in range(T):
        ps=[p0]*4+[p1]*4
        w=[sum(random.random()<p for _ in range(20)) for p in ps]
        fl=[flag(w[t],20,sum(w[t-4:t]),80) for t in range(4,8)]
        for i,x in enumerate(fl): f[i]+=x
        if fl[0] and fl[1]: reg1+=1
        if fl[0] and fl[1] and fl[2]: regpair+=1
    print(p0,"->",p1," P(flag) weeks1-4 after step:",[round(x/T,3) for x in f]," P(REGRESSION by wk2)",round(reg1/T,3)," P(two consecutive REGRESSION, wks1-3)",round(regpair/T,3))
print("--- with p<0.001 escape hatch; REGRESSION(t)= flag(t-1)&flag(t) or strong(t); rollback = REGRESSION at two consecutive weeks")
def strong(x_cur,n_cur,x_ref,n_ref):
    if not flag(x_cur,n_cur,x_ref,n_ref): return False
    pc=x_cur/n_cur; pr=x_ref/n_ref; pp=(x_cur+x_ref)/(n_cur+n_ref)
    se=math.sqrt(pp*(1-pp)*(1/n_cur+1/n_ref)); z=(pr-pc)/se
    return 1-N.cdf(z)<0.001
for p0,p1 in ((0.8,0.5),(0.8,0.6)):
    T=100000; rb=0; reg_any=0
    for _ in range(T):
        ps=[p0]*4+[p1]*4
        w=[sum(random.random()<p for _ in range(20)) for p in ps]
        fl=[flag(w[t],20,sum(w[t-4:t]),80) for t in range(4,8)]
        st=[strong(w[t],20,sum(w[t-4:t]),80) for t in range(4,8)]
        reg=[st[0]]+[ (fl[i-1] and fl[i]) or st[i] for i in range(1,4)]
        reg_any+=any(reg)
        rb+=any(reg[i] and reg[i+1] for i in range(3))
    print(p0,"->",p1," P(any REGRESSION in 4 wks)",round(reg_any/T,3)," P(rollback trigger in 4 wks)",round(rb/T,3))
