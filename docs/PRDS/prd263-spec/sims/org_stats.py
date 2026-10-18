from math import comb, sqrt, erfc
def pmf(n,k,p): return comb(n,k)*p**k*(1-p)**(n-k)
def zp(x1,n1,x2,n2):  # one-sided: is cur (x1/n1) lower than ref (x2/n2)
    p1=x1/n1;p2=x2/n2;pp=(x1+x2)/(n1+n2)
    se=sqrt(pp*(1-pp)*(1/n1+1/n2)) if 0<pp<1 else 0
    if se==0: return 1.0
    z=(p2-p1)/se
    return 0.5*erfc(z/sqrt(2))
def flag_prob(pc,pr,nc=20,nr=80):
    tot=0
    for x1 in range(nc+1):
        for x2 in range(nr+1):
            r1=x1/nc;r2=x2/nr
            if r2-r1>=0.15 and r2>0 and (r2-r1)/r2>=0.20 and zp(x1,nc,x2,nr)<0.05:
                tot+=pmf(nc,x1,pc)*pmf(nr,x2,pr)
    return tot
def old_rule(p,n):  # single week vs single week, >10% relative drop in success rate
    tot=0
    for a in range(n+1):
        for b in range(n+1):
            ra=a/n; rb=b/n
            if rb>0 and (rb-ra)/rb*100>10: tot+=pmf(n,a,p)*pmf(n,b,p)
    return tot
print("false flag p=.8:",round(flag_prob(.8,.8),4)," p=.9:",round(flag_prob(.9,.9),4))
print("power .8->.5:",round(flag_prob(.5,.8),3)," .8->.6:",round(flag_prob(.6,.8),3))
print("old rule p=.8 n=10:",round(old_rule(.8,10),3)," n=20:",round(old_rule(.8,20),3))
