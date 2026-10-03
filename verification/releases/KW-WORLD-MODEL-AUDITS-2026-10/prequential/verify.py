"""Bounded, protocol-specified validity checks. No private company code or training."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib,json,math
import numpy as np
from scipy.special import betaln, xlogy, xlog1py

ROOT=Path(__file__).parent
EPS=.4; TRUE=.32; ALPHA=.025; DELTA=.025; MS=(1,2,3); J=sum(6**m for m in MS)
SEEDS=(664921,1729,20261003,42,99173); T=200000

def counterexample():
    n=6;delta=F(1,20);likelihood_truth=F(1,2**n)
    naive_cutoff=F(2)/delta
    failed=sum(likelihood_truth for seq in product((0,1),repeat=n) if 1/likelihood_truth>naive_cutoff)
    assert failed==1
    full_size=2**n+1
    assert 1/likelihood_truth <= F(full_size)/delta
    return dict(T=n,delta=float(delta),naive_cardinality=2,likelihood_ratio=2**n,naive_cutoff=float(naive_cutoff),truth_exclusion_probability=str(failed),predeclared_cardinality=full_size)

def exact_risk(theta,depth=8):
    """Exact forward enumeration: observations only, actions adapt to observations."""
    eps=F(2,5); alpha=F(1,20); threshold=1/alpha; prior=[F(17,100),F(83,100)]
    badmass=F(0);mass=F(0);maxe=F(0);leaves=0
    def visit(t,y,b,p,n,k,e,bad):
        nonlocal badmass,mass,maxe,leaves
        maxe=max(maxe,e)
        if t==depth:
            mass+=p;badmass+=p*bad;leaves+=1;return
        a=y if y<2 else t%2
        pred1=theta+(1-2*theta)*(b[1] if a==0 else b[0]);pred=[1-pred1,pred1]
        for z in (0,1,2):
            prob=eps if z==2 else (1-eps)*pred[z]
            if not prob:continue
            ee=e;nn=n;kk=k
            if y<2 and z<2:
                flip=y^a^z
                denom=theta if flip else 1-theta
                assert denom>0
                q=F(2*k+1,2*n+2)
                ee=e*(q if flip else 1-q)/denom
                nn+=1;kk+=flip
            bb=pred if z==2 else [F(int(z==0)),F(int(z==1))]
            visit(t+1,z,bb,p*prob,nn,kk,ee,bad or ee>threshold)
    for y in (0,1,2):
        prob=eps if y==2 else (1-eps)*prior[y]
        bb=prior if y==2 else [F(int(y==0)),F(int(y==1))]
        visit(1,y,bb,prob,0,0,F(1),False)
    assert mass==1 and badmass<=alpha
    return dict(theta=str(theta),depth=depth,total_probability=str(mass),ever_exclusion_probability=float(badmass),exact_ever_exclusion_probability=str(badmass),declared_alpha=float(alpha),max_observed_evalue=float(maxe),positive_probability_leaves=leaves)

def matrix_check():
    H=np.array([[1-EPS,0],[0,1-EPS],[EPS,EPS]])
    worst=0.; cases=0; impossible=0
    for theta in (0.,.1,.32,.5):
        K=np.array([[1-theta,theta],[theta,1-theta]])
        Ts=[K,K[:,::-1]]
        for m in range(1,6):
            for obs in product(range(3),repeat=m):
                for acts in product(range(2),repeat=m):
                    B=np.diag(H[obs[0]])
                    for a,o in zip(acts[:-1],obs[1:]):B=np.diag(H[o])@Ts[a]@B
                    d=B.sum(axis=0); active=d>0
                    if not active.any():impossible+=1;continue
                    V=(H@Ts[acts[-1]]@B)[:,active]/d[active]
                    got=float(np.abs(V[:,0]-V[:,-1]).sum()/2)
                    expected=(1-EPS)*(1-2*theta)**m if all(o==2 for o in obs) else 0.
                    worst=max(worst,abs(got-expected));cases+=1
    assert worst<1e-12
    return dict(cases=cases,impossible_contexts_skipped=impossible,max_error=worst)

def log_e(n,k,theta):
    if n==0:return 0.
    logq=betaln(k+.5,n-k+.5)-betaln(.5,.5)
    return float(logq-xlogy(k,theta)-xlog1py(n-k,-theta))

def theta_interval(n,k):
    """Concave likelihood gives one interval. Return outward brackets + margin."""
    if n==0:return 0.,.5
    mode=min(.5,k/n);threshold=math.log(1/ALPHA)
    if log_e(n,k,mode)>threshold:return None
    # Keep endpoints OUTSIDE feasible set, to avoid narrowing it numerically.
    if log_e(n,k,0.)<=threshold:low=0.
    else:
        out=0.;inside=mode
        for _ in range(60):
            mid=(out+inside)/2
            if log_e(n,k,mid)>threshold:out=mid
            else:inside=mid
        low=max(0.,out-1e-10)
    if log_e(n,k,.5)<=threshold:high=.5
    else:
        inside=mode;out=.5
        for _ in range(60):
            mid=(inside+out)/2
            if log_e(n,k,mid)>threshold:out=mid
            else:inside=mid
        high=min(.5,out+1e-10)
    return low,high

def radius(n):
    if not n:return 1.
    return min(1.,math.sqrt(math.log(math.pi**2*J*8*n*n/(6*DELTA))/(2*n)))

def all_contexts(m):
    # Key is o1,a1,o2,...,om,am; each transition histogram has next observation.
    for obs in product(range(3),repeat=m):
        for acts in product(range(2),repeat=m):
            yield tuple(v for pair in zip(obs,acts) for v in pair)

def summarize(counts,n,k):
    interval=theta_interval(n,k)
    if interval is None:return dict(informative_transitions=n,flips=k,theta_interval=None,selected=None)
    low,high=interval;rows=[]
    for m in MS:
        rr=[];bc=[];ns=[];et=[];initial=[]
        for c in all_contexts(m):
            nn=int(counts[m][c].sum());bb=radius(nn)
            eta=(1-EPS)*(1-2*low)**m if all(y==2 for y in c[::2]) else 0.
            rho=min(1.,bb+eta)
            rr.append(rho);bc.append(bb);ns.append(nn);et.append(eta)
            initial.append(min(1.,bb+(1-EPS if all(y==2 for y in c[::2]) else 0.)))
        rows.append(dict(memory=m,min_count=min(ns),max_count=max(ns),max_data_radius=max(bc),max_envelope=max(et),max_combined_radius=max(rr),policy_loss_bound=min(2.,4*max(rr)+1e-8),broad_prior_policy_bound=min(2.,4*max(initial)+1e-8)))
    selected=next((r['memory'] for r in rows if r['policy_loss_bound']<=.55),None)
    return dict(informative_transitions=n,flips=k,theta_interval=list(interval),rows=rows,selected=selected)

def generate(seed):
    """Simulator only; X is never returned or accessed by inference/planning."""
    rng=np.random.default_rng(seed)
    acts=rng.integers(0,2,T,dtype=np.int8)
    flips=(rng.random(T)<TRUE).astype(np.int8)
    erased=rng.random(T+1)<EPS
    x=np.empty(T+1,dtype=np.int8);x[0]=int(rng.random()<.83)
    x[1:]=x[0]^np.bitwise_xor.accumulate(acts^flips)
    obs=x.copy();obs[erased]=2
    return obs,acts

def solve_mdp(counts,m):
    states=[c[:-1] for c in all_contexts(m)]
    states=list(dict.fromkeys(states));ix={s:i for i,s in enumerate(states)}
    ns=len(states);P=np.zeros((ns,2,ns));R=np.zeros((ns,2))
    for i,s in enumerate(states):
        for a in (0,1):
            z=counts[m][s+(a,)];tot=z.sum();q=z/tot if tot else np.ones(3)/3
            R[i,a]=.95*int(s[-1]==1)+.05*(1-a)
            for o in range(3):
                nxt=(o,) if m==1 else s[2:]+(a,o)
                P[i,a,ix[nxt]]+=q[o]
    v=np.zeros(ns)
    for it in range(1000):
        Q=R+.5*np.einsum('iaj,j->ia',P,v);nv=Q.max(axis=1)
        residual=float(abs(nv-v).max());v=nv
        if residual<=1e-12:break
    pi=(R+.5*np.einsum('iaj,j->ia',P,v)).argmax(axis=1)
    vpi=np.linalg.solve(np.eye(ns)-.5*P[np.arange(ns),pi],R[np.arange(ns),pi])
    optimality_gap_upper=2*residual/(1-.5)
    assert optimality_gap_upper<1e-8 and abs(vpi-v).max()<1e-8
    return dict(states=ns,iterations=it+1,bellman_residual=residual,certified_empirical_policy_gap_upper=optimality_gap_upper,policy_action_counts=[int((pi==a).sum()) for a in (0,1)],value_range=[float(vpi.min()),float(vpi.max())])

def run(seed):
    obs,acts=generate(seed)
    counts={m:defaultdict(lambda:np.zeros(3,dtype=np.int64)) for m in MS}
    n=k=0; history=[];checkpoints={0:summarize(counts,0,0)}
    maxloge=0.
    # Update e-value sequentially rather than reevaluate special functions at all t.
    currloge=0.
    for t in range(T):
        y=int(obs[t]);a=int(acts[t]);z=int(obs[t+1]);history.extend([y,a])
        for m in MS:
            if t+1>=m:counts[m][tuple(history[-2*m:])][z]+=1
        if y<2 and z<2:
            flip=y^a^z;q=(k+.5)/(n+1)
            currloge+=math.log(q/TRUE if flip else (1-q)/(1-TRUE))
            maxloge=max(maxloge,currloge)
            n+=1;k+=flip
        if t+1 in (2000,20000,T):checkpoints[t+1]=summarize(counts,n,k)
    assert abs(currloge-log_e(n,k,TRUE))<2e-7
    last=checkpoints[T]
    result=dict(seed=seed,checkpoints=checkpoints,max_truth_evalue=math.exp(maxloge),truth_ever_excluded=maxloge>math.log(1/ALPHA),observation_action_sha256=hashlib.sha256(obs.tobytes()+acts.tobytes()).hexdigest())
    if last['selected']:result['empirical_planning']=solve_mdp(counts,last['selected'])
    return result

def nuisance_profile_nonmartingale():
    before=F(1,2)/F(9,10)
    after_if0=F(1,4)/F(9,100)
    after_if1=F(1,4)/F(81,100)
    expectation=F(9,10)*after_if0+F(1,10)*after_if1
    assert expectation>before
    return dict(before=str(before),conditional_expected_next=str(expectation),remark='Profiled ratio retains Ville coverage by domination; it need not itself be a supermartingale.')

if __name__=='__main__':
    out=dict(protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),same_data_counterexample=counterexample(),exact_adaptive_tests=[exact_risk(t) for t in (F(0),F(1,8),F(8,25),F(1,2))],matrix_envelope_check=matrix_check(),nuisance_profile_nonmartingale=nuisance_profile_nonmartingale(),runs=[run(s) for s in SEEDS])
    (ROOT/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2,allow_nan=False))
