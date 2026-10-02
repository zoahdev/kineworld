#!/usr/bin/env python3
"""Contact world-model audit. CPU-only, no network, standalone synthetic data."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def true_step(s, a):
    s=np.asarray(s); a=np.asarray(a)
    w=.9*s[...,1]+.18*a
    q=s[...,0]+w
    hit=(q<0)|(q>1)
    return np.stack([np.clip(q,0,1),np.where(hit,-.6*w,w)],axis=-1)


def dataset(seed,n):
    r=np.random.default_rng(seed)
    wide=r.random(n)<.1
    x=np.where(wide,r.uniform(0,1,n),r.uniform(.25,.75,n))
    v=np.where(wide,r.uniform(-.5,.5,n),r.uniform(-.15,.15,n))
    a=r.uniform(-1,1,n)
    X=np.column_stack([x,v,a]); Y=true_step(X[:,:2],a)
    return X,Y


class Model:
    def __init__(self, name, model=None): self.name=name; self.model=model
    def fit(self,X,Y):
        if self.name=='hybrid':
            interior=(Y[:,0]>0)&(Y[:,0]<1)
            self.free=Ridge(alpha=1e-12,fit_intercept=False).fit(X[interior,1:],Y[interior,1])
            w=self.free.predict(X[:,1:]); hits=~interior
            if not np.any(hits): raise ValueError('No training contacts; hybrid not identified')
            self.e=-float(w[hits] @ Y[hits,1]) / float(w[hits] @ w[hits])
            self.lo=float(Y[:,0].min()); self.hi=float(Y[:,0].max())
        else: self.model.fit(X[:,:2] if self.name=='blind_ridge' else X,Y)
        return self
    def predict(self,X):
        if self.name=='oracle': return true_step(X[:,:2],X[:,2])
        if self.name=='hybrid':
            w=self.free.predict(X[:,1:]); q=X[:,0]+w
            return np.column_stack([np.clip(q,self.lo,self.hi),np.where((q<self.lo)|(q>self.hi),-self.e*w,w)])
        return self.model.predict(X[:,:2] if self.name=='blind_ridge' else X)


def fit_models(X,Y,V,W,seed):
    models=[]; selections={}
    for name in ['blind_ridge','ridge','poly2']:
        candidates=[]
        for alpha in [1e-6,1e-3,1.]:
            pipeline=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=alpha)) if name=='poly2' else make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            m=Model(name,pipeline).fit(X,Y)
            candidates.append((float(np.mean((m.predict(V)-W)**2)),alpha,m))
        loss,alpha,m=min(candidates,key=lambda z:z[0]); models.append(m); selections[name]={'alpha':alpha,'validation_mse':loss}
    candidates=[]
    for alpha in [1e-4,1e-2]:
        net=MLPRegressor(hidden_layer_sizes=(32,32),activation='tanh',solver='adam',alpha=alpha,
                         batch_size=256,learning_rate_init=.003,max_iter=300,early_stopping=True,
                         validation_fraction=.1,n_iter_no_change=20,tol=1e-6,random_state=seed)
        m=Model('mlp',make_pipeline(StandardScaler(),net)).fit(X,Y)
        candidates.append((float(np.mean((m.predict(V)-W)**2)),alpha,m))
    loss,alpha,m=min(candidates,key=lambda z:z[0]); models.append(m); selections['mlp']={'alpha':alpha,'validation_mse':loss,'iterations':m.model[-1].n_iter_}
    models.append(Model('hybrid').fit(X,Y)); models.append(Model('oracle'))
    return models,selections


def rollout(model,states,acts,goals):
    E,K,H=acts.shape
    s=np.repeat(states[:,None,:],K,axis=1).reshape(-1,2)
    g=np.repeat(goals[:,None],K,axis=1).ravel()
    cost=np.zeros(E*K)
    for t in range(H):
        a=acts[:,:,t].ravel()
        s=model.predict(np.column_stack([s,a]))
        cost+=(s[:,0]-g)**2+.1*s[:,1]**2+.02*a**2
    return cost.reshape(E,K)/H


def one_seed(seed,episodes,ntrain):
    X,Y=dataset(seed,ntrain); V,W=dataset(seed+3000,2000); T,U=dataset(seed+1000,20000)
    models,selection=fit_models(X,Y,V,W,seed)
    hit=(U[:,0]==0)|(U[:,0]==1)
    r=np.random.default_rng(seed+2000)
    states=np.column_stack([r.uniform(.45,.85,episodes),r.uniform(-.1,.35,episodes)])
    goals=r.uniform(.7,.95,episodes); acts=r.uniform(-1,1,(episodes,256,8))
    actual=rollout(Model('oracle'),states,acts,goals)
    rows=[]
    for m in models:
        err=(m.predict(T)-U)**2
        pred=actual if m.name=='oracle' else rollout(m,states,acts,goals)
        for K in [1,16,256]:
            idx=np.argmin(pred[:,:K],axis=1); chosen=actual[np.arange(episodes),idx]
            predicted=pred[np.arange(episodes),idx]; oracle=actual[:,:K].min(axis=1)
            rows.append(dict(seed=seed,model=m.name,K=K,test_mse=float(err.mean()),contact_mse=float(err[hit].mean()),
                             noncontact_mse=float(err[~hit].mean()),actual_cost=float(chosen.mean()),regret=float((chosen-oracle).mean()),
                             optimism=float((chosen-predicted).mean()),train_contacts=int(((Y[:,0]==0)|(Y[:,0]==1)).sum()),
                             test_contact_fraction=float(hit.mean())))
    return rows,selection


def mean_ci(a):
    a=np.asarray(a,float); m=float(a.mean()); n=len(a)
    if n < 2: raise ValueError('At least two independent seed summaries are required')
    if not np.all(np.isfinite(a)): raise ValueError('Non-finite summary input')
    se=float(a.std(ddof=1)/np.sqrt(n))
    rad=float(stats.t.ppf(.975,n-1)*se)
    return dict(mean=m,low=m-rad,high=m+rad,n=n)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=10);p.add_argument('--start',type=int,default=1000)
    p.add_argument('--episodes',type=int,default=128);p.add_argument('--ntrain',type=int,default=6000);p.add_argument('--output',default='results.json')
    args=p.parse_args();
    if args.seeds < 2: p.error('--seeds must be at least 2 for seed-level confidence intervals')
    rows=[]; selection={}; start=time.time()
    with threadpool_limits(limits=1):
        for seed in range(args.start,args.start+args.seeds):
            chunk,chosen=one_seed(seed,args.episodes,args.ntrain); rows.extend(chunk); selection[seed]=chosen
            print(json.dumps({'finished_seed':seed,'seconds':time.time()-start}),flush=True)
    summary=[]
    for model in sorted({r['model'] for r in rows}):
        for K in [1,16,256]:
            group=[r for r in rows if r['model']==model and r['K']==K]
            s={'model':model,'K':K}
            for key in ['test_mse','contact_mse','noncontact_mse','actual_cost','regret','optimism']:
                s[key]=mean_ci([r[key] for r in group])
            summary.append(s)
    paired={}
    for model in sorted({r['model'] for r in rows}):
        groups={K:{r['seed']:r for r in rows if r['model']==model and r['K']==K} for K in [1,16,256]}
        paired[model]={'cost_K256_minus_K16':mean_ci([groups[256][s]['actual_cost']-groups[16][s]['actual_cost'] for s in groups[16]]),
                       'regret_K256_minus_K16':mean_ci([groups[256][s]['regret']-groups[16][s]['regret'] for s in groups[16]])}
    import platform, sklearn, scipy
    config=vars(args).copy(); config['output']=Path(args.output).name
    result=dict(config=config,runtime_seconds=time.time()-start,versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__),rows=rows,summary=summary,paired=paired,selection=selection)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'saved':args.output,'seconds':result['runtime_seconds']}))
if __name__=='__main__': main()
