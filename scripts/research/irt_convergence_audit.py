"""Separate exploratory IRT numerical audit; historical criteria remain unchanged."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import expit,logsumexp
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/completion_20261009/irt_convergence'
INPUT=ROOT/'交付与证据/源数据复核_20261009/m1_sensitivity/responses_local.npz'

def solve(A,C,penalty,nodes=25,max_iter=200):
    theta=np.linspace(-5,5,nodes);lw=-theta**2/2;lw-=logsumexp(lw)
    b=np.zeros(A.shape[1]);success=np.asarray(C.sum(axis=0)).ravel()
    trace=[];stable=0
    def objective(v):
        x=theta[None,:]-v[:,None]
        ll=C@(-np.logaddexp(0,-x))+(A-C)@(-np.logaddexp(0,x))+lw
        return ll,float(logsumexp(ll,axis=1).sum()-.5*penalty*(v@v))
    ll,previous=objective(b)
    for iteration in range(1,max_iter+1):
        old=b.copy();post=np.exp(ll-logsumexp(ll,axis=1)[:,None]);R=A.T@post
        for j in range(len(b)):
            bj=b[j]
            for _ in range(50):
                pr=expit(theta-bj);g=R[j]@pr-success[j]-penalty*bj;h=-(R[j]@(pr*(1-pr)))-penalty
                if abs(h)<1e-12:break
                step=np.clip(g/h,-2,2);bj-=step
                if abs(step)<1e-8:break
            b[j]=bj
        ll,current=objective(b);delta=float(np.max(np.abs(b-old)));relative=abs(current-previous)/max(1,abs(previous))
        stable=stable+1 if delta<1e-4 and relative<1e-7 else 0
        trace.append({'iteration':iteration,'objective':current,'max_parameter_change':delta,'relative_objective_change':relative,'objective_decreased_beyond_1e_6':current<previous-1e-6})
        previous=current
        if iteration%25==0:print('penalty',penalty,'iteration',iteration,'delta',delta,flush=True)
        if stable>=3:break
    return b,trace,stable>=3

def main():
    OUT.mkdir(exist_ok=False)
    protocol={'scope':'Exploratory numerical sensitivity on the existing full-item B-half captured responses; no new data and no learner bootstrap. Not preregistered or recovery of historical convergence.','input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'nodes':25,'grid':[-5,5],'max_em':200,'initial_b':0,'settings':[{'penalty':0,'interpretation':'unpenalized marginal likelihood'},{'penalty':0.16,'interpretation':'Gaussian item prior SD=2.5, posterior-mode criterion; different estimator'}],'stopping':'max item-parameter change <1e-4 AND relative objective change <1e-7 for 3 consecutive EM iterations; this is numerical stopping, not proof of global optimality or criterion validity'}
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    d=np.load(INPUT);p=d['p'];j=d['j'];c=d['c'];n=int(d['nperson']);m=int(d['nitem'])
    A=coo_matrix((np.ones(len(p)),(p,j)),shape=(n,m)).tocsr();C=coo_matrix((c,(p,j)),shape=(n,m)).tocsr()
    N=np.asarray(A.sum(axis=0)).ravel();S=np.asarray(C.sum(axis=0)).ravel();boundary=int(np.sum((S==0)|(S==N)))
    vectors=json.loads((ROOT/'results/m1/m1_criteria_results.json').read_text())['a1']['aligned_item_vectors']
    rows=[]
    for penalty in [0,.16]:
        start=time.monotonic();b,trace,stopped=solve(A,C,penalty)
        rf=float(spearmanr(b,vectors['fusion_a_half']).statistic);rs=float(spearmanr(b,vectors['success_a_half']).statistic)
        row={'penalty':penalty,'items':m,'persons':n,'responses':len(p),'all_correct_or_all_incorrect_items':boundary,'iterations_executed':len(trace),'numerical_stopping_met':stopped,'finite_unpenalized_estimate_supported':False if penalty==0 and boundary else None,'fusion_rho':rf,'success_rho':rs,'fusion_minus_success':rf-rs,'max_abs_b':float(np.max(np.abs(b))),'objective_decreases':sum(x['objective_decreased_beyond_1e_6'] for x in trace),'elapsed_seconds':time.monotonic()-start,'trace':trace}
        rows.append(row);(OUT/f'penalty_{penalty}.json').write_text(json.dumps(row,indent=2)+'\n')
    (OUT/'SUMMARY.json').write_text(json.dumps({'protocol':protocol,'rows':rows,'interpretation':'Fixed existing A-half predictors; numerical stopping and regularization sensitivity do not validate the IRT construct or recover population uncertainty.'},indent=2)+'\n')
if __name__=='__main__':main()
