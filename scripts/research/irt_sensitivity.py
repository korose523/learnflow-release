"""Exploratory full-item A1 sensitivities; sparse implementation of the stored EM.
All responses remain local. The original capped file-order baseline is retained.
"""
import argparse,importlib.util,sys,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import expit,logsumexp
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
def fit(p,j,c,nperson,nitem,n_iter=25,n_nodes=15,**kwargs):
 p=np.asarray(p);j=np.asarray(j);c=np.asarray(c)
 A=coo_matrix((np.ones(len(p)),(p,j)),shape=(nperson,nitem)).tocsr()
 C=coo_matrix((c,(p,j)),shape=(nperson,nitem)).tocsr()
 theta=np.linspace(-5,5,n_nodes);lw=-theta**2/2;lw-=logsumexp(lw);b=np.zeros(nitem);trace=[]
 for it in range(n_iter):
  x=theta[None,:]-b[:,None]
  ll=C@(-np.logaddexp(0,-x))+(A-C)@(-np.logaddexp(0,x))+lw
  post=np.exp(ll-logsumexp(ll,axis=1)[:,None]);R=A.T@post;S=np.asarray(C.sum(axis=0)).ravel();max_delta=0.
  for k in range(nitem):
   bj=b[k]
   for _ in range(25):
    pr=expit(theta-bj);f=R[k]@pr-S[k];df=-(R[k]@(pr*(1-pr)))
    if abs(df)<1e-12:break
    step=np.clip(f/df,-2,2);bj-=step
    if abs(step)<1e-6:break
    max_delta=max(max_delta,abs(step))
   b[k]=bj
  trace.append(float(max_delta))
  if max_delta<1e-4 and it>4:break
 fit.trace=trace
 return b.tolist()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
 if a.output_dir.exists():ap.error('Output directory must be new')
 a.output_dir.mkdir(parents=True)
 sys.path.insert(0,str(ROOT/'results/m1'))
 spec=importlib.util.spec_from_file_location('a1',ROOT/'results/m1/run_a1a2_fit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 arrays={}
 def capture(p,j,c,nperson,nitem,**kwargs):
  arrays.update(p=np.asarray(p),j=np.asarray(j),c=np.asarray(c),nperson=nperson,nitem=nitem)
  np.savez_compressed(a.output_dir/'responses_local.npz',**arrays)
  return fit(p,j,c,nperson,nitem,**kwargs)
 m.fit_irt_1pl_scaled=capture
 baseline=m.run_a1(); (a.output_dir/'sparse_baseline.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2))
 vec=baseline['aligned_item_vectors'];p=arrays['p'];j=arrays['j'];c=arrays['c'];nperson=arrays['nperson'];nitem=arrays['nitem']
 rows=[]
 for label,it,nodes,selection in [('EM50',50,15,'all'),('nodes21',25,21,'all'),('first750',25,15,'first'),('last750_of_first1500',25,15,'last')]:
  start=time.time();keep=np.ones(len(p),dtype=bool)
  if selection!='all':
   count=np.zeros(nitem,dtype=int);total=np.bincount(j,minlength=nitem)
   for ix,jj in enumerate(j):
    keep[ix]=count[jj]<750 if selection=='first' else count[jj]>=max(0,total[jj]-750)
    count[jj]+=1
  b=fit(p[keep],j[keep],c[keep],nperson,nitem,n_iter=it,n_nodes=nodes)
  rf=float(spearmanr(b,vec['fusion_a_half']).statistic);rs=float(spearmanr(b,vec['success_a_half']).statistic)
  row={'setting':label,'max_em':it,'nodes':nodes,'n_responses':int(keep.sum()),'n_items':nitem,'fusion_rho':rf,'success_rho':rs,'fusion_minus_success':rf-rs,'criterion_rank_rho_vs_baseline':float(spearmanr(b,vec['criterion_b_half_irt']).statistic),'mean_abs_criterion_change':float(np.mean(np.abs(np.asarray(b)-vec['criterion_b_half_irt']))),'iterations_executed':len(fit.trace),'max_newton_step_trace':fit.trace,'elapsed_seconds':time.time()-start}
  rows.append(row);(a.output_dir/(label+'.json')).write_text(json.dumps({'summary':row,'criterion':b},indent=2));print(row,flush=True)
 (a.output_dir/'sensitivity.json').write_text(json.dumps({'scope':'Exploratory conditional fitted-criterion sensitivity; same fixed A-half predictors and item set; no learner bootstrap, no new inference; last750 is only within original first1500, not last records of full corpus','baseline':{'fusion_rho':baseline['spearman_fusion_vs_bj'],'success_rho':baseline['spearman_successrate_vs_bj']},'rows':rows},indent=2))
if __name__=='__main__':main()
