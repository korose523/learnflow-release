"""Post-review offline item-selection stress test. No response records are exported."""
from pathlib import Path
import csv,hashlib,json,math
from collections import defaultdict
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[2]
def minmax(x):
 return (x-x.min())/(x.max()-x.min()) if x.max()>x.min() else np.zeros_like(x)
def ordinal_logit(x):
 u=(np.argsort(np.argsort(x))+.5)/len(x);u=np.clip(u,1e-4,1-1e-4);return np.log(u/(1-u))
def contributions(z,w):
 s=z@w;return w*np.mean((z-z.mean(axis=0))*(s-s.mean())[:,None],axis=0)/s.var()
def main():
 src=ROOT/'data/assist09_corrected.csv';out=ROOT/'results/venue_review_20261009/m1_selection_sensitivity.json'
 if out.exists():raise FileExistsError(out)
 d=defaultdict(lambda:[0,0,[],[]])
 with src.open(encoding='utf-8',errors='replace') as f:
  for r in csv.DictReader(f):
   if r.get('correct') not in ('0','1') or r.get('original')!='1':continue
   v=d[r['problem_id']];v[0]+=1;v[1]+=int(r['correct'])
   for j,k in [(2,'ms_first_response'),(3,'attempt_count')]:
    try:v[j].append(float(r[k]))
    except ValueError:pass
 ids=[];cols=[]
 for q,(n,c,t,a) in d.items():
  if n>=50 and t and a:
   p=(c+.5)/(n+1);ids.append(q);cols.append([math.log((1-p)/p),float(np.median(t)),float(np.mean(a))])
 x=np.array(cols);w=np.array([.4,.3,.3]);n=len(ids);k=math.ceil(.1*n)
 transforms={'logit_minmax':lambda a:np.log(np.clip(minmax(a),1e-4,1-1e-4)/(1-np.clip(minmax(a),1e-4,1-1e-4))),'probit_std':lambda a:stats.norm.cdf((a-a.mean())/(a.std()+1e-12)),'log':lambda a:np.log(a-a.min()+1e-6)}
 def selected(s):return set(sorted(range(n),key=lambda i:(-float(s[i]),ids[i]))[:k])
 records=[]
 for metric,prep in [('minmax',minmax),('historical_ordinal_logit',ordinal_logit),('average_rank_logit',lambda a:stats.logistic.ppf((stats.rankdata(a,method='average')-.5)/len(a)))]:
  z=np.column_stack([prep(a) for a in x.T]);s=z@w;pi=contributions(z,w);base=selected(s)
  for j,name in enumerate(['success_log_odds_difficulty','median_first_response_ms','mean_attempt_count']):
   for t,tf in transforms.items():
    a=x[:,j];b=tf(a);u,idx=np.unique(a,return_index=True);strict=bool(np.all(np.diff(b[idx])>0))
    zz=z.copy();zz[:,j]=prep(b);ss=zz@w;new=selected(ss);delta=contributions(zz,w)-pi
    records.append({'metric':metric,'column':name,'transform':t,'strict_on_observed_unique_values':strict,'introduced_ties':int(len(np.unique(a))-len(np.unique(b))),'L1_contribution_drift_ddof0':float(np.abs(delta).sum()),'score_spearman':float(stats.spearmanr(s,ss).statistic),'mean_absolute_percentile_rank_change':float(np.mean(np.abs(stats.rankdata(s)-stats.rankdata(ss)))/n),'top_k_replaced':len(base-new),'top_k_replacement_fraction':len(base-new)/k,'top_k_jaccard':len(base&new)/len(base|new)})
 groups={}
 for metric in ['minmax','historical_ordinal_logit','average_rank_logit']:
  for scope in ['full_library','observed_strict_subset']:
   r=[v for v in records if v['metric']==metric and (scope=='full_library' or v['strict_on_observed_unique_values'])];groups[metric+'|'+scope]={'cells':len(r),'mean_L1':float(np.mean([v['L1_contribution_drift_ddof0'] for v in r])),'mean_selection_replacement_fraction':float(np.mean([v['top_k_replacement_fraction'] for v in r])),'max_selection_replacement_fraction':max(v['top_k_replacement_fraction'] for v in r)}
 result={'status':'POST_REVIEW_EXPLORATORY_STRESS_TEST_NOT_LEARNING_OR_CRITERION_VALIDATION','protocol_fixed_before_execution':{'input':'original=1; correct in 0,1; at least 50 original responses per item; nonempty time/attempt arrays','direction':'All three observed columns numerically oriented upward; does not validate common construct','nominal_weights':w.tolist(),'selection':'Largest fused scores; k=ceil(0.10*n), deterministic item-ID tie-break','calibration':'Same fixed item bank for linking and evaluation, not inductive','strict_test':'Each transform is tested over observed unique source values with full-bank parameters; strict subset excludes numerical plateaus','covariance':'All covariance and variance use ddof=0; historical O1 mixes ddof=1 covariance with ddof=0 variance','uncertainty':'No inferential CI, fixed-bank deterministic stress test','interpretation':'Alternative representations are stress cases, not established admissible changes of a shared interval scale'},'items':n,'top_k':k,'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'summary':groups,'cells':records}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'items':n,'k':k,'summary':groups},indent=2))
if __name__=='__main__':main()
