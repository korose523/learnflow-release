"""Retain fold-level evidence for the historical transductive O4 CV design."""
import hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/code'))
import dbe_signals as D

def main():
    ids,signals,labels=D.build(minn=30)
    M=np.asarray(signals,float);y=np.asarray(labels,float)
    folds=np.array_split(np.random.default_rng(42).permutation(len(ids)),5)
    predictions=np.empty(len(ids));rows=[]
    for fi,te in enumerate(folds):
        tr=np.concatenate([f for j,f in enumerate(folds) if j!=fi]);best=-2;weight=None
        for combo in itertools.product([0.,.5,1.,1.5,2.],repeat=M.shape[1]):
            if not sum(combo):continue
            w=np.array([c/sum(combo) for c in combo]);rho=float(spearmanr(M[tr]@w,y[tr]).statistic)
            if rho>best:best=rho;weight=w
        predictions[te]=M[te]@weight
        rows.append({'fold':fi+1,'test_indices':te.tolist(),'test_item_ids':[ids[i] for i in te],
                     'train_rho':best,'test_rho':float(spearmanr(predictions[te],y[te]).statistic),'weights':weight.tolist()})
    mean=float(np.mean([row['test_rho'] for row in rows]));old=json.loads((ROOT/'results/code/o4_fusion_results.json').read_text())
    assert round(mean,4)==old['cv5_mean_spearman_optimized'],(mean,old['cv5_mean_spearman_optimized'])
    output={'design':'Historical transductive ECDF-before-split CV, not inductive new-item validation','seed':42,
      'folds':rows,'mean_fold_spearman':mean,'pooled_out_of_fold_spearman':float(spearmanr(predictions,y).statistic),
      'pooled_interpretation':'Fold-specific weightings can have different score scales; pooled correlation is a separate descriptive quantity, not the mean-fold estimate.',
      'item_ids':ids,'labels':labels,'out_of_fold_predictions':predictions.tolist(),
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/dbe_kt22/csv/Questions.csv',ROOT/'data/dbe_kt22/csv/Transaction.csv',ROOT/'results/code/dbe_signals.py']}}
    (ROOT/'results/review_20261009/o4_cv_recheck.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Fold correlations',[x['test_rho'] for x in rows]);print('Mean',mean,'pooled',output['pooled_out_of_fold_spearman'])
if __name__=='__main__':main()
