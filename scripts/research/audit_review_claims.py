"""Read stored evidence without modifying historical results or calling models."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
ROOT = Path(__file__).resolve().parents[2]

def load_module(path):
    spec = importlib.util.spec_from_file_location('protocol', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def audit():
    source = ROOT/'results/code/llm_labeling_v2.jsonl'
    m = load_module(ROOT/'results/code/o13_llm_protocol_audit.py')
    recs = m._load_records(); absolute = {r['key']: r for r in recs['E1A']}
    ids = sorted(absolute, key=int); idx = {q:i for i,q in enumerate(ids)}
    selected = [r for r in recs['E1B'] if m._rank_style(r.get('raw','')) == 'enum_array' and r.get('ok_parse')]
    wins, opp = m._constraints(selected); n=len(ids)
    graph=np.zeros((n,n),dtype=int); directed=np.zeros_like(graph)
    for (a,b),count in wins.items():
        if count: graph[idx[a],idx[b]]=graph[idx[b],idx[a]]=1; directed[idx[a],idx[b]]=1
    count,labels=connected_components(csr_matrix(graph),directed=False)
    sizes=np.bincount(labels); covered=np.flatnonzero(graph.sum(axis=1)>0)
    largest=np.flatnonzero(labels==np.argmax(sizes))
    strong, _=connected_components(csr_matrix(directed[np.ix_(largest,largest)]),directed=True,connection='strong')
    fitted=m._fit_bt(wins,opp,idx,n)
    expert=np.array([absolute[q]['expert'] for q in ids],float)
    ratings=np.array([absolute[q]['llm'] for q in ids],float)
    protocol={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'n_items':n,'valid_enum_batches':len(selected),
        'covered_items':len(covered),'isolated_items':[ids[i] for i in range(n) if not graph[i].any()],
        'undirected_components':int(count),'largest_component_items':len(largest),'largest_component_strong_components':int(strong),
        'finite_unpenalized_mle_supported':strong==1,
        'items':[{'id':q,'degree':int(graph[i].sum()),'component':int(labels[i])} for i,q in enumerate(ids)],
        'covered_absolute_spearman':float(stats.spearmanr(expert[covered],ratings[covered]).statistic),
        'covered_historical_300_iteration_score_spearman':float(stats.spearmanr(expert[covered],fitted[covered]).statistic),
        'inference_status':'WITHDRAW_FULL_BANK_BT_AND_PROTOCOL_DIFFERENCE_INTERVALS',
        'score_status':'Historical finite-iteration diagnostic; graph coverage alone does not establish MLE convergence or ranking validity.'}
    generator=load_module(ROOT/'results/code/m1_instability.py')
    domain_min=float('inf'); nonstrict=[]; excluded_saturation=[]
    for repetition in range(200):
        rng=np.random.default_rng(902000+repetition)
        evaluation,_=generator.gen_sources(500,rng); calibration,_=generator.gen_sources(500,rng)
        domain_min=min(domain_min,float(evaluation.min()),float(calibration.min()))
        for column in range(4):
            values=np.unique(np.r_[evaluation[:,column],calibration[:,column]])
            for name,(family,transform) in generator.build_transforms().items():
                if name=='probit_std' and not np.all(np.diff(transform(values,evaluation[:,column]))>0):
                    excluded_saturation.append({'repetition':repetition,'source':column,'transform':name})
                if family!='nonlinear' or name in ('logit_minmax','probit_std'): continue
                if not np.all(np.diff(transform(values,evaluation[:,column]))>0):
                    nonstrict.append({'repetition':repetition,'source':column,'transform':name})
    if domain_min<=0 or nonstrict:
        raise ValueError('Strict-domain subset failed observed-domain verification')
    stored=json.loads((ROOT/'results/code/m1_results.json').read_text())
    synthetic={}
    for name,c in stored['config'].items():
        groups={}
        for label, filt in [('full_nonlinear',lambda k,v:v['family']=='nonlinear'),('strict_domain_nonlinear',lambda k,v:v['family']=='nonlinear' and not k.endswith(('|logit_minmax','|probit_std')))]:
            rows=[v for k,v in c['per_transform_linear'].items() if filt(k,v)]
            groups[label]={'source_transform_cells':len(rows),'repeats_per_cell':stored['settings'].get('R',200),
                'mean_cell_L1':float(np.mean([v['L1_mean'] for v in rows])),
                'mean_cell_reversal_fraction':float(np.mean([v['rev_any_frac'] for v in rows]))}
        synthetic[name]=groups
    x=np.array([0.,1.,2.,3.,20.]); x=(x-x.mean())/x.std()
    y=np.array([0.,1.,9.,4.,16.]); y=(y-y.mean())/y.std(); rp=float(stats.pearsonr(x,y).statistic);rs=float(stats.spearmanr(x,y).statistic)
    residual=(y-rs*x)/np.sqrt(1-rs*rs)
    return {'protocol_graph':protocol,'synthetic_stratification':synthetic, 'strict_subset_domain_check':{'seed_base':902000,'repeats':200,'evaluation_n':500,'calibration_n':500,'minimum_generated_value':domain_min,'non_strict_numeric_cases':nonstrict,'excluded_probit_saturation_cases':excluded_saturation},
        'adjustment_counterexample':{'pearson':rp,'spearman':rs,'adjusted_covariance_ddof0':float(np.mean(residual*x)),
            'expected_covariance':(rp-rs)/np.sqrt(1-rs*rs)}}

if __name__=='__main__':
    out=ROOT/'results/review_20261009/claim_audit.json';out.parent.mkdir(parents=True,exist_ok=True)
    data=audit();out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({**data,'protocol_graph':{k:v for k,v in data['protocol_graph'].items() if k!='items'}},ensure_ascii=False,indent=2))
