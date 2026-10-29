from pathlib import Path
import sys,json,hashlib,time
import argparse
ap=argparse.ArgumentParser(description="Run frozen M1/M3 source pipelines with isolated outputs; never call a model API")
ap.add_argument('kind',choices=['m1_criteria','m1_windows','m3_calibration','m1_o1','m1_exploratory'])
ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
a=ap.parse_args(); root=a.project_root.resolve(); kind=a.kind
out=root/'交付与证据/源数据复核_20261009'/kind
out.mkdir(parents=True,exist_ok=True)
if (out/'completed.json').exists(): raise SystemExit('Already completed; refuse rerun')
start=time.time()
if kind=='m1_criteria':
 source=root/'results/m1/run_a1a2_fit.py'; code=source.read_text(); code=code.replace('out_path = os.path.join(HERE, "m1_criteria_results.json")','out_path = '+repr(str(out/'m1_criteria_results.json')))
elif kind=='m1_windows':
 source=root/'results/code/m1_real_data.py'; code=source.read_text().replace('OUT = os.path.join(BASE, "results", "code")','OUT = '+repr(str(out)))
elif kind=='m3_calibration':
 source=root/'results/code/junyi_m3.py'; code=source.read_text().split('# arm pools:')[0]; code=code.replace('E:/learnflow',str(root)); code=code.replace('CACHE = '+repr(str(root/'results/code/junyi_m3_cache.json')).replace("'",'"'),'CACHE = '+repr(str(out/'junyi_m3_cache.json')))
 # Exact double-quoted historical assignment.
 code=code.replace('CACHE = "'+str(root)+'/results/code/junyi_m3_cache.json"','CACHE = '+repr(str(out/'junyi_m3_cache.json')))
elif kind=='m1_o1':
 source=root/'results/code/optimized_experiments.py'; code=source.read_text().split('# ================= O2 DBE-KT22')[0].replace('E:/learnflow',str(root)); code+='\njson.dump(R, open('+repr(str(out/'o1_results.json'))+', "w"), indent=2)\n'
elif kind=='m1_exploratory':
 source=root/'results/code/optimized_experiments.py'; code=source.read_text().replace('E:/learnflow',str(root)).replace('OUT = os.path.join(BASE, "results", "code")','OUT = '+repr(str(out)))
else: raise ValueError(kind)
sys.path.insert(0,str(source.parent))
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
(out/'completed.json').write_text(json.dumps({'kind':kind,'source':str(source.relative_to(root)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-start,'output_redirect_only':True},indent=2))
