"""OncoLens EDU: FastAPI local web demonstration. NOT FOR CLINICAL USE."""
from pathlib import Path
from functools import lru_cache
import math
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE=Path(__file__).resolve().parent
app=FastAPI(title='OncoLens EDU',version='1.0',description='Educational benchmark only; never use for patient care.')

@lru_cache(maxsize=1)
def bundle():
    path=BASE/'models'/'selected_model.joblib'
    if not path.exists():
        raise RuntimeError('Model missing. Run: python train_evaluate.py')
    return joblib.load(path)

@lru_cache(maxsize=1)
def examples():
    return pd.read_csv(BASE/'data'/'demo_cases.csv')

@lru_cache(maxsize=1)
def bounds():
    df=pd.read_csv(BASE/'data'/'wdbc_569.csv').drop(columns=['diagnosis'])
    return {c:[float(df[c].min()),float(df[c].max())] for c in df.columns}

class PredictionInput(BaseModel):
    features: dict[str,float] = Field(...,description='All 30 measured cytological features')

@app.get('/health')
def health():return {'status':'ok','model_loaded':(BASE/'models'/'selected_model.joblib').exists(),'clinical_use':False}

@app.get('/metadata')
def metadata():
    b=bundle()
    return {'model':b['model_name'],'n_features':len(b['feature_names']),'features':b['feature_names'],
            'threshold':b['threshold'],'example_count':len(examples()),'clinical_use':False}

@app.get('/examples')
def sample_cases():
    df=examples(); cols=bundle()['feature_names']
    return [{'id':i,'reference_label':str(row['true_diagnosis']),
             'features':{f:float(row[f]) for f in cols}} for i,row in df.iterrows()]

@app.post('/predict')
def predict(payload:PredictionInput):
    b=bundle(); names=b['feature_names'];values=payload.features
    missing=set(names)-set(values); unknown=set(values)-set(names)
    if missing or unknown:raise HTTPException(422,detail={'missing':sorted(missing),'unknown':sorted(unknown)})
    bound=bounds(); cleaned={}
    for f in names:
        v=float(values[f]); low,high=bound[f]
        if not math.isfinite(v) or v < low - .1*(high-low) or v > high+.1*(high-low):
            raise HTTPException(422,detail=f'{f}: value outside plausible reference range ({low:.4g} to {high:.4g})')
        cleaned[f]=v
    x=pd.DataFrame([{f:cleaned[f] for f in names}])
    pro=float(b['pipeline'].predict_proba(x)[0,1])
    return {'malignant_pattern_probability':round(pro,5),
            'benign_pattern_probability':round(1-pro,5),
            'predicted_pattern':'malignant-like' if pro>=b['threshold'] else 'benign-like',
            'decision_threshold':b['threshold'],'model':b['model_name'],
            'disclaimer':'Research demonstration based on a historic dataset; not a clinical diagnosis and not validated for patient care.'}

HTML=r"""<!doctype html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OncoLens EDU | MIT105</title><style>
:root{font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;color:#1a2b3d;background:#f4f7fb}
*{box-sizing:border-box}body{margin:0}.top{background:#102c48;color:white;padding:22px 5.5%}.top strong{font-size:24px}.top span{color:#c8e3e9}main{max-width:1180px;margin:26px auto;padding:0 20px}.panel{background:white;padding:26px;border:1px solid #dfe7ef;border-radius:16px;box-shadow:0 8px 22px rgba(29,45,71,.04);margin-bottom:20px}h1{font-size:28px;margin:0 0 10px}h2{font-size:19px;margin-top:0}.sub{color:#5d6978;line-height:1.6}.notice{background:#fff3e8;border-left:4px solid #dc7231;border-radius:8px;padding:13px 16px;color:#723d1d;margin-top:14px}.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:15px 0}select,button,input{font:inherit}select,button{padding:11px 14px;border-radius:9px;border:1px solid #c5d4df}button{cursor:pointer;background:#117d92;color:white;border:none;font-weight:700}button:hover{background:#086779}button:disabled{opacity:.6;cursor:wait}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.field{display:flex;flex-direction:column;gap:5px;font-size:12px;color:#466070}.field input{width:100%;padding:9px;border-radius:7px;border:1px solid #c7d7e3;color:#14283c}.grouphead{grid-column:1/-1;color:#1a536c;margin-top:12px;font-weight:800;border-bottom:1px solid #e5edf2;padding-bottom:7px}.out{padding:18px;border-radius:12px;background:#edf6fa;border:1px solid #cde8f2;margin:15px 0}.big{font-size:30px;font-weight:800;color:#1a536c}.muted{color:#607387;font-size:13px}.footer{font-size:12px;color:#6b7885;padding:20px;text-align:center}code{background:#edf2f6;padding:2px 5px;border-radius:4px}@media(max-width:760px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:460px){.grid{grid-template-columns:1fr}}
</style></head><body><div class="top"><strong>◈ OncoLens <span>EDU</span></strong><div style="opacity:.75;font-size:13px">MIT105 Data Analytics · Muhammad Awais · MLB000037 · SDG 3</div></div>
<main><div class="panel"><h1>Explainable cytology-pattern research prototype</h1><p class="sub">Compare morphological measurements from the Breast Cancer Wisconsin Diagnostic benchmark using a fitted machine-learning classifier. Load a historical sample case or enter a complete feature vector.</p><div class="notice"><b>Education and research demonstration ONLY.</b> The model is trained on historical, non-representative data. It is not approved, clinically validated, or suitable for medical screening, diagnosis, triage, or treatment.</div></div>
<div class="panel"><h2>Benchmark sample and measured inputs</h2><div class="controls"><label for="case">Historical sample:</label><select id="case" aria-label="Sample case"></select><button onclick="loadCase()" type="button">Load sample</button><button onclick="run()" id="run" type="button">Evaluate complete feature vector</button></div><div id="fields" class="grid"></div><div id="result" role="status" aria-live="polite"></div></div>
<div class="panel"><h2>How to read the output</h2><p class="sub">The score is a model-estimated probability of a malignant-like pattern in this benchmark, <b>not a patient's probability of cancer</b>. Lowering the threshold prioritised malignant recall in training-fold validation, but false negatives and false positives remain possible. Each sample contains all 30 cytology-derived numeric features.</p><p class="muted">The threshold was chosen without inspecting test outcomes. Input measurements are not routinely available to the general public. A clinician must not rely on this prototype.</p></div></main><div class="footer">Educational demonstration only · No sign-in · No data stored by this application · POST /predict API also available at /docs</div>
<script>
let cases=[],meta={};const fields=document.getElementById('fields');
function nice(s){return s.replaceAll(' ',' ').replace(/\b\w/g,c=>c.toUpperCase())}
async function init(){try{meta=await(await fetch('/metadata')).json();cases=await(await fetch('/examples')).json();
 let c=document.getElementById('case');cases.forEach((s,i)=>{let op=document.createElement('option');op.value=i;op.textContent='Case '+(i+1)+' · Reference: '+s.reference_label;c.appendChild(op)});
 let prior='';meta.features.forEach((f,i)=>{let g=f.startsWith('mean ')?'Mean features':f.endsWith(' error')?'Standard-error features':'Worst features';if(g!==prior){let p=document.createElement('div');p.className='grouphead';p.textContent=g;fields.appendChild(p);prior=g;}
 let div=document.createElement('label');div.className='field';let t=document.createElement('span');t.textContent=nice(f);let input=document.createElement('input');input.type='number';input.step='any';input.required=true;input.dataset.name=f;input.setAttribute('aria-label',f);div.append(t,input);fields.appendChild(div)});loadCase();}catch(e){document.getElementById('result').textContent='Unable to load the saved benchmark model: '+e}}
function loadCase(){const sample=cases[Number(document.getElementById('case').value)];if(!sample)return;document.querySelectorAll('[data-name]').forEach(el=>el.value=sample.features[el.dataset.name]);document.getElementById('result').textContent='Complete historical sample loaded. Click Evaluate.'}
async function run(){const btn=document.getElementById('run');btn.disabled=true;let obj={};for(let el of document.querySelectorAll('[data-name]')){if(el.value.trim()===''){document.getElementById('result').textContent='All 30 measurements are required.';btn.disabled=false;return;}obj[el.dataset.name]=Number(el.value)}
 try{let r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({features:obj})});let data=await r.json();if(!r.ok)throw Error(JSON.stringify(data.detail));let val=(100*data.malignant_pattern_probability).toFixed(1);document.getElementById('result').innerHTML='<div class="out"><div class="muted">Benchmark malignant-like score</div><div class="big">'+val+'%</div><p><b>Predicted pattern:</b> '+data.predicted_pattern+'</p><p class="muted">Threshold '+data.decision_threshold+' · '+data.model+'</p><p class="muted">'+data.disclaimer+'</p></div>';}catch(e){document.getElementById('result').textContent='Error: '+e.message}finally{btn.disabled=false}}
init();
</script></body></html>"""

@app.get('/',response_class=HTMLResponse)
def homepage():return HTML
