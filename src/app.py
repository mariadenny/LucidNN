import subprocess
import os

if not os.path.exists("build/app"):
    print("Building C++ backend...")
    subprocess.run("mkdir -p build", shell=True)
    subprocess.run("cd build && cmake ..", shell=True)
    subprocess.run("cd build && make", shell=True)


import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json, os, hashlib, subprocess
import plotly.graph_objects as go

st.set_page_config(page_title="LucidNN", layout="wide", page_icon="⬡")

# ──────────────────────────────────────────────────────────────
#  GLOBAL CSS  — LIGHT THEME
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');
:root{
  --bg:#F5F1EB;    --sb:#EDE8DF;   --s1:#E8E2D8;
  --s2:#DED7CA;    --s3:#D2C9B8;
  --tx:#2C1A08;    --tx2:rgba(44,26,8,.65);  --tx3:rgba(44,26,8,.38);
  --or:#C05E00;    --am:#B8860B;   --tl:#0A6B6F; --tl2:#0D8F94;
  --bd:rgba(44,26,8,.1);  --bd2:rgba(44,26,8,.18);
  --red:#B33000;
}
*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:var(--bg)!important;color:var(--tx)!important;}
.main .block-container{padding-top:1.2rem!important;padding-bottom:2rem!important;max-width:100%!important;}

/* sidebar toggle */
[data-testid="collapsedControl"],[data-testid="collapsedControl"] *{
  background:var(--s2)!important;color:var(--am)!important;border-color:var(--bd2)!important;}
[data-testid="collapsedControl"] svg{fill:var(--or)!important;stroke:var(--or)!important;}

/* sidebar */
[data-testid="stSidebar"]{background:var(--sb)!important;border-right:1px solid var(--bd2)!important;}
[data-testid="stSidebar"] *{color:var(--tx)!important;}
[data-testid="stSidebar"] [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--bd2)!important;}
[data-testid="stSidebar"] [data-baseweb="tab"]{
  background:transparent!important;font-size:11px!important;letter-spacing:.09em!important;
  text-transform:uppercase!important;color:var(--tx3)!important;padding:6px 8px!important;}
[data-testid="stSidebar"] [aria-selected="true"]{color:var(--or)!important;border-bottom:2px solid var(--or)!important;}
[data-testid="stSidebar"] [data-baseweb="tab-panel"]{background:transparent!important;}

/* inputs */
input,textarea,select,
[data-baseweb="input"],[data-baseweb="base-input"],
[data-baseweb="input"] *,[data-baseweb="base-input"] *,
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] div,[data-testid="stNumberInput"]>div,
[data-testid="stNumberInput"]>div>div{
  background:var(--s1)!important;border-color:var(--bd2)!important;
  color:var(--tx)!important;font-family:'DM Sans',sans-serif!important;}
[data-baseweb="input"]{border:1px solid var(--bd2)!important;border-radius:4px!important;background:var(--s1)!important;}
input:focus{border-color:var(--tl)!important;box-shadow:0 0 0 2px rgba(10,107,111,.15)!important;outline:none!important;}

/* steppers */
[data-testid="stNumberInput"] button{background:var(--s2)!important;border-color:var(--bd2)!important;color:var(--or)!important;}
[data-testid="stNumberInput"] button:hover{background:var(--s3)!important;}
[data-testid="stNumberInput"] button svg{fill:var(--or)!important;}

/* selectbox */
[data-baseweb="select"]>div,[data-testid="stSelectbox"]>div>div{
  background:var(--s1)!important;border:1px solid var(--bd2)!important;border-radius:4px!important;color:var(--tx)!important;}
[data-baseweb="popover"] ul,[data-baseweb="menu"]{background:var(--s1)!important;border:1px solid var(--bd2)!important;}
[data-baseweb="menu"] li{color:var(--tx)!important;background:transparent!important;}
[data-baseweb="menu"] li:hover,[data-baseweb="option"]:hover{background:var(--s2)!important;}
[data-baseweb="option"]{background:var(--s1)!important;color:var(--tx)!important;}
[aria-selected="true"][data-baseweb="option"]{background:var(--s2)!important;}

/* slider */
[data-baseweb="slider"] [role="slider"]{background:var(--or)!important;border-color:var(--or)!important;}
[data-baseweb="slider"]>div>div{background:var(--s3)!important;}

/* buttons */
.stButton>button{
  background:var(--s1)!important;border:1px solid var(--bd2)!important;border-radius:5px!important;
  color:var(--tx)!important;font-family:'DM Sans',sans-serif!important;font-size:13px!important;
  font-weight:400!important;padding:8px 16px!important;transition:all .15s!important;}
.stButton>button:hover{background:var(--s2)!important;border-color:var(--or)!important;color:var(--or)!important;}
.stButton>button[kind="primary"]{background:var(--tl)!important;border-color:var(--tl)!important;color:#fff!important;font-weight:500!important;}
.stButton>button[kind="primary"]:hover{background:var(--tl2)!important;border-color:var(--tl2)!important;color:#fff!important;}

/* main tabs */
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--bd2)!important;display:flex!important;}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;border:none!important;color:var(--tx3)!important;
  font-size:12px;font-weight:400;letter-spacing:.07em;text-transform:uppercase;
  padding:10px 0!important;flex:1!important;text-align:center!important;transition:color .15s;}
.stTabs [aria-selected="true"]{color:var(--or)!important;border-bottom:2px solid var(--or)!important;}
.stTabs [data-baseweb="tab"]:hover{color:var(--tx)!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1rem!important;background:transparent!important;}

/* metrics */
[data-testid="stMetric"]{background:var(--s1)!important;border:1px solid var(--bd2)!important;border-radius:6px!important;padding:14px 18px!important;}
[data-testid="stMetricLabel"]{font-size:10px!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:var(--tx3)!important;}
[data-testid="stMetricValue"]{font-family:'DM Serif Display',serif!important;color:var(--or)!important;font-size:26px!important;}
[data-testid="stMetricDelta"] *{font-size:12px!important;}

/* alerts */
.stSuccess>div{background:rgba(10,107,111,.1)!important;border:1px solid rgba(10,107,111,.35)!important;border-radius:4px!important;color:var(--tx)!important;}
.stError>div{background:rgba(179,48,0,.08)!important;border:1px solid rgba(179,48,0,.3)!important;border-radius:4px!important;color:var(--tx)!important;}
.stInfo>div{background:var(--s1)!important;border:1px solid var(--bd2)!important;border-radius:4px!important;color:var(--tx)!important;}
.stWarning>div{background:rgba(184,134,11,.08)!important;border:1px solid rgba(184,134,11,.3)!important;border-radius:4px!important;color:var(--tx)!important;}

/* code */
[data-testid="stCode"],.stCode{background:var(--s2)!important;border:1px solid var(--bd)!important;border-radius:4px!important;}
pre,code{background:transparent!important;color:var(--or)!important;}

/* expander */
[data-testid="stExpander"] details{border:1px solid var(--bd2)!important;border-radius:6px!important;background:var(--s1)!important;}
[data-testid="stExpander"] summary{color:var(--tx2)!important;font-size:13px!important;}

/* caption */
[data-testid="stCaptionContainer"] p{color:var(--tx3)!important;font-size:12px!important;}

h1,h2,h3,h4{font-family:'DM Serif Display',serif!important;color:var(--tx)!important;}
p,label{color:var(--tx2);}
hr{border-color:var(--bd)!important;}
div[data-testid="stVerticalBlock"]{background:transparent!important;}

::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--s1);}
::-webkit-scrollbar-thumb{background:rgba(44,26,8,.2);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:rgba(192,94,0,.4);}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────────────
DEFAULTS = {
    'layers':           [{"id":0,"neurons":3}],
    'layer_counter':    0,
    'network_data':     {},
    'results_loaded':   False,
    'results_data':     {},
    'current_epoch':    1,
    'training_inputs':  [[0.,0.],[0.,1.],[1.,0.],[1.,1.]],
    'training_targets': [[0.],[1.],[1.],[0.]],
}
for k,v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k]=v

# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────
def get_topology(inp,layers,out):
    return [inp]+[l["neurons"] for l in layers]+[out]

def init_neuron(li,ni,np_):
    k=f"L{li}_N{ni}"
    if k not in st.session_state.network_data or \
       len(st.session_state.network_data[k]['weights'])!=np_:
        st.session_state.network_data[k]={
            "bias":   float(np.random.uniform(-.5,.5)),
            "weights":[float(x) for x in np.random.uniform(-1,1,np_)]}
    return k

def init_all(topo):
    for l in range(1,len(topo)):
        for n in range(topo[l]): init_neuron(l,n,topo[l-1])

def calc_stats(topo):
    return len(topo),sum(topo),sum(topo[i]*topo[i+1] for i in range(len(topo)-1))

def to_latex_matrix(mat):
    rows = [" & ".join([f"{v:.4f}" for v in r]) for r in mat]
    return r"\begin{bmatrix}" + r" \\\\ ".join(rows) + r"\end{bmatrix}"

def reset_all():
    for k,v in DEFAULTS.items():
        st.session_state[k] = v if not isinstance(v,(list,dict)) else \
            (list(v) if isinstance(v,list) else dict(v))
    for f in ["config.json","results.json","predict_request.json","prediction.json"]:
        if os.path.exists(f): os.remove(f)

# ──────────────────────────────────────────────────────────────
#  HTML TABLE EDITOR  (light themed)
# ──────────────────────────────────────────────────────────────
def make_table_editor(x_cols, y_cols, inputs_data, targets_data):
    all_cols  = x_cols + y_cols
    rows_data = []
    for i in range(max(len(inputs_data),1)):
        row={}
        for j,c in enumerate(x_cols):
            row[c]=inputs_data[i][j] if i<len(inputs_data) and j<len(inputs_data[i]) else 0.
        for j,c in enumerate(y_cols):
            row[c]=targets_data[i][j] if i<len(targets_data) and j<len(targets_data[i]) else 0.
        rows_data.append(row)

    return f"""<!DOCTYPE html><html><head>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
html,body{{background:#EDE8DF;font-family:'DM Sans',sans-serif;color:#2C1A08;overflow:hidden;font-size:13px;}}
table{{width:100%;border-collapse:collapse;}}
thead th{{
  background:#DED7CA;color:rgba(44,26,8,.5);font-size:10px;letter-spacing:.09em;
  text-transform:uppercase;font-weight:500;padding:7px 8px;
  border-bottom:1px solid rgba(44,26,8,.15);text-align:right;}}
thead th:first-child{{text-align:left;}}
thead th.x{{color:#C05E00;}} thead th.y{{color:#0A6B6F;}}
tbody tr{{border-bottom:1px solid rgba(44,26,8,.07);}}
tbody tr:hover{{background:rgba(44,26,8,.03);}}
tbody td{{padding:3px 4px;}}
tbody td input{{
  width:100%;background:transparent;border:none;
  border-bottom:1px solid rgba(44,26,8,.12);
  color:#2C1A08;font-size:12px;padding:4px 6px;
  outline:none;font-family:'DM Sans',sans-serif;text-align:right;}}
tbody td input:focus{{border-bottom:1.5px solid #0A6B6F;color:#C05E00;}}
#bar{{display:flex;gap:5px;padding:6px 0 0;}}
.btn{{flex:1;padding:5px 0;background:#E8E2D8;border:1px solid rgba(44,26,8,.18);
  border-radius:3px;color:rgba(44,26,8,.6);font-size:11px;cursor:pointer;
  font-family:'DM Sans',sans-serif;transition:all .13s;}}
.btn:hover{{background:#DED7CA;color:#C05E00;border-color:#C05E00;}}
.pri{{background:#0A6B6F;border-color:#0A6B6F;color:#fff;}}
.pri:hover{{background:#0D8F94;border-color:#0D8F94;color:#fff;}}
</style></head><body>
<table><thead><tr id="hdr"></tr></thead><tbody id="body"></tbody></table>
<div id="bar">
  <button class="btn" onclick="addRow()">+ Row</button>
  <button class="btn" onclick="rmRow()">− Row</button>
  <button class="btn pri" id="applyBtn" onclick="send()">✓ Apply</button>
</div>
<script>
const COLS={json.dumps(all_cols)},XCOLS={json.dumps(x_cols)},YCOLS={json.dumps(y_cols)};
let rows={json.dumps(rows_data)};
const hdr=document.getElementById('hdr');
COLS.forEach(c=>{{const th=document.createElement('th');th.textContent=c;
  th.className=XCOLS.includes(c)?'x':'y';hdr.appendChild(th);}});
function render(){{
  const b=document.getElementById('body');b.innerHTML='';
  rows.forEach((row,ri)=>{{const tr=document.createElement('tr');
    COLS.forEach(col=>{{const td=document.createElement('td');
      const i=document.createElement('input');i.type='number';i.step='0.1';
      i.value=row[col]??0;i.dataset.r=ri;i.dataset.c=col;
      i.oninput=e=>{{rows[ri][col]=parseFloat(e.target.value)||0;}};
      td.appendChild(i);tr.appendChild(td);}});b.appendChild(tr);}});
}}
render();
function addRow(){{const r={{}};COLS.forEach(c=>r[c]=0);rows.push(r);render();}}
function rmRow(){{if(rows.length>1){{rows.pop();render();}}}}
function send(){{
  const inp=rows.map(r=>XCOLS.map(c=>parseFloat(r[c])||0));
  const tgt=rows.map(r=>YCOLS.map(c=>parseFloat(r[c])||0));
  window.parent.postMessage({{type:'training_data',inputs:inp,targets:tgt}},'*');
  const b=document.getElementById('applyBtn');
  b.textContent='✓ Saved';setTimeout(()=>b.textContent='✓ Apply',1200);
}}
document.getElementById('body').addEventListener('change',send);
send();
</script></body></html>"""

# ──────────────────────────────────────────────────────────────
#  CANVAS COMPONENT  — light bg, fast slider, rich popup
# ──────────────────────────────────────────────────────────────
def make_canvas(topology, init_state, history, height=440):
    topo_js       = json.dumps(topology)
    history_js    = json.dumps(history)
    init_state_js = json.dumps(init_state)
    trained       = len(history) > 0
    n_epochs      = len(history)
    _tr           = "block" if trained else "none"
    _no           = "none"  if trained else "inline"
    _epmax        = max(n_epochs, 1)
    _dis          = "" if trained else "disabled"
    _elbl         = f"Epoch 1 / {n_epochs}" if trained else "Epoch — / —"
    _mse_d        = "inline" if trained else "none"
    _noh_d        = "none"   if trained else "inline"
    _nep          = n_epochs

    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        '*{margin:0;padding:0;box-sizing:border-box;}'
        'html,body{background:#F5F1EB;font-family:DM Sans,sans-serif;color:#2C1A08;overflow-x:hidden;}'
        '#ctrl{display:flex;align-items:center;gap:10px;padding:8px 14px;height:50px;'
        'background:#EDE8DF;border-bottom:1px solid rgba(44,26,8,.12);}'
        '#pb{width:32px;height:32px;border-radius:50%;border:1.5px solid rgba(192,94,0,.5);'
        'background:transparent;color:#C05E00;font-size:15px;cursor:pointer;flex-shrink:0;'
        'display:flex;align-items:center;justify-content:center;transition:background .12s;}'
        '#pb:hover{background:rgba(192,94,0,.12);}#pb:disabled{opacity:.28;cursor:default;}'
        '#es{flex:1;-webkit-appearance:none;appearance:none;height:5px;min-width:0;outline:none;cursor:pointer;'
        'background:linear-gradient(to right,#C05E00 var(--p,0%),#D2C9B8 var(--p,0%));border-radius:3px;}'
        '#es::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;border-radius:50%;'
        'background:#C05E00;border:2px solid #F5F1EB;cursor:pointer;}'
        '#es::-moz-range-thumb{width:15px;height:15px;border-radius:50%;'
        'background:#C05E00;border:2px solid #F5F1EB;cursor:pointer;}'
        '#es:disabled{opacity:.28;cursor:default;}'
        '#el{font-size:12px;color:#C05E00;font-weight:600;white-space:nowrap;min-width:100px;}'
        '#ml{font-size:11px;color:rgba(44,26,8,.45);white-space:nowrap;}'
        '#nh{font-size:11px;color:rgba(44,26,8,.32);letter-spacing:.05em;text-transform:uppercase;}'
        '#net{display:block;background:#F5F1EB;width:100%;cursor:crosshair;}'
        f'#ch{{padding:4px 12px 16px;background:#F5F1EB;display:{_tr};}}'
        '.cR{display:flex;gap:8px;margin-top:12px;}.cC{flex:1;min-width:0;}'
        '.ct{font-size:10px;letter-spacing:.09em;text-transform:uppercase;'
        'color:rgba(44,26,8,.4);margin:0 0 4px;font-weight:500;}'
        'canvas.cc{display:block;width:100%;background:#EDE8DF;border-radius:5px;}'
        '#ns{display:block;width:100%;margin:2px 0 5px;padding:4px 7px;'
        'background:#EDE8DF;border:1px solid rgba(44,26,8,.18);border-radius:4px;'
        'color:#2C1A08;font-size:11px;outline:none;cursor:pointer;}'
        '#pop{display:none;position:fixed;z-index:9999;background:#fff;'
        'border:1px solid rgba(44,26,8,.18);border-radius:8px;padding:13px 15px;'
        'min-width:220px;max-width:280px;box-shadow:0 6px 28px rgba(44,26,8,.18);'
        'overflow-y:auto;max-height:80vh;}'
        '#pc{position:absolute;top:7px;right:9px;background:none;border:none;'
        'color:rgba(44,26,8,.35);font-size:16px;cursor:pointer;line-height:1;}'
        '#pc:hover{color:#C05E00;}'
        '#pt{font-size:9px;letter-spacing:.13em;text-transform:uppercase;color:#C05E00;margin:0 0 2px;}'
        '#pn{font-size:.92rem;color:#2C1A08;font-weight:600;margin:0 0 9px;}'
        '.fl{display:block;font-size:9px;letter-spacing:.07em;text-transform:uppercase;'
        'color:rgba(44,26,8,.4);margin-bottom:3px;}'
        '.fi{width:100%;background:#F5F1EB;border:1px solid rgba(44,26,8,.18);border-radius:3px;'
        'color:#2C1A08;font-size:12px;padding:4px 7px;outline:none;margin-bottom:7px;}'
        '.fi:focus{border-color:#0A6B6F;}'
        '#wgp{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin:4px 0 9px;}'
        '.wcp{background:#F5F1EB;border:1px solid rgba(44,26,8,.12);border-radius:3px;padding:4px 5px;}'
        '.wcl{display:block;font-size:8px;letter-spacing:.04em;text-transform:uppercase;'
        'color:rgba(44,26,8,.35);margin-bottom:2px;}'
        '.wci{width:100%;background:transparent;border:none;color:#C05E00;font-size:11px;outline:none;}'
        '#ps{width:100%;padding:6px;background:#0A6B6F;border:none;border-radius:3px;'
        'color:#fff;font-size:11px;font-weight:500;cursor:pointer;margin-bottom:3px;}'
        '#ps:hover{background:#0D8F94;}'
        '#pno{font-size:9px;color:rgba(44,26,8,.3);text-align:center;}'
        '#pst table{width:100%;border-collapse:collapse;font-size:11px;margin-top:3px;}'
        '#pst thead th{background:#F5F1EB;color:rgba(44,26,8,.42);font-size:9px;letter-spacing:.06em;'
        'text-transform:uppercase;padding:4px 5px;border-bottom:1px solid rgba(44,26,8,.12);'
        'text-align:right;font-weight:500;}'
        '#pst thead th:first-child{text-align:left;}'
        '#pst tbody td{padding:4px 5px;border-bottom:1px solid rgba(44,26,8,.05);text-align:right;font-size:11px;}'
        '#pst tbody td:first-child{text-align:left;color:rgba(44,26,8,.48);font-size:10px;}'
        '.pos{color:#0A6B6F;font-weight:600;}.neg{color:#B33000;font-weight:600;}'
        '#pbr{display:flex;justify-content:space-between;align-items:center;'
        'padding:4px 0;border-top:1px solid rgba(44,26,8,.08);margin-top:4px;}'
        '#pbl{font-size:9px;letter-spacing:.07em;text-transform:uppercase;color:rgba(44,26,8,.4);}'
        '#pbv{font-size:13px;font-weight:700;}'
        '</style></head><body>'
        '<div id="ctrl">'
        f'<button id="pb" {_dis}>▶</button>'
        f'<input id="es" type="range" min="1" max="{_epmax}" value="1" step="1" {_dis}>'
        f'<span id="el">{_elbl}</span>'
        f'<span id="ml" style="display:{_mse_d}"></span>'
        f'<span id="nh" style="display:{_noh_d}">load results to animate</span>'
        '</div>'
        f'<canvas id="net" height="{height}"></canvas>'
        '<div id="pop">'
        '<button id="pc">✕</button>'
        '<p id="pt"></p><p id="pn"></p>'
        '<div id="pre">'
        '<span class="fl">Bias</span>'
        '<input id="pbe" class="fi" type="number" step="0.001">'
        '<span class="fl">Incoming Weights</span>'
        '<div id="wgp"></div>'
        '<button id="ps">Save to network</button>'
        '<p id="pno">edits apply before training</p>'
        '</div>'
        '<div id="pst">'
        '<div id="pbr"><span id="pbl">Bias</span><span id="pbv">—</span></div>'
        '<table><thead><tr><th>Weight</th><th>Initial</th><th>Now</th><th>Δ</th>'
        '</tr></thead><tbody id="ptb"></tbody></table>'
        '</div>'
        '</div>'
        '<div id="ch">'
        '<div class="cR">'
        f'<div class="cC"><p class="ct">MSE Loss — all {_nep} epochs</p>'
        '<canvas class="cc" id="cM" height="160"></canvas></div>'
        '<div class="cC"><p class="ct">Bias Evolution</p>'
        '<canvas class="cc" id="cB" height="160"></canvas></div>'
        '</div>'
        '<div style="margin-top:12px;">'
        '<p class="ct">Weight Evolution — select neuron</p>'
        '<select id="ns"></select>'
        '<canvas class="cc" id="cW" height="160"></canvas>'
        '</div></div>'
    )

    js_data = (
        f'const TOPO={topo_js};'
        f'const H={history_js};'
        f'const S0={init_state_js};'
        f'const TR={json.dumps(trained)};'
        f'const NE={n_epochs};'
        'const CL=["#C05E00","#0A6B6F","#B8860B","#8B0000","#2F6038","#6B3FA0","#1A5276"];'
    )

    js_main = '\nconst net=document.getElementById(\'net\'),ctx=net.getContext(\'2d\');\n\n// Size the network canvas to actual rendered width\nfunction fitNet(){\n  net.width = net.parentElement.clientWidth||document.body.clientWidth||800;\n}\n\n// Size a chart canvas - must be called right before drawing (layout must be complete)\nfunction sizeChart(id){\n  const e=document.getElementById(id);\n  const w=e.parentElement.clientWidth;\n  if(w>10) e.width=w;  // only resize if we have a real width\n  return {e, x:e.getContext(\'2d\'), W:e.width, H:e.height};\n}\n\nconst sl=document.getElementById(\'es\');\nconst el=document.getElementById(\'el\');\nconst ml=document.getElementById(\'ml\');\nconst pb=document.getElementById(\'pb\');\nlet ep=1, playing=false, tick=null;\n\nfunction setEp(e){\n  ep=Math.max(1,Math.min(e,NE||1));\n  sl.value=ep;\n  const p=NE>1?((ep-1)/(NE-1)*100).toFixed(1):0;\n  sl.style.setProperty(\'--p\',p+\'%\');\n  el.textContent=TR?\'Epoch \'+ep+\' / \'+NE:\'Epoch \\u2014 / \\u2014\';\n  if(TR&&H[ep-1]) ml.textContent=\'MSE \'+H[ep-1].error.toFixed(6);\n  drawNet();\n  if(TR) drawAll();\n  if(pop.style.display!==\'none\'&&an&&TR) refreshPost(an);\n}\nsl.addEventListener(\'input\',()=>{stopP();setEp(+sl.value);});\nfunction stopP(){playing=false;pb.textContent=\'\\u25B6\';clearInterval(tick);tick=null;}\nfunction startP(){\n  if(!TR)return;\n  playing=true;pb.textContent=\'\\u23F8\';\n  if(ep>=NE)setEp(1);\n  tick=setInterval(()=>{ep<NE?setEp(ep+1):stopP();},150);\n}\npb.addEventListener(\'click\',()=>{playing?stopP():startP();});\n\n// ── Network layout + draw ─────────────────────────────────────\nfunction getNodes(){\n  const W=net.width,Hh=net.height,L=TOPO.length;\n  const px=64,xs=L>1?(W-px*2)/(L-1):0,arr=[];\n  for(let l=0;l<L;l++){\n    const cnt=TOPO[l],ys=Hh/(cnt+1);\n    for(let n=0;n<cnt;n++)\n      arr.push({id:l+\'_\'+n,l,n,x:L===1?W/2:px+l*xs,y:ys*(n+1),r:21});\n  }\n  return arr;\n}\nfunction stAt(e){\n  if(!TR||!H.length) return S0;\n  return H[Math.max(0,Math.min(e-1,NE-1))].network_state||{};\n}\nfunction maxW(st){\n  let m=0.001;\n  for(const d of Object.values(st))\n    for(const w of(d.weights||[]))\n      if(Math.abs(w)>m) m=Math.abs(w);\n  return m;\n}\nlet selN=null;\n\nfunction drawNet(){\n  const W=net.width,Hh=net.height;\n  ctx.clearRect(0,0,W,Hh);\n  ctx.fillStyle=\'#F5F1EB\';ctx.fillRect(0,0,W,Hh);\n  const nd=getNodes(),st=stAt(ep),mw=maxW(st);\n\n  // edges\n  for(let l=0;l<TOPO.length-1;l++)\n    for(let n1=0;n1<TOPO[l];n1++)\n      for(let n2=0;n2<TOPO[l+1];n2++){\n        const src=nd.find(x=>x.l===l&&x.n===n1);\n        const dst=nd.find(x=>x.l===l+1&&x.n===n2);\n        const d=st[\'L\'+(l+1)+\'_N\'+n2];\n        let color=\'rgba(44,26,8,0.08)\',lw=0.8;\n        if(d&&d.weights&&n1<d.weights.length){\n          const w=d.weights[n1],norm=Math.abs(w)/mw;\n          const a=(0.18+norm*0.82).toFixed(2);\n          if(w>=0) color=\'rgba(\'+Math.round(10+170*(1-norm))+\',\'+Math.round(107+80*(1-norm))+\',\'+Math.round(111+40*(1-norm))+\',\'+a+\')\';\n          else     color=\'rgba(\'+Math.round(192+28*norm)+\',\'+Math.round(94*(1-norm))+\',\'+Math.round(40*(1-norm))+\',\'+a+\')\';\n          lw=0.8+norm*8.5;\n        }\n        ctx.beginPath();ctx.moveTo(src.x,src.y);ctx.lineTo(dst.x,dst.y);\n        ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.stroke();\n      }\n\n  // nodes\n  for(const n of nd){\n    const iIn=n.l===0,iOut=n.l===TOPO.length-1,iSel=selN===n.id;\n    const key=\'L\'+n.l+\'_N\'+n.n;\n    const bias=(stAt(ep)[key]||S0[key]||{bias:0}).bias;\n    if(iSel){ctx.save();ctx.shadowColor=iIn?\'#C05E00\':iOut?\'#0A6B6F\':\'#C05E00\';ctx.shadowBlur=20;}\n    let fill,stroke,lw;\n    if(iIn){fill=iSel?\'#A04A00\':\'#C05E00\';stroke=iSel?\'#FF9040\':\'#7A3800\';lw=iSel?3:1.5;}\n    else if(iOut){fill=iSel?\'#0D8F94\':\'#0A6B6F\';stroke=iSel?\'#4DD4D8\':\'#064F52\';lw=iSel?3:1.5;}\n    else{fill=iSel?\'#E0D8CC\':\'#DED7CA\';stroke=iSel?\'#C05E00\':\'rgba(44,26,8,.28)\';lw=iSel?2.5:1.2;}\n    ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2);\n    ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=lw;ctx.stroke();\n    if(iSel)ctx.restore();\n    ctx.font=\'500 11px DM Sans,sans-serif\';\n    ctx.fillStyle=(iIn||iOut)?\'#fff\':(iSel?\'#C05E00\':\'#2C1A08\');\n    ctx.textAlign=\'center\';ctx.textBaseline=\'middle\';\n    ctx.fillText((iIn?\'x\':iOut?\'y\':\'h\')+n.n,n.x,n.y);\n    if(!iIn){\n      ctx.font=\'9px DM Sans,sans-serif\';ctx.fillStyle=\'rgba(44,26,8,.5)\';\n      ctx.textAlign=\'center\';ctx.textBaseline=\'top\';\n      ctx.fillText((bias>=0?\'+\':\'\')+bias.toFixed(3),n.x,n.y+n.r+5);\n    }\n  }\n  if(TR){\n    ctx.font=\'10px DM Sans,sans-serif\';ctx.fillStyle=\'rgba(44,26,8,.2)\';\n    ctx.textAlign=\'right\';ctx.textBaseline=\'top\';\n    ctx.fillText(\'Epoch \'+ep+\'/\'+NE+\'  MSE \'+(H[ep-1]?H[ep-1].error.toFixed(5):\'\'),W-10,8);\n  }\n}\n\n// ── Click popup ───────────────────────────────────────────────\nnet.addEventListener(\'click\',function(e){\n  const r=net.getBoundingClientRect();\n  const mx=(e.clientX-r.left)*(net.width/r.width);\n  const my=(e.clientY-r.top)*(net.height/r.height);\n  for(const n of getNodes()){\n    if((mx-n.x)*(mx-n.x)+(my-n.y)*(my-n.y)<=(n.r+5)*(n.r+5)){\n      selN=n.id;drawNet();openPop(n,e.clientX,e.clientY);return;\n    }\n  }\n  selN=null;drawNet();closePop();\n});\nconst pop=document.getElementById(\'pop\');\nconst preD=document.getElementById(\'pre\');\nconst postD=document.getElementById(\'pst\');\nlet an=null;\nfunction closePop(){pop.style.display=\'none\';an=null;}\ndocument.getElementById(\'pc\').onclick=()=>{closePop();selN=null;drawNet();};\nfunction openPop(n,cx,cy){\n  an=n;\n  document.getElementById(\'pt\').textContent=n.l===0?\'Input Node\':n.l===TOPO.length-1?\'Output Neuron\':\'Hidden Neuron \\u2014 Layer \'+n.l;\n  document.getElementById(\'pn\').textContent=\'Layer \'+n.l+\'  \\u00B7  Neuron \'+n.n;\n  if(TR){preD.style.display=\'none\';postD.style.display=\'block\';refreshPost(n);}\n  else{\n    postD.style.display=\'none\';\n    if(n.l===0){preD.style.display=\'none\';}\n    else{\n      preD.style.display=\'block\';\n      const key=\'L\'+n.l+\'_N\'+n.n,d=S0[key]||{bias:0,weights:[]};\n      document.getElementById(\'pbe\').value=d.bias.toFixed(4);\n      const wg=document.getElementById(\'wgp\');wg.innerHTML=\'\';\n      const prev=TOPO[n.l-1]||0;\n      for(let i=0;i<prev;i++){\n        const wv=d.weights[i]!=null?d.weights[i]:0;\n        wg.innerHTML+=\'<div class="wcp"><span class="wcl">L\'+(n.l-1)+\'\\u00B7N\'+i+\'</span>\'\n          +\'<input class="wci" type="number" step="0.001" value="\'+wv.toFixed(4)+\'" data-wi="\'+i+\'"></div>\';\n      }\n    }\n  }\n  pop.style.display=\'block\';\n  const pw=pop.offsetWidth||240,ph=pop.offsetHeight||200;\n  let px=cx+12,py=cy-10;\n  if(px+pw>window.innerWidth-6) px=cx-pw-12;\n  if(py+ph>window.innerHeight-6) py=window.innerHeight-ph-8;\n  pop.style.left=Math.max(4,px)+\'px\';\n  pop.style.top=Math.max(4,py)+\'px\';\n}\nfunction refreshPost(n){\n  if(n.l===0)return;\n  const key=\'L\'+n.l+\'_N\'+n.n;\n  const init=S0[key]||{bias:0,weights:[]};\n  const cur=stAt(ep)[key]||{bias:0,weights:[]};\n  const bd=cur.bias-init.bias;\n  const bv=document.getElementById(\'pbv\');\n  bv.textContent=(cur.bias>=0?\'+\':\'\')+cur.bias.toFixed(4)+\'  (\'+(bd>=0?\'+\':\'\')+bd.toFixed(4)+\')\';\n  bv.className=bd>=0?\'pos\':\'neg\';\n  const tb=document.getElementById(\'ptb\');tb.innerHTML=\'\';\n  const prev=TOPO[n.l-1]||0;\n  for(let i=0;i<prev;i++){\n    const w0=init.weights[i]!=null?init.weights[i]:0;\n    const w1=cur.weights[i]!=null?cur.weights[i]:0;\n    const dw=w1-w0;\n    const cls=dw>=0?\'pos\':\'neg\';\n    tb.innerHTML+=\'<tr><td>L\'+(n.l-1)+\'\\u00B7N\'+i+\'</td>\'\n      +\'<td>\'+w0.toFixed(4)+\'</td>\'\n      +\'<td class="\'+cls+\'">\'+(w1>=0?\'+\':\'\')+w1.toFixed(4)+\'</td>\'\n      +\'<td class="\'+cls+\'">\'+(dw>=0?\'+\':\'\')+dw.toFixed(4)+\'</td></tr>\';\n  }\n}\ndocument.getElementById(\'ps\').onclick=function(){\n  if(!an||an.l===0)return;\n  const key=\'L\'+an.l+\'_N\'+an.n;\n  const wInputs=document.getElementById(\'wgp\').querySelectorAll(\'input[data-wi]\');\n  S0[key]={\n    bias:parseFloat(document.getElementById(\'pbe\').value)||0,\n    weights:Array.from(wInputs).map(i=>parseFloat(i.value)||0)\n  };\n  this.textContent=\'\\u2713 Saved\';\n  setTimeout(()=>this.textContent=\'Save to network\',1400);\n  drawNet();\n};\n\n// ── Pure-canvas charts ────────────────────────────────────────\n// Drawn from scratch on every setEp() call — no state, no external deps.\n// sizeChart() measures the parent width AT DRAW TIME so we always get real pixels.\n\nconst PAD={l:46,r:14,t:14,b:32};\n\nfunction mX(v,mn,mx,W){return PAD.l+(v-mn)/(mx-mn||1)*(W-PAD.l-PAD.r);}\nfunction mY(v,mn,mx,H){return PAD.t+(mx-v)/(mx-mn||1)*(H-PAD.t-PAD.b);}\n\nfunction drawFrame(gx,W,H,xs,allYs,yLbl){\n  // allYs is array of arrays\n  const flat=[];\n  for(const ys of allYs) for(const v of ys) if(v!=null&&!isNaN(v)) flat.push(v);\n  if(!flat.length) return null;\n  let yMn=Math.min.apply(null,flat), yMx=Math.max.apply(null,flat);\n  const pd=(yMx-yMn)*0.08||0.01; yMn-=pd; yMx+=pd;\n  const xMn=xs[0], xMx=xs[xs.length-1];\n\n  gx.fillStyle=\'#EDE8DF\'; gx.fillRect(0,0,W,H);\n  gx.fillStyle=\'#F9F6F0\'; gx.fillRect(PAD.l,PAD.t,W-PAD.l-PAD.r,H-PAD.t-PAD.b);\n\n  gx.strokeStyle=\'rgba(44,26,8,.07)\'; gx.lineWidth=1;\n  gx.fillStyle=\'rgba(44,26,8,.42)\'; gx.font=\'8px DM Sans,sans-serif\';\n  for(let i=0;i<=4;i++){\n    const v=yMn+(yMx-yMn)*(i/4), py=mY(v,yMn,yMx,H);\n    gx.beginPath(); gx.moveTo(PAD.l,py); gx.lineTo(W-PAD.r,py); gx.stroke();\n    gx.textAlign=\'right\'; gx.textBaseline=\'middle\';\n    gx.fillText(v.toFixed(4),PAD.l-3,py);\n  }\n  const nt=Math.min(5,xs.length);\n  for(let i=0;i<nt;i++){\n    const xi=Math.round(i*(xs.length-1)/(nt-1||1));\n    const px=mX(xs[xi],xMn,xMx,W);\n    gx.textAlign=\'center\'; gx.textBaseline=\'top\';\n    gx.fillText(xs[xi],px,H-PAD.b+3);\n  }\n  gx.save(); gx.translate(9,H/2); gx.rotate(-Math.PI/2);\n  gx.textAlign=\'center\'; gx.textBaseline=\'middle\';\n  gx.fillText(yLbl,0,0); gx.restore();\n\n  // current epoch vertical line\n  const curX=mX(xs[Math.min(ep-1,xs.length-1)],xMn,xMx,W);\n  gx.strokeStyle=\'rgba(44,26,8,.3)\'; gx.lineWidth=1.5;\n  gx.setLineDash([3,3]);\n  gx.beginPath(); gx.moveTo(curX,PAD.t); gx.lineTo(curX,H-PAD.b); gx.stroke();\n  gx.setLineDash([]);\n\n  return {xMn,xMx,yMn,yMx};\n}\n\nfunction drawSeries(gx,xs,ys,xMn,xMx,yMn,yMx,W,H,col,lw,doFill){\n  if(doFill){\n    gx.beginPath();\n    gx.moveTo(mX(xs[0],xMn,xMx,W),H-PAD.b);\n    for(let i=0;i<xs.length;i++) gx.lineTo(mX(xs[i],xMn,xMx,W),mY(ys[i],yMn,yMx,H));\n    gx.lineTo(mX(xs[xs.length-1],xMn,xMx,W),H-PAD.b);\n    gx.closePath(); gx.fillStyle=\'rgba(192,94,0,.10)\'; gx.fill();\n  }\n  gx.beginPath(); gx.strokeStyle=col; gx.lineWidth=lw;\n  for(let i=0;i<xs.length;i++){\n    const px=mX(xs[i],xMn,xMx,W), py=mY(ys[i],yMn,yMx,H);\n    i===0?gx.moveTo(px,py):gx.lineTo(px,py);\n  }\n  gx.stroke();\n  // dot at current epoch\n  const ci=Math.min(ep-1,xs.length-1);\n  const dpx=mX(xs[ci],xMn,xMx,W), dpy=mY(ys[ci],yMn,yMx,H);\n  gx.beginPath(); gx.arc(dpx,dpy,4,0,Math.PI*2);\n  gx.fillStyle=col; gx.fill();\n  gx.strokeStyle=\'#F5F1EB\'; gx.lineWidth=1.5; gx.stroke();\n}\n\nfunction drawMse(){\n  if(!TR||!H.length)return;\n  const g=sizeChart(\'cM\');\n  const xs=H.map(function(h){return h.epoch;});\n  const ys=H.map(function(h){return h.error;});\n  const bounds=drawFrame(g.x,g.W,g.H,xs,[ys],\'MSE\');\n  if(!bounds)return;\n  drawSeries(g.x,xs,ys,bounds.xMn,bounds.xMx,bounds.yMn,bounds.yMx,g.W,g.H,\'#C05E00\',2,true);\n}\n\nfunction drawBias(){\n  if(!TR||!H.length)return;\n  const g=sizeChart(\'cB\');\n  const xs=H.map(function(h){return h.epoch;});\n  const ser=[];\n  for(let l=1;l<TOPO.length;l++){\n    for(let n=0;n<TOPO[l];n++){\n      const key=\'L\'+l+\'_N\'+n;\n      const ys=H.map(function(h){\n        if(h.network_state&&h.network_state[key]) return h.network_state[key].bias;\n        return 0;\n      });\n      ser.push({\n        label:l===TOPO.length-1?\'Out\\u00B7N\'+n:\'L\'+l+\'\\u00B7N\'+n,\n        ys:ys,\n        color:CL[(l*5+n)%CL.length]\n      });\n    }\n  }\n  const allYs=ser.map(function(s){return s.ys;});\n  const bounds=drawFrame(g.x,g.W,g.H,xs,allYs,\'Bias\');\n  if(!bounds)return;\n  ser.forEach(function(s){\n    drawSeries(g.x,xs,s.ys,bounds.xMn,bounds.xMx,bounds.yMn,bounds.yMx,g.W,g.H,s.color,1.6,false);\n  });\n  // legend\n  ser.forEach(function(s,i){\n    g.x.fillStyle=s.color; g.x.font=\'8px DM Sans,sans-serif\';\n    g.x.textAlign=\'right\'; g.x.textBaseline=\'top\';\n    g.x.fillText(s.label, g.W-PAD.r-2, PAD.t+2+i*11);\n  });\n}\n\nlet snk=null;\nfunction buildSel(){\n  const sel=document.getElementById(\'ns\');\n  sel.innerHTML=\'\';\n  for(let l=1;l<TOPO.length;l++){\n    for(let n=0;n<TOPO[l];n++){\n      const k=\'L\'+l+\'_N\'+n;\n      const lab=l===TOPO.length-1?\'Output\\u00B7N\'+n:\'Hidden L\'+l+\'\\u00B7N\'+n;\n      const opt=document.createElement(\'option\');\n      opt.value=k; opt.textContent=lab;\n      sel.appendChild(opt);\n    }\n  }\n  snk=sel.value;\n  sel.onchange=function(){snk=sel.value;drawW();};\n}\n\nfunction drawW(){\n  if(!TR||!H.length||!snk)return;\n  const g=sizeChart(\'cW\');\n  const xs=H.map(function(h){return h.epoch;});\n  const parts=snk.match(/L(\\d+)_N(\\d+)/);\n  if(!parts)return;\n  const li=parseInt(parts[1]), pc=li>0?TOPO[li-1]:0;\n  if(!pc){\n    g.x.fillStyle=\'#EDE8DF\'; g.x.fillRect(0,0,g.W,g.H);\n    g.x.fillStyle=\'rgba(44,26,8,.35)\'; g.x.font=\'11px DM Sans,sans-serif\';\n    g.x.textAlign=\'center\'; g.x.textBaseline=\'middle\';\n    g.x.fillText(\'No incoming weights\',g.W/2,g.H/2);\n    return;\n  }\n  const ser=[];\n  for(let wi=0;wi<pc;wi++){\n    const ys=H.map(function(h){\n      if(h.network_state&&h.network_state[snk]&&h.network_state[snk].weights)\n        return h.network_state[snk].weights[wi]!=null?h.network_state[snk].weights[wi]:0;\n      return 0;\n    });\n    ser.push({label:\'W\'+wi+\'\\u2190L\'+(li-1)+\'\\u00B7N\'+wi, ys:ys, color:CL[wi%CL.length]});\n  }\n  const allYs=ser.map(function(s){return s.ys;});\n  const bounds=drawFrame(g.x,g.W,g.H,xs,allYs,\'Weight\');\n  if(!bounds)return;\n  ser.forEach(function(s){\n    drawSeries(g.x,xs,s.ys,bounds.xMn,bounds.xMx,bounds.yMn,bounds.yMx,g.W,g.H,s.color,1.6,false);\n  });\n  ser.forEach(function(s,i){\n    g.x.fillStyle=s.color; g.x.font=\'8px DM Sans,sans-serif\';\n    g.x.textAlign=\'right\'; g.x.textBaseline=\'top\';\n    g.x.fillText(s.label, g.W-PAD.r-2, PAD.t+2+i*11);\n  });\n}\n\nfunction drawAll(){drawMse();drawBias();drawW();}\n\n// ── Init: use requestAnimationFrame so layout is fully painted ─\n// This is the key fix: waiting one frame ensures all element\n// clientWidths are non-zero before we try to draw.\nrequestAnimationFrame(function(){\n  fitNet();\n  if(TR){buildSel();setEp(1);}\n  else{drawNet();}\n});\nwindow.addEventListener(\'resize\',function(){fitNet();drawNet();if(TR)drawAll();});\n'

    return html + "<script>" + js_data + js_main + "</script></body></html>"



# ──────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 .8rem;">
      <p style="font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:#C05E00;margin:0 0 3px;font-weight:500;">Configuration</p>
      <h2 style="font-family:'DM Serif Display',serif;font-size:1.25rem;color:#2C1A08;margin:0;">Model Setup</h2>
    </div>""", unsafe_allow_html=True)

    ta,th,td = st.tabs(["Architecture","Hyperparams","Data"])

    with ta:
        st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)
        input_nodes  = st.number_input("Input Nodes",  min_value=1,max_value=5,value=2,step=1)
        output_nodes = st.number_input("Output Nodes", min_value=1,max_value=5,value=1,step=1)
        st.markdown("<hr style='margin:10px 0;border-color:rgba(44,26,8,.1);'/>",unsafe_allow_html=True)
        st.markdown("<p style='font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:rgba(44,26,8,.36);margin:0 0 6px;'>Hidden Layers</p>",unsafe_allow_html=True)
        if st.button("+ Add Hidden Layer",use_container_width=True,
                     disabled=len(st.session_state.layers)>=5):
            st.session_state.layer_counter+=1
            st.session_state.layers.append({"id":st.session_state.layer_counter,"neurons":3})
            st.rerun()
        to_rm=[]
        for i,layer in enumerate(st.session_state.layers):
            c1,c2=st.columns([4,1])
            with c1:
                st.session_state.layers[i]['neurons']=st.number_input(
                    f"Layer {i+1} Neurons",min_value=1,max_value=5,
                    value=min(layer['neurons'],5),step=1,key=f"ln_{layer['id']}")
            with c2:
                st.markdown("<div style='height:24px'></div>",unsafe_allow_html=True)
                if st.button("✕",key=f"del_{layer['id']}"): to_rm.append(i)
        if to_rm:
            for idx in sorted(to_rm,reverse=True): del st.session_state.layers[idx]
            st.rerun()

    with th:
        st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)
        activation = st.selectbox("Activation",["ReLU","Sigmoid","Tanh","Linear","Leaky ReLU"])
        st.selectbox("Loss Function",["Mean Squared Error (MSE)"])
        st.markdown("<hr style='margin:10px 0;border-color:rgba(44,26,8,.1);'/>",unsafe_allow_html=True)
        epochs_s = st.slider("Epochs",min_value=10,max_value=5000,step=10,value=100)
        lr = st.number_input("Learning Rate",min_value=0.0001,max_value=1.0,
                              value=0.01,step=0.001,format="%.4f")

    with td:
        xc=[f"X{i}" for i in range(input_nodes)]
        yc=[f"Y{i}" for i in range(output_nodes)]
        tk=f"{input_nodes}x{output_nodes}"
        if st.session_state.get('_tk')!=tk:
            st.session_state['_tk']=tk
            if input_nodes==2 and output_nodes==1:
                st.session_state.training_inputs  = [[0.,0.],[0.,1.],[1.,0.],[1.,1.]]
                st.session_state.training_targets = [[0.],[1.],[1.],[0.]]
            else:
                st.session_state.training_inputs  = [[0.]*input_nodes  for _ in range(4)]
                st.session_state.training_targets = [[0.]*output_nodes for _ in range(4)]
        rc=len(st.session_state.training_inputs)
        th_=max(140,48+rc*34+42)
        components.html(make_table_editor(xc,yc,
            st.session_state.training_inputs,
            st.session_state.training_targets),
            height=th_,scrolling=False)
        u_inp=st.session_state.training_inputs
        u_tgt=st.session_state.training_targets

    # ── ACTIONS ──────────────────────────────────────────────
    st.markdown("<hr style='margin:12px 0;border-color:rgba(44,26,8,.1);'/>",unsafe_allow_html=True)
    st.markdown("<p style='font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:rgba(44,26,8,.32);margin:0 0 7px;'>Actions</p>",unsafe_allow_html=True)

    topology=get_topology(input_nodes,st.session_state.layers,output_nodes)
    init_all(topology)

    # Action 1: Train the model directly via subprocess
    if st.button("⟳  Train Model", type="primary", use_container_width=True):
        if os.path.exists("results.json"): os.remove("results.json")
        hcfg=[{"neurons":l["neurons"],"activation":activation.lower()} for l in st.session_state.layers]
        vk=[f"L{l}_N{n}" for l in range(1,len(topology)) for n in range(topology[l])]
        cfg={
            "type":"INIT_NETWORK",
            "network":{"input_size":input_nodes,"hidden_layers":hcfg,
                       "output_layer":{"neurons":output_nodes,"activation":activation.lower()}},
            "hyperparameters":{"epochs":epochs_s,"learning_rate":lr},
            "training_data":{"inputs":u_inp,"targets":u_tgt},
            "initial_state":{k:v for k,v in st.session_state.network_data.items() if k in vk}
        }
        try:
            with open("config.json","w") as f: json.dump(cfg,f,indent=4)
            
            with st.spinner("Training model in backend..."):
                result = subprocess.run(["./build/app", "config.json"], capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists("results.json"):
                    with open("results.json") as f:
                        st.session_state.results_data = json.load(f)
                    if st.session_state.results_data.get("status") == "success":
                        st.session_state.results_loaded = True
                        st.session_state.current_epoch = len(st.session_state.results_data["history"])
                        st.success(f"Training complete! Loaded {st.session_state.current_epoch} epochs.")
                        st.rerun()
                    else:
                        st.error("Training completed, but 'status' was not success in results.json.")
                else:
                    st.error(f"Backend execution failed. Return code: {result.returncode}\n\nError output:\n{result.stderr}")
        except Exception as e: 
            st.error(f"Execution Error: {str(e)}. Make sure ./build/app exists and is executable.")

    st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)

    # Action 2: Train New Network (full reset)
    if st.button("✦  Train New Network",use_container_width=True):
        reset_all()
        st.rerun()

# ──────────────────────────────────────────────────────────────
#  RESOLVE STATE
# ──────────────────────────────────────────────────────────────
topology  = get_topology(input_nodes,st.session_state.layers,output_nodes)
init_all(topology)
history   = st.session_state.results_data.get("history",[]) if st.session_state.results_loaded else []
max_epoch = max(len(history),1)
cur_epoch = max(1,min(st.session_state.current_epoch,max_epoch))
step_data = history[cur_epoch-1] if history else None

# ──────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:.4rem 0 .9rem;border-bottom:1px solid rgba(44,26,8,.1);margin-bottom:.9rem;
     display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;">
  <span style="font-family:'DM Serif Display',serif;font-size:2.1rem;color:#2C1A08;letter-spacing:-.02em;line-height:1;">LucidNN</span>
  <span style="font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:#C05E00;font-weight:500;
               padding:3px 9px;border:1px solid #C05E00;border-radius:2px;">v1.0</span>
  <span style="font-size:12px;color:rgba(44,26,8,.3);letter-spacing:.04em;text-transform:uppercase;">
        Neural Network Designer &amp; Educational Visualizer</span>
</div>""",unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  TABS
# ──────────────────────────────────────────────────────────────
t1,t2,t3,t4 = st.tabs(["Network Diagram","Training Results","Matrix Math","Predictions"])

# ════════════════════════════════════════════════════════════
#  TAB 1 — NETWORK DIAGRAM
# ════════════════════════════════════════════════════════════
with t1:
    init_s = {k:{"bias":v["bias"],"weights":list(v["weights"])}
              for k,v in st.session_state.network_data.items()}
    # Height: ctrl(50) + canvas(440) + charts if trained(700) + buffer(20)
    component_h = 50 + 440 + (700 if history else 0) + 20
    # Cache-buster: Streamlit hashes the HTML string to decide whether to reuse
    # the existing iframe or rebuild it. Without a unique token, reloading the
    # same results.json returns the frozen cached iframe and JS never re-executes.
    _bust = hashlib.md5(f"{len(history)}-{topology}-{id(st.session_state.results_data)}-{st.session_state.results_loaded}".encode()).hexdigest()[:8]
    components.html(
        make_canvas(topology, init_s, history, height=440) + f"",
        height=component_h,
        scrolling=False,  # MUST be False: True puts canvas in scroll container, breaks width detection
    )

    st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
    tl,tn,tc = calc_stats(topology)
    m1,m2,m3 = st.columns(3)
    m1.metric("Total Layers", tl)
    m2.metric("Neurons",      tn)
    m3.metric("Connections",  tc)

    st.markdown("""
    <div style="display:flex;gap:16px;justify-content:center;margin:10px 0 2px;flex-wrap:wrap;">
      <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:rgba(44,26,8,.4);">
        <span style="width:10px;height:10px;border-radius:50%;background:#C05E00;display:inline-block;"></span>Input</span>
      <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:rgba(44,26,8,.4);">
        <span style="width:10px;height:10px;border-radius:50%;background:#DED7CA;border:1px solid rgba(44,26,8,.3);display:inline-block;"></span>Hidden</span>
      <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:rgba(44,26,8,.4);">
        <span style="width:10px;height:10px;border-radius:50%;background:#0A6B6F;display:inline-block;"></span>Output</span>
      <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:rgba(44,26,8,.4);">
        <span style="width:20px;height:2px;background:#0A6B6F;display:inline-block;"></span>+weight</span>
      <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:rgba(44,26,8,.4);">
        <span style="width:20px;height:2px;background:#C05E00;display:inline-block;"></span>−weight · thickness=|w|</span>
    </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TAB 2 — TRAINING RESULTS
# ════════════════════════════════════════════════════════════
with t2:
    if not history:
        st.markdown("""
        <div style="background:#E8E2D8;border:1px solid rgba(44,26,8,.1);border-radius:8px;
             padding:40px;text-align:center;margin-top:16px;">
          <p style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#C05E00;margin:0 0 8px;">No Data</p>
          <h3 style="font-family:'DM Serif Display',serif;color:#2C1A08;margin:0 0 8px;">Awaiting Training Results</h3>
          <p style="color:rgba(44,26,8,.4);font-size:13px;margin:0;">Click 'Train Model' in the sidebar to start.</p>
        </div>""",unsafe_allow_html=True)
    else:
        last = history[-1]
        total_epochs = len(history)

        # ── MSE loss curve — full history, all epochs ──
        st.markdown(f"<p style='font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:rgba(44,26,8,.32);margin:0 0 6px;'>MSE Loss over {total_epochs} Epochs</p>",unsafe_allow_html=True)
        fig = go.Figure()
        epochs_x = [s["epoch"] for s in history]
        errors_y = [s["error"] for s in history]
        fig.add_trace(go.Scatter(
            x=epochs_x, y=errors_y, mode='lines',
            line=dict(color='#C05E00', width=2.5),
            fill='tozeroy', fillcolor='rgba(192,94,0,.08)',
            name='MSE Loss',
            hovertemplate='Epoch %{x}<br>MSE: %{y:.6f}<extra></extra>'))
        # start and end markers
        fig.add_trace(go.Scatter(
            x=[epochs_x[0], epochs_x[-1]],
            y=[errors_y[0], errors_y[-1]],
            mode='markers+text',
            marker=dict(color=['#B8860B','#0A6B6F'], size=10, line=dict(color='#F5F1EB',width=2)),
            text=[f"Start<br>{errors_y[0]:.4f}", f"Final<br>{errors_y[-1]:.4f}"],
            textposition=["top right","top left"],
            textfont=dict(size=10, color='rgba(44,26,8,.55)'),
            showlegend=False, hoverinfo='skip'))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Sans', color='rgba(44,26,8,.5)', size=11),
            xaxis=dict(title='Epoch', gridcolor='rgba(44,26,8,.07)', zerolinecolor='rgba(44,26,8,.1)',
                       tickfont=dict(size=10)),
            yaxis=dict(title='MSE Loss', gridcolor='rgba(44,26,8,.07)', zerolinecolor='rgba(44,26,8,.1)',
                       tickfont=dict(size=10)),
            margin=dict(l=0,r=0,t=10,b=0), height=280,
            hovermode='x unified',
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

        # ── Summary metrics ──
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Epochs",   total_epochs)
        c2.metric("Initial MSE",    f"{errors_y[0]:.5f}")
        c3.metric("Final MSE",      f"{errors_y[-1]:.5f}")
        improvement = ((errors_y[0]-errors_y[-1])/errors_y[0]*100) if errors_y[0]>0 else 0
        c4.metric("Improvement",    f"{improvement:.1f}%")

        st.markdown("<hr style='margin:16px 0 12px;border-color:rgba(44,26,8,.08);'/>",unsafe_allow_html=True)

        # ── Bias evolution — one chart per layer, all neurons overlaid ──
        st.markdown("<p style='font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:rgba(44,26,8,.32);margin:0 0 8px;'>Bias Evolution per Layer</p>",unsafe_allow_html=True)
        colors_list = ['#C05E00','#0A6B6F','#B8860B','#8B0000','#2F6038','#6B3FA0','#1A5276']
        for li in range(1, len(topology)):
            lname = "Output Layer" if li==len(topology)-1 else f"Hidden Layer {li}"
            fig2 = go.Figure()
            has_data = False
            for ni in range(topology[li]):
                key = f"L{li}_N{ni}"
                bv = [h["network_state"][key]["bias"]
                      for h in history if key in h.get("network_state",{})]
                if bv:
                    has_data = True
                    fig2.add_trace(go.Scatter(
                        x=list(range(1, len(bv)+1)), y=bv, mode='lines',
                        line=dict(color=colors_list[ni % len(colors_list)], width=2),
                        name=f"Neuron {ni}",
                        hovertemplate=f'Neuron {ni}<br>Epoch %{{x}}<br>Bias: %{{y:.5f}}<extra></extra>'))
            if has_data:
                fig2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='DM Sans', color='rgba(44,26,8,.45)', size=10),
                    xaxis=dict(title='Epoch', gridcolor='rgba(44,26,8,.06)', tickfont=dict(size=9)),
                    yaxis=dict(title='Bias', gridcolor='rgba(44,26,8,.06)', tickfont=dict(size=9)),
                    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='rgba(44,26,8,.6)', size=10),
                                orientation='h', y=1.1),
                    title=dict(text=lname, font=dict(family='DM Serif Display',color='#2C1A08',size=13), x=0),
                    margin=dict(l=0,r=0,t=28,b=0), height=200, hovermode='x unified')
                st.plotly_chart(fig2, use_container_width=True)

        # ── Weight evolution — pick a neuron ──
        st.markdown("<hr style='margin:14px 0 12px;border-color:rgba(44,26,8,.08);'/>",unsafe_allow_html=True)
        st.markdown("<p style='font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:rgba(44,26,8,.32);margin:0 0 8px;'>Weight Evolution — Select Neuron</p>",unsafe_allow_html=True)
        all_neurons = [f"L{l}_N{n}" for l in range(1,len(topology)) for n in range(topology[l])]
        sel_neuron = st.selectbox("Neuron", all_neurons,
            format_func=lambda k: f"Layer {k.split('_')[0][1:]} · Neuron {k.split('_')[1][1:]}",
            label_visibility="collapsed")
        if sel_neuron:
            l_idx = int(sel_neuron.split("_")[0][1:])
            n_idx = int(sel_neuron.split("_")[1][1:])
            prev_count = topology[l_idx-1] if l_idx > 0 else 0
            fig3 = go.Figure()
            for wi in range(prev_count):
                wv = [h["network_state"].get(sel_neuron, {}).get("weights", [])[wi]
                      for h in history
                      if sel_neuron in h.get("network_state",{})
                      and wi < len(h["network_state"][sel_neuron].get("weights",[]))]
                if wv:
                    fig3.add_trace(go.Scatter(
                        x=list(range(1,len(wv)+1)), y=wv, mode='lines',
                        line=dict(color=colors_list[wi % len(colors_list)], width=1.8),
                        name=f"W from L{l_idx-1}·N{wi}",
                        hovertemplate=f'W{wi}<br>Epoch %{{x}}<br>Value: %{{y:.5f}}<extra></extra>'))
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', color='rgba(44,26,8,.45)', size=10),
                xaxis=dict(title='Epoch', gridcolor='rgba(44,26,8,.06)', tickfont=dict(size=9)),
                yaxis=dict(title='Weight Value', gridcolor='rgba(44,26,8,.06)', tickfont=dict(size=9)),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='rgba(44,26,8,.6)', size=10),
                            orientation='h', y=1.12),
                margin=dict(l=0,r=0,t=10,b=0), height=220, hovermode='x unified')
            st.plotly_chart(fig3, use_container_width=True)

        # ── Last epoch prediction vs target ──
        st.markdown("<hr style='margin:10px 0 10px;border-color:rgba(44,26,8,.08);'/>",unsafe_allow_html=True)
        st.markdown("<p style='font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:rgba(44,26,8,.32);margin:0 0 8px;'>Final Epoch — Last Sample</p>",unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.metric("Final Epoch", last["epoch"])
            st.metric("Final MSE",   f"{last['error']:.6f}")
        with fc2:
            st.markdown("<p style='font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:rgba(44,26,8,.32);margin:0 0 4px;'>Prediction</p>",unsafe_allow_html=True)
            st.code(str([round(x,4) for x in last.get("actual_output",[])]))
        with fc3:
            st.markdown("<p style='font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:rgba(44,26,8,.32);margin:0 0 4px;'>Target</p>",unsafe_allow_html=True)
            st.code(str([round(x,4) for x in last.get("expected_output",[])]))

# ════════════════════════════════════════════════════════════
#  TAB 3 — MATRIX MATH
# ════════════════════════════════════════════════════════════

with t3:
    if not st.session_state.get("results_loaded", False) or not step_data:
        st.markdown("""
        <div style="background:#E8E2D8;border:1px solid rgba(44,26,8,.1);border-radius:8px;
             padding:40px;text-align:center;margin-top:16px;">
          <p style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#C05E00;margin:0 0 8px;">No Data</p>
          <h3 style="font-family:'DM Serif Display',serif;color:#2C1A08;margin:0 0 8px;">Awaiting Training</h3>
          <p style="color:rgba(44,26,8,.4);font-size:13px;margin:0;">Train the model and load results first.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div style="margin-bottom:14px;">
          <p style="font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:rgba(44,26,8,.3);margin:0 0 3px;">Mathematical Transformations</p>
          <h2 style="font-family:'DM Serif Display',serif;color:#2C1A08;margin:0;">Epoch {cur_epoch}</h2>
          <p style="color:rgba(44,26,8,.38);font-size:12px;margin:3px 0 0;">Matrix math for the last processed sample</p>
        </div>
        """, unsafe_allow_html=True)

        math_details = step_data.get("math_details", {})

        if not math_details:
            st.info("No math_details found in results.json for this epoch.")
        else:
            for l in range(1, len(topology)):

                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin:14px 0 8px;">
                  <span style="width:6px;height:6px;border-radius:50%;background:#C05E00;display:inline-block;"></span>
                  <p style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#C05E00;margin:0;font-weight:500;">Layer {l}</p>
                </div>
                """, unsafe_allow_html=True)

                layer_str = f"Layer_{l}"
                prev_layer_str = f"Layer_{l-1}"

                if layer_str in math_details and prev_layer_str in math_details:

                    W = to_latex_matrix(math_details[layer_str]["W"])
                    A_prev = to_latex_matrix(math_details[prev_layer_str]["A"])
                    B = to_latex_matrix(math_details[layer_str]["B"])
                    Z = to_latex_matrix(math_details[layer_str]["Z"])
                    A = to_latex_matrix(math_details[layer_str]["A"])
                    Delta = to_latex_matrix(math_details[layer_str]["Delta"])

                    st.markdown("**1 · Weighted Sum**")
                    st.latex(rf"Z^{{({l})}} = W^{{({l})}} \cdot A^{{({l-1})}} + B^{{({l})}}")
                    st.latex(rf"{Z} = {W} \cdot {A_prev} + {B}")

                    st.markdown("**2 · Activation**")
                    st.latex(rf"A^{{({l})}} = \sigma(Z^{{({l})}}) = {A}")

                    st.markdown("**3 · Backprop Delta**")
                    st.latex(rf"\delta^{{({l})}} = {Delta}")

                    st.markdown("<hr>", unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════
#  TAB 4 — PREDICTIONS
# ════════════════════════════════════════════════════════════
with t4:
    st.markdown("""
    <div style="margin-bottom:14px;">
      <p style="font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:rgba(44,26,8,.3);margin:0 0 3px;">Inference</p>
      <h2 style="font-family:'DM Serif Display',serif;color:#2C1A08;margin:0;">Run Prediction</h2>
      <p style="color:rgba(44,26,8,.38);font-size:13px;margin:4px 0 0;">Feed unseen data to your trained model</p>
    </div>""",unsafe_allow_html=True)
    pc=st.columns(min(input_nodes,4))
    preds=[]
    for i in range(input_nodes):
        with pc[i%len(pc)]:
            preds.append(st.number_input(f"Feature X{i}",value=0.0,step=0.1,key=f"pi_{i}"))
    st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
    
    # Combined prediction into a single button
    if st.button("Run Prediction", type="primary", use_container_width=True):
        if os.path.exists("prediction.json"): os.remove("prediction.json")
        if "pr" in st.session_state: del st.session_state["pr"]
        try:
            with open("predict_request.json","w") as f:
                json.dump({"type":"PREDICT","input":preds},f,indent=4)
            
            with st.spinner("Running inference via backend..."):
                result = subprocess.run(["./build/app", "predict_request.json"], capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists("prediction.json"):
                    with open("prediction.json") as f: 
                        res = json.load(f)
                    if res.get("status") == "success": 
                        st.session_state["pr"] = res
                        st.rerun()
                    else: 
                        st.error("Prediction failed inside backend logic.")
                else: 
                    st.error(f"Backend execution failed. Return code: {result.returncode}\n\nError output:\n{result.stderr}")
        except Exception as e: 
            st.error(f"Execution Error: {str(e)}")
            
    if "pr" in st.session_state:
        r=st.session_state["pr"]
        st.markdown("""
        <div style="background:rgba(10,107,111,.08);border:1px solid rgba(10,107,111,.25);
             border-radius:6px;padding:14px 18px;margin-top:10px;">
          <p style="font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:#0A6B6F;margin:0 0 5px;">Model Output</p>""",
        unsafe_allow_html=True)
        st.code(str([round(x,6) for x in r["prediction"]]))
        st.markdown("</div>",unsafe_allow_html=True)
        if st.button("Clear Result"):
            del st.session_state["pr"]
            if os.path.exists("prediction.json"): os.remove("prediction.json")
            st.rerun()
