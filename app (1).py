from __future__ import annotations
import io
from datetime import datetime
from typing import Any
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageDraw
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title='ROI Studio', page_icon='✦', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:#0b0d10;color:#f4f4f5}
[data-testid="stHeader"]{background:#0b0d10}[data-testid="stSidebar"]{background:#101318;border-right:1px solid #252a31}
.block-container{max-width:1500px;padding-top:1rem}.brand{display:flex;gap:12px;align-items:center;margin-bottom:16px}
.mark{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,#fff,#8b949e);color:#101216;display:flex;align-items:center;justify-content:center;font-weight:800}
.title{font-size:18px;font-weight:700}.sub{font-size:11px;color:#89929e}.hero{display:flex;justify-content:space-between;align-items:end;padding:5px 0 18px}.hero h1{margin:0;font-size:29px;letter-spacing:-1px}.hero p{margin:5px 0 0;color:#89929e;font-size:13px}
.panel{background:#11151a;border:1px solid #252a31;border-radius:16px;padding:16px}.section{color:#9aa3ae;text-transform:uppercase;letter-spacing:1.2px;font-size:10px;font-weight:700;margin:8px 0 10px}.badge{padding:4px 8px;border:1px solid #30363f;border-radius:99px;background:#171b21;color:#aeb6c1;font-size:10px;font-weight:600}
div.stButton>button{border-radius:9px;border:1px solid #30363f;background:#171b21;color:#f4f4f5;font-weight:600;min-height:38px}div.stButton>button:hover{border-color:#697384}div.stDownloadButton>button{width:100%;border-radius:10px;background:#f1f1f1;color:#101216;font-weight:700;border:0}
[data-testid="stFileUploaderDropzone"]{background:#11151a;border:1px dashed #3b424c;border-radius:12px}[data-baseweb="tab-list"]{gap:8px}button[data-baseweb="tab"]{color:#89929e!important;border-radius:8px}button[data-baseweb="tab"][aria-selected="true"]{color:#fff!important;background:#1a1f26!important}
.small{color:#7f8995;font-size:11px;line-height:1.55}.stMetric{background:#11151a;border:1px solid #252a31;padding:10px;border-radius:12px}
</style>''', unsafe_allow_html=True)

FILTERS=['Original','Gaussian Blur','Median Filter','Sharpening','Edge Detection','Grayscale','Detail Boost','Soft Glow']
PRESETS={'Clean':(0,8,4,0,10),'Cinematic':(-2,18,-8,5,8),'Vivid':(3,12,22,2,15),'Soft Portrait':(4,-7,4,3,-5),'B&W Classic':(2,12,-100,0,12)}

def pil(arr): return Image.fromarray(np.clip(arr,0,255).astype(np.uint8),'RGB')
def arr(img): return np.array(img.convert('RGB'))
def preview(img,maxw=1000):
    p=img.copy();
    if p.width>maxw: p.thumbnail((maxw,10000),Image.Resampling.LANCZOS)
    return p

def global_edit(rgb,exposure,contrast,saturation,warmth,sharpness):
    x=np.clip(np.array(rgb,dtype=np.float32)*(2**(exposure/50)),0,255).astype(np.uint8); im=pil(x)
    im=ImageEnhance.Contrast(im).enhance(max(0,1+contrast/100)); im=ImageEnhance.Color(im).enhance(max(0,1+saturation/100))
    if warmth:
        a=np.array(im,dtype=np.float32); a[:,:,0]+=warmth*.55; a[:,:,2]-=warmth*.55; im=pil(a)
    if sharpness: im=ImageEnhance.Sharpness(im).enhance(max(0,1+sharpness/100))
    return arr(im)

def roi_filter(crop,name):
    if name=='Original': return crop.copy()
    if name=='Gaussian Blur': return cv2.GaussianBlur(crop,(11,11),0)
    if name=='Median Filter':
        m=min(crop.shape[:2]); k=min(7,m if m%2 else m-1); return crop.copy() if k<3 else cv2.medianBlur(crop,k)
    if name=='Sharpening': return cv2.filter2D(crop,-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32))
    if name=='Edge Detection': return cv2.cvtColor(cv2.Canny(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),60,150),cv2.COLOR_GRAY2RGB)
    if name=='Grayscale': return cv2.cvtColor(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
    if name=='Detail Boost':
        b=cv2.GaussianBlur(crop,(0,0),2); return cv2.addWeighted(crop,1.45,b,-.45,0)
    if name=='Soft Glow':
        b=cv2.GaussianBlur(crop,(0,0),7); return cv2.addWeighted(crop,.72,b,.28,0)
    return crop.copy()

def render(base,rois):
    out=base.copy(); h,w=out.shape[:2]
    for r in rois:
        name=r.get('filter')
        if not name or name=='Original': continue
        x1=max(0,min(w,int(r['x']))); y1=max(0,min(h,int(r['y']))); x2=max(0,min(w,int(r['x']+r['w']))); y2=max(0,min(h,int(r['y']+r['h'])))
        if x2>x1 and y2>y1: out[y1:y2,x1:x2]=roi_filter(out[y1:y2,x1:x2].copy(),name)
    return out

def overlay(image,rois):
    p=preview(pil(image)); sx=p.width/image.shape[1]; sy=p.height/image.shape[0]; d=ImageDraw.Draw(p)
    for i,r in enumerate(rois,1):
        x1,y1=int(r['x']*sx),int(r['y']*sy); x2,y2=int((r['x']+r['w'])*sx),int((r['y']+r['h'])*sy); c='#70e1b5' if r.get('filter') else '#f2c94c'
        d.rectangle((x1,y1,x2,y2),outline=c,width=3); d.text((x1+6,y1+5),f"ROI {i} · {r.get('filter') or 'None'}",fill=c)
    return p

def reset():
    for k in ['rois','canvas_ver','preset','exposure','contrast','saturation','warmth','sharpness','image_id']: st.session_state.pop(k,None)
    st.rerun()

for k,v in {'rois':[],'canvas_ver':0,'preset':'Custom','exposure':0,'contrast':0,'saturation':0,'warmth':0,'sharpness':0}.items(): st.session_state.setdefault(k,v)

st.markdown('<div class="brand"><div class="mark">✦</div><div><div class="title">ROI Studio</div><div class="sub">Precision image editing · OpenCV powered</div></div></div>',unsafe_allow_html=True)
upload=st.file_uploader('Import image',type=['jpg','jpeg','png','webp','bmp','tif','tiff'],label_visibility='collapsed')
if upload is None:
    st.markdown('<div class="hero"><div><h1>Professional ROI image editing</h1><p>Fine-tune the full frame, then style individual regions with precision.</p></div><span class="badge">PRO EDITOR</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="panel" style="text-align:center;padding:85px 20px"><div style="font-size:46px">◎</div><h3>Drop an image to begin</h3><div class="small">JPG · PNG · WEBP · BMP · TIFF</div></div>',unsafe_allow_html=True); st.stop()

try: source=Image.open(upload).convert('RGB')
except Exception as e: st.error(f'Unable to read image: {e}'); st.stop()
original=arr(source); h,w=original.shape[:2]; image_id=f'{upload.name}:{upload.size}'
if st.session_state.get('image_id')!=image_id: st.session_state.image_id=image_id; st.session_state.rois=[]; st.session_state.canvas_ver+=1

with st.sidebar:
    st.markdown('### Adjust'); st.caption('Global controls affect the complete image.')
    names=['Custom']+list(PRESETS); p=st.selectbox('Preset',names,key='preset')
    if p!='Custom' and st.session_state.get('_applied_preset')!=p:
        vals=PRESETS[p]
        for k,v in zip(['exposure','contrast','saturation','warmth','sharpness'],vals): st.session_state[k]=v
        st.session_state['_applied_preset']=p; st.rerun()
    st.slider('Exposure',-50,50,key='exposure'); st.slider('Contrast',-50,50,key='contrast'); st.slider('Saturation',-100,100,key='saturation'); st.slider('Warmth',-30,30,key='warmth'); st.slider('Sharpness',-30,60,key='sharpness')
    st.divider(); c1,c2=st.columns(2)
    if c1.button('↺ Reset',use_container_width=True): reset()
    if c2.button('Clear ROI',use_container_width=True): st.session_state.rois=[]; st.session_state.canvas_ver+=1; st.rerun()
    st.divider(); st.markdown('### Export'); st.markdown('<div class="small">PNG export keeps the original resolution. ROI labels are never included.</div>',unsafe_allow_html=True)

adjusted=global_edit(original,st.session_state.exposure,st.session_state.contrast,st.session_state.saturation,st.session_state.warmth,st.session_state.sharpness); result=render(adjusted,st.session_state.rois)
st.markdown(f'<div class="hero"><div><h1>{upload.name}</h1><p>{w:,} × {h:,} px · {len(st.session_state.rois)} ROI(s) · non-destructive workflow</p></div><span class="badge">{p.upper()}</span></div>',unsafe_allow_html=True)
a,b,c,d=st.columns(4); a.metric('Resolution',f'{w:,} × {h:,}'); b.metric('ROIs',len(st.session_state.rois)); c.metric('Active filters',sum(bool(r.get('filter')) and r.get('filter')!='Original' for r in st.session_state.rois)); d.metric('Color','RGB / 8-bit')

edit,compare,regions,export=st.tabs(['✦ EDIT','◐ BEFORE / AFTER','⌖ ROIs','↓ EXPORT'])
with edit:
    st.markdown('<div class="section">ROI selection canvas</div>',unsafe_allow_html=True); st.caption('Draw rectangles, add them to the ROI stack, then assign a filter to each region.')
    bg=preview(pil(adjusted),1050); canvas=st_canvas(fill_color='rgba(112,225,181,.16)',stroke_width=2,stroke_color='#70e1b5',background_image=bg,update_streamlit=True,height=bg.height,width=bg.width,drawing_mode='rect',key=f'canvas_{st.session_state.canvas_ver}')
    x1,x2,x3=st.columns([1.3,1,1])
    if x1.button('＋ Add drawn rectangles',type='primary',use_container_width=True):
        objs=(canvas.json_data or {}).get('objects',[]); sx=w/bg.width; sy=h/bg.height; added=0
        for o in objs:
            if o.get('type')!='rect': continue
            left=float(o.get('left',0)); top=float(o.get('top',0)); ow=float(o.get('width',0))*float(o.get('scaleX',1)); oh=float(o.get('height',0))*float(o.get('scaleY',1)); xx=max(0,min(w,round(left*sx))); yy=max(0,min(h,round(top*sy))); xend=max(0,min(w,round((left+ow)*sx))); yend=max(0,min(h,round((top+oh)*sy)))
            if xend-xx>=4 and yend-yy>=4: st.session_state.rois.append({'x':xx,'y':yy,'w':xend-xx,'h':yend-yy,'filter':None}); added+=1
        if added: st.session_state.canvas_ver+=1; st.rerun()
        else: st.warning('Draw at least one rectangle first.')
    if x2.button('Remove last ROI',use_container_width=True) and st.session_state.rois: st.session_state.rois.pop(); st.session_state.canvas_ver+=1; st.rerun()
    if x3.button('Clear canvas',use_container_width=True): st.session_state.rois=[]; st.session_state.canvas_ver+=1; st.rerun()
    st.markdown('<div class="section">Region controls</div>',unsafe_allow_html=True)
    if not st.session_state.rois: st.info('No ROI selected yet. Draw a rectangle above to create a precision region.')
    for i,r in enumerate(st.session_state.rois):
        with st.container(border=True):
            q1,q2,q3=st.columns([2,2.5,.8]); q1.markdown(f'**ROI {i+1}** · `{r["w"]} × {r["h"]} px`'); q1.caption(f'x={r["x"]}, y={r["y"]}')
            cur=r.get('filter') or 'Original'; sel=q2.selectbox('Filter',FILTERS,index=FILTERS.index(cur),key=f'f_{image_id}_{i}',label_visibility='collapsed'); r['filter']=sel
            if q3.button('Delete',key=f'd_{image_id}_{i}',use_container_width=True): st.session_state.rois.pop(i); st.rerun()

with compare:
    st.markdown('<div class="section">Visual comparison</div>',unsafe_allow_html=True); mode=st.radio('View',['Side by side','Original','Edited'],horizontal=True,label_visibility='collapsed')
    if mode=='Side by side':
        q1,q2=st.columns(2); q1.caption('ORIGINAL'); q1.image(preview(pil(original)),use_container_width=True); q2.caption('EDITED'); q2.image(preview(pil(result)),use_container_width=True)
    elif mode=='Original': st.image(preview(pil(original)),use_container_width=True)
    else: st.image(preview(pil(result)),use_container_width=True)
with regions:
    st.markdown('<div class="section">ROI map</div>',unsafe_allow_html=True); st.image(overlay(result,st.session_state.rois),use_container_width=True)
    for i,r in enumerate(st.session_state.rois,1): st.write(f'**ROI {i}** — `{r.get("filter") or "No filter"}` — {r["w"]} × {r["h"]} px')
    if st.session_state.rois: st.caption('For overlapping regions, later ROIs take priority.')
with export:
    st.markdown('<div class="section">Final export</div>',unsafe_allow_html=True); q1,q2=st.columns([1.5,1]); q1.image(preview(pil(result),1100),use_container_width=True); q2.markdown('### Ready to export'); q2.write('Full-resolution result with no editor overlays.'); q2.metric('Output',f'{w:,} × {h:,} px'); q2.metric('Format','PNG / RGB')
    buf=io.BytesIO(); Image.fromarray(result).save(buf,format='PNG',optimize=True); q2.download_button('↓ Download full-resolution PNG',data=buf.getvalue(),file_name=f'roi_studio_{datetime.now():%Y%m%d_%H%M%S}.png',mime='image/png')

st.divider(); st.markdown('<div class="small">ROI Studio · Professional ROI filtering demo · Streamlit + OpenCV</div>',unsafe_allow_html=True)
