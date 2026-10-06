"""Development-only heart-sound candidate, timing-label and boundary pipeline.

No BP labels, band-pass filtering, Stage-1 recomputation, or final features.
Timing labels are an explicitly provisional short/long-cycle hypothesis.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import argparse,csv,json,hashlib,shutil
import numpy as np
import scipy
from scipy.signal import find_peaks,peak_widths,correlate
from scipy.ndimage import median_filter,uniform_filter1d

BASE=Path(r'/PCG-BP_analysis/S1S2 detection')
STAGE1=Path(r'C:\SYMBIO-X\SCG_PCG_BP\PCG-BP_analysis\envolope_energy_30-500')
IDS=[8,9,16,17,18,19,23,24,32,33,37,41,42,43,44,52,53,54,55,65,68,72,75,78]

@dataclass(frozen=True)
class Config:
    # Initial engineering settings; not fitted to one participant or BP values.
    candidate_distance_s:float=.09
    prominence_fraction:float=.07  # relative to segment p95-p10 composite range
    local_height_fraction:float=.12
    local_window_s:float=.8
    prominence_window_s:float=.5
    support_width_min_s:float=.012  # confidence support, not candidate exclusion
    support_width_max_s:float=.22
    cycle_min_s:float=.40
    cycle_max_s:float=1.50  # broad 40-150 bpm development search; flag limitations
    systole_min_s:float=.14
    systole_max_s:float=.45
    diastole_min_s:float=.16
    interval_tolerance_s:float=.085
    interval_tolerance_fraction:float=.12
    min_repeating_cycles:int=3
    max_cycle_relative_error:float=.18
    min_short_long_gap_s:float=.065
    min_classification_confidence:float=.60
    boundary_relative_height:float=.20  # baseline + 20% of local peak prominence
    boundary_search_s:float=.20
    duration_min_s:float=.025
    duration_max_s:float=.25
    boundary_baseline_quantile:float=.10
    local_energy_window_s:float=.0505
    cycle_mad_fraction_max:float=.12
    min_model_coverage:float=.55
    harmonic_competition_margin:float=.035
    min_transition_s:float=.10
    systole_grid_step_s:float=.02

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as d: return {k:d[k] for k in d.files}
def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def prepare():
    """Inventory everything before copying. Original Stage 1 remains read-only."""
    pairs=[];errors=[]
    wavroot=STAGE1/'data_36participant'
    if not wavroot.exists(): wavroot=STAGE1/'data'
    for pid in IDS:
        name=f'Participant_{pid:02d}'
        for sid in range(3,10):
            n=list((STAGE1/'representations'/name).glob(f'segment_{sid:02d}_*.npz'))
            w=list((wavroot/name).glob(f'segment_{sid:02d}_*.wav'))
            if len(n)!=1 or len(w)!=1:
                errors.append(f'{name} segment {sid}: {len(n)} NPZ, {len(w)} WAV');continue
            d=load(n[0])
            if str(d['participant_id'])!=name or int(d['segment_id'])!=sid or int(d['fs_hz'])!=2000:
                errors.append(f'Metadata mismatch: {n[0]}')
            pairs.append((n[0],w[0]))
    report=dict(expected_participants=24,expected_segments=168,missing_or_duplicate_errors=errors)
    BASE.mkdir(parents=True,exist_ok=True)
    (BASE/'inventory_report.json').write_text(json.dumps(report,indent=2))
    if errors: raise ValueError('\n'.join(errors))
    data=BASE/'data'; data.mkdir(exist_ok=True)
    expected={f'Participant_{pid:02d}' for pid in IDS}
    if {p.name for p in data.iterdir() if p.is_dir()}-expected:
        raise ValueError('Unexpected existing participant directories in destination; no data removed')
    rows=[]
    from scipy.io import wavfile
    for npz,wav in pairs:
        dest=data/npz.parent.name;dest.mkdir(exist_ok=True)
        for source in (npz,wav):
            target=dest/source.name
            if target.exists() and digest(target)!=digest(source): raise ValueError(f'Conflicting copy {target}')
            if not target.exists(): shutil.copy2(source,target)
            assert digest(source)==digest(target)
            rows.append(dict(participant=npz.parent.name,segment_id=int(load(npz)['segment_id']),source=str(source),copy=str(target),sha256=digest(source)))
        d=load(dest/npz.name);fs,x=wavfile.read(dest/wav.name)
        assert np.array_equal(x,d['filtered_original']) and len(x)==20000 and fs==2000
    for name in expected:
        files=list((data/name).glob('*.npz'))
        assert len(files)==7 and {int(load(p)['segment_id']) for p in files}==set(range(3,10))
        assert len(list((data/name).glob('*.wav')))==7
    assert len(list(data.glob('Participant_*/*.npz')))==168
    write_csv(BASE/'copy_manifest.csv',rows)
    for name in ('processing_settings.json','representation_manifest.csv'):
        shutil.copy2(STAGE1/'representations'/name,BASE/('stage1_'+name))
    report.update(verified_participants=24,verified_segments=168,segments_per_participant=7,segment_ids=list(range(3,10)),hash_verified=True)
    (BASE/'inventory_report.json').write_text(json.dumps(report,indent=2))
    print('COPY VERIFIED: 24 participants, exactly segments 03–09, 168 NPZ + 168 WAV; missing/duplicates: none.',flush=True)

def normalize_envelope(x):
    x=np.maximum(x,0);lo,hi=np.quantile(x,[.1,.95])
    return np.clip((x-lo)/max(hi-lo,1e-15),0,3)

def detect_candidates(d,c):
    fs=int(d['fs_hz']);h=d['hilbert_smoothed'];s=d['shannon_smoothed']
    env=(normalize_envelope(h)+normalize_envelope(s))/2
    spread=max(float(np.quantile(env,.95)-np.quantile(env,.1)),1e-12)
    # Local baseline only adapts candidate threshold, not the saved envelopes.
    local=median_filter(env,size=(int(c.local_window_s*fs)//2)*2+1,mode='reflect')
    peaks,props=find_peaks(env,distance=max(1,round(c.candidate_distance_s*fs)),
                          prominence=c.prominence_fraction*spread,
                          height=local+c.local_height_fraction*spread,
                          wlen=(round(c.prominence_window_s*fs)//2)*2+1,
                          width=(None,None))
    energy=uniform_filter1d(env**2,size=max(1,round(c.local_energy_window_s*fs)),mode='reflect')
    rows=[]
    for i,p in enumerate(peaks):
        width=props['widths'][i]/fs
        support=bool(c.support_width_min_s<=width<=c.support_width_max_s)
        rows.append(dict(candidate_index=i,sample_index=int(p),candidate_time_s=p/fs,
                         envelope_peak_value=float(h[p]),shannon_energy_peak_value=float(s[p]),
                         composite_peak_value=float(env[p]),prominence=float(props['prominences'][i]),
                         width_samples=float(props['widths'][i]),width_s=float(width),
                         local_height_threshold=float(local[p]+c.local_height_fraction*spread),
                         local_representation_energy=float(energy[p]),width_supported=support,
                         edge_review=bool(d['edge_review_mask'][p])))
    return env,rows

def cycle_hypotheses(env,fs,c):
    # Downsample the representation (not the PCG) for inexpensive autocorrelation.
    y=env[::max(1,fs//100)];rate=fs/max(1,fs//100);y=y-y.mean()
    ac=correlate(y,y,mode='full',method='fft')[len(y)-1:]
    ac=ac/np.maximum(np.arange(len(y),0,-1),1)
    ac=ac/max(ac[0],1e-15)
    lo=int(c.cycle_min_s*rate);hi=min(len(ac)-1,int(c.cycle_max_s*rate))
    pp,_=find_peaks(ac[lo:hi+1],distance=max(1,int(.08*rate)))
    pp=pp+lo
    if not len(pp): pp=np.array([lo+np.argmax(ac[lo:hi+1])])
    order=pp[np.argsort(ac[pp])[::-1]][:4]
    return [(p/rate,float(ac[p])) for p in order]

def fit_sequence(candidates,env,fs,c):
    """Search alternating-state paths, allowing skipped (ambiguous) candidates.

    Amplitude contributes equally to S1/S2 eligibility; never chooses sound type.
    Absolute phase is only proposed when repeated short-long timing is supported.
    Equal/near-equal intervals and competing half/double-cycle models are flagged.
    This is NOT the trained Springer HSMM or a validated physiological classifier.
    """
    n=len(candidates)
    if n<6: return None
    t=np.array([r['candidate_time_s'] for r in candidates])
    prom=np.array([r['prominence'] for r in candidates])
    q=np.clip(prom/max(np.quantile(prom,.75),1e-12),0,1)
    q*=np.array([1 if r['width_supported'] else .55 for r in candidates])
    hypotheses=[]
    for rr,ac in cycle_hypotheses(env,fs,c):
        for sys in np.arange(c.systole_min_s,min(c.systole_max_s,rr-c.diastole_min_s)+.001,c.systole_grid_step_s):
            dia=rr-sys
            # Both orientations are searched. Typical short-long ordering is a
            # modest prior; classification later requires an explicit timing gap.
            prior=.20 if dia>sys else 0
            tol=max(c.interval_tolerance_s,c.interval_tolerance_fraction*rr)
            score=np.full((n,2),-1e9);prev=np.full((n,2),-1,int)
            for i in range(n):
                for state,target in ((0,dia),(1,sys)):
                    dt=t[i]-t[:i]
                    valid=np.where((dt>=c.min_transition_s)&(np.abs(dt-target)<=2.2*tol))[0]
                    base=.6+q[i]
                    score[i,state]=base
                    if len(valid):
                        value=score[valid,1-state]+base-.7*((dt[valid]-target)/tol)**2
                        # Discourage skipping strong candidates, without deleting any.
                        value-=np.array([.16*np.sum(q[j+1:i]) for j in valid])
                        k=int(np.argmax(value))
                        if value[k]>base: score[i,state]=value[k];prev[i,state]=valid[k]
            i,state=np.unravel_index(np.argmax(score),score.shape)
            path=[]
            while i>=0:
                path.append((int(i),int(state)));j=prev[i,state];i=int(j);state=1-state
            path=path[::-1]
            coverage=(t[path[-1][0]]-t[path[0][0]])/max(len(env)/fs,1)
            objective=float(np.max(score)/max(n,1)+.10*ac+prior*len(path)/max(n,1)+.2*coverage)
            hypotheses.append(dict(rr=rr,sys=float(sys),dia=float(dia),autocorrelation=ac,
                                   score=objective,path=path,coverage=coverage))
    if not hypotheses:return None
    hypotheses.sort(key=lambda r:r['score'],reverse=True)
    best=hypotheses[0]
    # Label-phase inversion of the same cycle is assessed by short-long gap, not
    # mistaken for independent evidence. Distinct RR competitors flag harmonics.
    distinct=[h for h in hypotheses if abs(h['rr']-best['rr'])>.18*best['rr']]
    best['rr_competition_margin']=best['score']-distinct[0]['score'] if distinct else 1.0
    return best

def classify(candidates,model,c):
    labels=['ambiguous']*len(candidates);confidence=np.zeros(len(candidates));reasons=['off_temporal_path']*len(candidates)
    if model is None:return labels,confidence,['insufficient_candidates']*len(candidates)
    path=model['path'];times=np.array([r['candidate_time_s'] for r in candidates])
    i1=[i for i,s in path if s==0]; rr=np.diff(times[i1]);med=float(np.median(rr)) if len(rr) else 0
    enough=len(rr)>=c.min_repeating_cycles
    consistent=bool(len(rr) and np.median(np.abs(rr-med))/max(med,1e-12)<c.cycle_mad_fraction_max)
    orientation=model['dia']-model['sys']>=c.min_short_long_gap_s
    harmonic=model['rr_competition_margin']<c.harmonic_competition_margin
    global_ok=enough and consistent and orientation and not harmonic and model['coverage']>=c.min_model_coverage
    for k,(idx,state) in enumerate(path):
        if not global_ok:
            reasons[idx]='weak_or_competing_cycle_model';continue
        # Require neighbor support on both sides; edge sounds remain candidates.
        if k==0 or k==len(path)-1: reasons[idx]='incomplete_temporal_context';continue
        prev,ps=path[k-1];nxt,ns=path[k+1]
        a=times[idx]-times[prev];b=times[nxt]-times[idx]
        expa=model['dia'] if state==0 else model['sys']
        expb=model['sys'] if state==0 else model['dia']
        err=max(abs(a-expa),abs(b-expb))
        tol=max(c.interval_tolerance_s,c.interval_tolerance_fraction*model['rr'])
        # Peaks in the Stage-1 edge mask are never confidently labeled.
        conf=float(np.clip(.95-.35*err/tol,0,.95))
        if not candidates[idx]['width_supported']: conf*=.65
        if abs(a+b-model['rr'])>c.max_cycle_relative_error*model['rr']: conf*=.4
        if candidates[idx]['edge_review']: conf=0
        confidence[idx]=conf
        if conf>=c.min_classification_confidence:
            labels[idx]='S1' if state==0 else 'S2';reasons[idx]='timing_consistent_provisional'
        else:reasons[idx]='low_local_timing_confidence_or_edge'
    return labels,confidence,reasons

def boundaries(env,candidates,labels,fs,c):
    """Local baseline + relative peak excursion, bounded by adjacent candidates.
    Missing threshold crossings or implausible widths are reported, never invented.
    """
    results={};peaks=[r['sample_index'] for r in candidates]
    for i,p in enumerate(peaks):
        if labels[i]=='ambiguous':continue
        left=max(0,p-round(c.boundary_search_s*fs));right=min(len(env)-1,p+round(c.boundary_search_s*fs))
        if i:left=max(left,(peaks[i-1]+p)//2+1)
        if i<len(peaks)-1:right=min(right,(p+peaks[i+1])//2)
        baseline=float(np.quantile(env[left:right+1],c.boundary_baseline_quantile))
        threshold=baseline+c.boundary_relative_height*(env[p]-baseline)
        ll=np.flatnonzero(env[left:p]<=threshold);rr=np.flatnonzero(env[p+1:right+1]<=threshold)
        if not len(ll) or not len(rr): results[i]=(None,None,threshold,'boundary_threshold_not_crossed');continue
        onset=left+int(ll[-1]);offset=p+1+int(rr[0]);dur=(offset-onset)/fs
        reason='' if onset<p<offset and c.duration_min_s<=dur<=c.duration_max_s else 'boundary_duration_out_of_range'
        results[i]=(onset if not reason else None,offset if not reason else None,threshold,reason)
    return results

def analyze(d,c=Config()):
    fs=int(d['fs_hz']);env,candidates=detect_candidates(d,c)
    model=fit_sequence(candidates,env,fs,c);labels,confidence,reasons=classify(candidates,model,c)
    bounds=boundaries(env,candidates,labels,fs,c);landmarks=[]
    pid=str(d['participant_id']);sid=int(d['segment_id'])
    for i,r in enumerate(candidates):
        r.update(participant_id=pid,segment_id=sid)
        onset,offset,threshold,bwarning=bounds.get(i,(None,None,None,''))
        landmarks.append(dict(participant_id=pid,segment_id=sid,sound_id=i+1,candidate_index=i,
            label=labels[i],onset_sample=onset,peak_sample=r['sample_index'],offset_sample=offset,
            onset_time_s=onset/fs if onset is not None else None,peak_time_s=r['candidate_time_s'],
            offset_time_s=offset/fs if offset is not None else None,
            duration_s=(offset-onset)/fs if onset is not None else None,peak_prominence=r['prominence'],
            classification_confidence=float(confidence[i]),classification_reason=reasons[i],
            boundary_threshold=threshold,boundary_warning=bwarning,
            recording_peak_time_s=r['candidate_time_s']+float(d['recording_start_s'])))
    s1=[r['peak_time_s'] for r in landmarks if r['label']=='S1'];intervals=np.diff(s1)
    # Only complete adjacent confident S1-S2-S1 triplets contribute QC timing.
    st=[];dt=[];cycles=[]
    for a,b,cc in zip(landmarks,landmarks[1:],landmarks[2:]):
        if [a['label'],b['label'],cc['label']]==['S1','S2','S1']:
            st.append(b['peak_time_s']-a['peak_time_s']);dt.append(cc['peak_time_s']-b['peak_time_s']);cycles.append(cc['peak_time_s']-a['peak_time_s'])
    median=lambda a: float(np.median(a)) if len(a) else None
    warnings=[]
    if not model:warnings.append('insufficient_temporal_model')
    if model and model['rr_competition_margin']<c.harmonic_competition_margin:warnings.append('competing_cycle_harmonic')
    if sum(x=='ambiguous' for x in labels)>len(labels)*.4:warnings.append('many_ambiguous_candidates')
    if not cycles:warnings.append('no_complete_confident_triplets')
    if any(r['boundary_warning'] for r in landmarks):warnings.append('some_boundaries_unresolved')
    qc=dict(participant_id=pid,segment_id=sid,n_candidates=len(candidates),n_s1=labels.count('S1'),n_s2=labels.count('S2'),
            n_ambiguous=labels.count('ambiguous'),estimated_heart_rate=60/median(cycles) if cycles else None,
            median_s1_s2_interval=median(st),median_s2_next_s1_interval=median(dt),median_s1_s1_interval=median(cycles),
            model_cycle_s=model['rr'] if model else None,model_autocorrelation=model['autocorrelation'] if model else None,
            n_valid_boundaries=sum(r['onset_sample'] is not None for r in landmarks),
            qc_warnings=';'.join(warnings),validation_status='provisional_unvalidated')
    return env,candidates,landmarks,qc,model

def plot(d,landmarks,qc):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(3,1,figsize=(17,10),sharex=True,layout='constrained');t=d['time_s']
    keys=['filtered_original','hilbert_smoothed','shannon_smoothed']
    titles=['30–500 Hz filtered PCG','Smoothed Hilbert amplitude envelope','Smoothed Shannon representation']
    colors={'S1':'#197d3a','S2':'#b72c40','ambiguous':'#d19316'}
    for ax,key,title in zip(axes,keys,titles):
        ax.plot(t,d[key],color='#34667d',lw=.6);ax.set_title(title,loc='left',fontsize=11);ax.grid(alpha=.15)
        for r in landmarks:
            p=r['peak_sample'];label=r['label'];color=colors[label]
            ax.scatter(t[p],d[key][p],s=20,marker='x' if label=='ambiguous' else 'o',color=color,zorder=4)
            if r['onset_sample'] is not None:
                a=r['onset_time_s'];b=r['offset_time_s']
                ax.axvspan(a,b,color=color,alpha=.10)
                ax.axvline(a,color=color,ls='--',lw=.6,alpha=.7);ax.axvline(b,color=color,ls=':',lw=.6,alpha=.7)
        ax.set_xlim(0,len(t)/int(d['fs_hz']))
    for r in landmarks:
        if r['label']!='ambiguous':axes[1].annotate(r['label'],(r['peak_time_s'],d['hilbert_smoothed'][r['peak_sample']]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7,color=colors[r['label']])
    axes[0].set_ylabel('WAV units');axes[1].set_ylabel('Normalized amplitude');axes[2].set_ylabel('Dimensionless');axes[2].set_xlabel('Time within segment (s)')
    handles=[Line2D([],[],color=color,marker='x' if label=='ambiguous' else 'o',ls='',label=label+' candidate') for label,color in colors.items()]
    handles += [Line2D([],[],color='gray',ls='--',label='Onset'),Line2D([],[],color='gray',ls=':',label='Offset')]
    axes[0].legend(handles=handles,ncol=5,fontsize=8,loc='upper right')
    hr_text=f'{qc["estimated_heart_rate"]:.1f} bpm' if qc['estimated_heart_rate'] else 'unresolved'
    fig.suptitle(f'{qc["participant_id"]} | Segment {qc["segment_id"]:02d} | PROVISIONAL timing-based labels\nS1: {qc["n_s1"]} | S2: {qc["n_s2"]} | ambiguous: {qc["n_ambiguous"]} | HR: {hr_text}',fontsize=14)
    fig.supxlabel('Onset/offset dashed/dotted; shaded intervals = resolved boundaries. Missing boundaries remain blank in CSV. Labels require human review.',fontsize=9)
    return fig

def run(config,output,plots=True):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    allc=[];alll=[];allq=[]
    for i,path in enumerate(sorted((BASE/'data').glob('Participant_*/*.npz')),1):
        d=load(path);env,cands,land,qc,model=analyze(d,config)
        allc.extend(cands);alll.extend(land);allq.append(qc)
        dest=output/path.parent.name;dest.mkdir(exist_ok=True)
        if land:write_csv(dest/(path.stem+'_landmarks.csv'),land)
        if cands:write_csv(dest/(path.stem+'_candidates.csv'),cands)
        # Save exact detection representation and model for reproducibility.
        np.savez_compressed(dest/(path.stem+'_detection.npz'),composite_envelope=env,fs_hz=d['fs_hz'])
        (dest/(path.stem+'_model.json')).write_text(json.dumps(model,indent=2))
        if plots:
            fig=plot(d,land,qc);fig.savefig(dest/(path.stem+'.png'),dpi=130);plt.close(fig)
        if i%28==0:print(f'Analyzed {i}/168 segments',flush=True)
    if allc:write_csv(output/'candidates.csv',allc)
    if alll:write_csv(output/'landmarks.csv',alll)
    write_csv(output/'segment_qc.csv',allq)
    (output/'config.json').write_text(json.dumps(asdict(config),indent=2))
    return allq

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,default=BASE/'results');p.add_argument('--no-plots',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    c=Config(**json.loads(a.config.read_text())) if a.config else Config()
    q=run(c,a.output,not a.no_plots)
    print(f'Completed {len(q)} segments; S1={sum(r["n_s1"] for r in q)}, S2={sum(r["n_s2"] for r in q)}, ambiguous={sum(r["n_ambiguous"] for r in q)}',flush=True)
