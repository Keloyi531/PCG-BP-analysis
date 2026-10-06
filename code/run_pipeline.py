"""Whole-record PCG detection, boundary-aligned segmentation and Tang features."""
from pathlib import Path
import sys, os, csv, json, hashlib, argparse, shutil
from dataclasses import asdict
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'qc/matplotlib')
sys.path.insert(0, str(Path(__file__).parent/'vendor'))
import stage1, stage2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
IDS = [8,9,16,17,18,19,23,24,32,33,37,41,42,44,53,54,55,68,72,78]
METRICS = ['RR','IntS1','IntS2','IntSys','IntDia','Ratio_SysRR','Ratio_DiaRR','Ratio_SysDia','Ratio_S1RR','Ratio_S2RR']
RATIOS = {'SysS1':('Sys','S1'),'DiaS2':('Dia','S2'),'S1S2':('S1','S2'),'S1Dia':('S1','Dia'),'SysDia':('Sys','Dia'),'S2Sys':('S2','Sys')}
FEATURE = ROOT/'Feature extraction_reorder 20 participants_resegment'
def write(path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def aggregate(rows, keys):
    result={}
    for k in keys:
        v=np.array([r[k] for r in rows],float)
        result['m_'+k]=float(v.mean()) if len(v) else float('nan')
        result['sd_'+k]=float(v.std(ddof=1)) if len(v)>1 else float('nan')
    return result
def cycles_from(lm,fs,n):
    cycles=[]; audit=[]
    for i,a in enumerate(lm):
        if a['label']!='S1': continue
        trip=lm[i:i+3]; reasons=[]; bounds=[]
        if len(trip)!=3 or [e['label'] for e in trip]!=['S1','S2','S1']:
            reasons.append('not_adjacent_S1_S2_S1')
        elif any(e['onset_sample'] is None or e['offset_sample'] is None for e in trip):
            reasons.append('unresolved_boundary')
        else:
            b,c=trip[1:]; bounds=[a['onset_sample'],a['offset_sample'],b['onset_sample'],b['offset_sample'],c['onset_sample']]
            if not (0<=bounds[0] and bounds[-1]<n and all(x<y for x,y in zip(bounds,bounds[1:]))): reasons.append('invalid_state_order')
        row=dict(cycle_id=len(audit)+1,s1_sound_id=a['sound_id'],valid=not reasons,reason=';'.join(reasons))
        audit.append(row)
        if reasons: continue
        d=np.diff(bounds)/fs; rr=(bounds[-1]-bounds[0])/fs
        vals=dict(zip(['IntS1','IntSys','IntS2','IntDia'],d))
        vals.update(RR=rr,Ratio_SysRR=d[1]/rr,Ratio_DiaRR=d[3]/rr,Ratio_SysDia=d[1]/d[3],Ratio_S1RR=d[0]/rr,Ratio_S2RR=d[2]/rr)
        cycles.append(dict(cycle_id=row['cycle_id'],start_sample=bounds[0],end_sample=bounds[-1],bounds=bounds,**vals))
    return cycles,audit
def cuts(onsets,fs,target):
    out=[onsets[0]]
    while onsets[-1]-out[-1]>1.5*target*fs:
        candidates=[v for v in onsets if v>out[-1]]
        out.append(min(candidates,key=lambda v:abs(v-(out[-1]+target*fs))))
    if out[-1]!=onsets[-1]: out.append(onsets[-1])
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=Path(r'C:\SYMBIO-X\SCG_PCG_BP\datasets\PCG-BP\Original_signal\Original_78_participants'));p.add_argument('--seconds',type=float,default=10);args=p.parse_args()
    alltime=[];allamp=[];allcycles=[];man=[];qcrows=[];inputs=[];mapping=[];edges=[]
    cfg=dict(participants=IDS,target_segment_seconds=args.seconds,filter={'band_hz':[30,500],'butter_N':4,'method':'sosfiltfilt on full recording'},detection=asdict(stage2.Config()),representations=asdict(stage1.Parameters()),sd_ddof=1,amplitude_normalization='segment zero mean, population SD=1',timing_pool='complete adjacent valid S1-S2-S1 cycles contained within segment; boundary S1 onset may equal exclusive segment end',sample_intervals='zero-based half-open [start,end)',RR='PCG S1 onset to next S1 onset, not ECG measured RR',edge_policy='preserve prefix/suffix separately; exclude from feature segments',validation_status='provisional; no manual reference labels')
    dump(ROOT/'processing_config.json',cfg)
    for new,original in enumerate(IDS,1):
        pid=f'Participant_{original:02d}'; rid=f'Participant_{new:02d}'; mapping.append(dict(original_id=original,reordered_id=new,original_folder=pid,reordered_folder=rid))
        src=args.raw/(pid+'.wav'); h=sha(src); fs,raw=wavfile.read(src); assert fs==2000 and raw.ndim==1 and np.isfinite(raw).all()
        inputs.append(dict(original_id=original,path=str(src),sha256=h,fs_hz=fs,n_samples=len(raw)))
        y=sosfiltfilt(butter(4,[30,500],btype='bandpass',fs=fs,output='sos'),raw.astype(float))
        d=stage1.represent(y,fs);d.update(participant_id=np.array(pid),segment_id=np.array(0),recording_start_s=np.array(0.))
        env,candidates,lm,qc,model=stage2.analyze(d)
        dest=ROOT/'S1S2 detection'/pid;dest.mkdir(parents=True,exist_ok=True)
        write(dest/'full_recording_landmarks.csv',lm);write(dest/'full_recording_candidates.csv',candidates);dump(dest/'full_recording_model.json',dict(model=model,qc=qc))
        np.savez_compressed(dest/'full_recording_detection.npz',**d,composite_envelope=env)
        cycles,audit=cycles_from(lm,fs,len(raw));write(dest/'cycle_audit.csv',audit)
        onsets=sorted({r['onset_sample'] for r in lm if r['label']=='S1' and r['onset_sample'] is not None and r['offset_sample'] is not None})
        if len(onsets)<2: raise RuntimeError(f'{pid}: no usable S1 span; review required')
        boundaries=cuts(onsets,fs,args.seconds)
        for name,a,b in [('prefix',0,boundaries[0]),('suffix',boundaries[-1],len(raw))]:
            if b>a:
                edge=ROOT/'data'/pid/'excluded_edges';edge.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(edge/(name+'.npz'),raw_original=raw[a:b],filtered_original=y[a:b],fs_hz=fs,start_sample=a,end_sample=b,original_participant_id=original)
                edges.append(dict(original_id=original,kind=name,start_sample=a,end_sample=b,duration_s=(b-a)/fs,reason='outside_first_to_last_resolved_S1_onset'))
        for sid,(a,b) in enumerate(zip(boundaries,boundaries[1:]),1):
            meta=dict(original_id=original,reordered_id=new,segment_id=sid,start_sample=a,end_sample=b,start_s=a/fs,end_s=b/fs,duration_s=(b-a)/fs)
            local=y[a:b];mu=float(local.mean());sd=float(local.std());assert sd>1e-12;z=(local-mu)/sd
            kept=[c for c in cycles if c['start_sample']>=a and c['end_sample']<=b];ar=[]
            for c in kept:
                amps={state:float(np.abs(z[l-a:r-a]).mean()) for state,l,r in zip(['S1','Sys','S2','Dia'],c['bounds'],c['bounds'][1:])}
                valid=all(v>1e-12 for v in amps.values());ratios={'Amp_'+k:amps[num]/amps[den] for k,(num,den) in RATIOS.items()} if valid else {}
                if valid:ar.append(ratios)
                allcycles.append(dict(**meta,**{k:v for k,v in c.items() if k not in ('bounds','start_sample','end_sample')},cycle_start_sample=c['start_sample'],cycle_end_sample=c['end_sample'],**dict(zip(['s1_onset','s1_offset','s2_onset','s2_offset','next_s1_onset'],c['bounds'])),**{'A_'+k:v for k,v in amps.items()},amplitude_valid=valid,**ratios))
            tr=dict(**meta,n_valid_cycles=len(kept),**aggregate(kept,METRICS));am=dict(**meta,n_valid_cycles=len(ar),**aggregate(ar,['Amp_'+k for k in RATIOS]))
            alltime.append(tr);allamp.append(am)
            stem=f'segment_{sid:02d}';od=ROOT/'data'/pid;rd=ROOT/'data_reorder'/rid
            od.mkdir(parents=True,exist_ok=True);rd.mkdir(parents=True,exist_ok=True)
            payload=dict(raw_original=raw[a:b],filtered_original=local,normalized_pcg=z,fs_hz=fs,time_s=np.arange(b-a)/fs,recording_start_s=a/fs,recording_end_s=b/fs,start_sample=a,end_sample=b,original_participant_id=original,reordered_participant_id=new,segment_id=sid,normalization_mean=mu,normalization_sd=sd,source_path=str(src),source_sha256=h)
            np.savez_compressed(od/(stem+'.npz'),participant_id=pid,**payload);np.savez_compressed(rd/(stem+'.npz'),participant_id=rid,**payload)
            for suffix,values in [('_raw.wav',raw[a:b]),('_filtered.wav',local)]:
                wavfile.write(od/(stem+suffix),fs,values);shutil.copy2(od/(stem+suffix),rd/(stem+suffix))
            selected=[]
            for r in lm:
                if a<=r['peak_sample']<b:
                    rr=dict(r);rr['segment_id']=sid
                    for key in ['onset','peak','offset']:
                        value=rr[key+'_sample'];rr['recording_'+key+'_sample']=value
                        rr[key+'_sample']=None if value is None else value-a;rr[key+'_time_s']=None if value is None else (value-a)/fs
                    selected.append(rr)
            write(dest/(stem+'_landmarks.csv'),selected)
            dd={k:(v[a:b] if isinstance(v,np.ndarray) and v.ndim==1 and len(v)==len(y) else v) for k,v in d.items()};dd['time_s']=np.arange(b-a)/fs
            q=dict(qc,segment_id=sid,n_s1=sum(r['label']=='S1' for r in selected),n_s2=sum(r['label']=='S2' for r in selected),n_ambiguous=sum(r['label']=='ambiguous' for r in selected))
            fig=stage2.plot(dd,selected,q);fig.savefig(dest/(stem+'.png'),dpi=125);plt.close(fig)
            man.append(dict(**meta,n_samples=b-a,n_valid_timing_cycles=len(kept),n_valid_amplitude_cycles=len(ar),review_flag='fewer_than_two_valid_cycles' if min(len(kept),len(ar))<2 else '',npz=str((od/(stem+'.npz')).relative_to(ROOT)),plot=str((dest/(stem+'.png')).relative_to(ROOT))))
        qcrows.append(dict(original_id=original,reordered_id=new,**qc,n_segments=len(boundaries)-1,n_complete_cycles=len(cycles),prefix_samples=boundaries[0],suffix_samples=len(raw)-boundaries[-1]))
        assert sha(src)==h
        print(f'{pid}: {len(boundaries)-1} segments, {len(cycles)} valid whole-record cycles; {qc["n_ambiguous"]} ambiguous candidates',flush=True)
    write(ROOT/'participant_id_mapping.csv',mapping);write(ROOT/'segment_manifest.csv',man);write(ROOT/'qc/edge_fragments.csv',edges);write(ROOT/'qc/participant_qc.csv',qcrows);write(ROOT/'qc/input_manifest.csv',inputs)
    write(FEATURE/'time domain features/segment_features.csv',alltime);write(FEATURE/'normalized amplitude features/segment_features.csv',allamp)
    write(FEATURE/'cycle_level_features.csv',allcycles)
    for folder,rows in [('time domain features',alltime),('normalized amplitude features',allamp)]:
        for new in range(1,21):write(FEATURE/folder/f'Participant_{new:02d}'/'segment_features.csv',[r for r in rows if r['reordered_id']==new])
    dump(ROOT/'qc/run_summary.json',dict(participants=20,segments=len(man),valid_timing_cycles=sum(r['n_valid_cycles'] for r in alltime),valid_amplitude_cycles=sum(r['n_valid_cycles'] for r in allamp),segments_requiring_review=[r for r in man if r['review_flag']],plots=len(man)))
if __name__=='__main__':main()
