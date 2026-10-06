from pathlib import Path
import csv,json,hashlib
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter,sosfiltfilt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def read(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
manifest=read(ROOT/'segment_manifest.csv');cycles=read(ROOT/'Feature extraction_reorder 20 participants_resegment/cycle_level_features.csv')
tables=[read(ROOT/'Feature extraction_reorder 20 participants_resegment'/folder/'segment_features.csv') for folder in ['time domain features','normalized amplitude features']]
assert len(manifest)==120 and len({r['original_id'] for r in manifest})==20
for source in read(ROOT/'qc/input_manifest.csv'):
    path=Path(source['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
    fs,raw=wavfile.read(path);y=sosfiltfilt(butter(4,[30,500],btype='bandpass',fs=fs,output='sos'),raw.astype(float))
    rows=[r for r in manifest if r['original_id']==source['original_id']];assert len(rows)==6
    assert all(int(x['end_sample'])==int(y['start_sample']) for x,y in zip(rows,rows[1:]))
    chunks=[]
    edge=ROOT/'data'/f"Participant_{int(source['original_id']):02d}"/'excluded_edges'
    with np.load(edge/'prefix.npz') as d:chunks.append(d['raw_original'])
    for r in rows:
        a,b=int(r['start_sample']),int(r['end_sample']);n=int(r['reordered_id']);sid=int(r['segment_id'])
        with np.load(ROOT/r['npz']) as d,np.load(ROOT/'data_reorder'/f'Participant_{n:02d}'/f'segment_{sid:02d}.npz') as reordered:
            assert np.array_equal(d['raw_original'],raw[a:b]);assert np.array_equal(d['filtered_original'],y[a:b]);chunks.append(d['raw_original'])
            assert str(reordered['participant_id'])==f'Participant_{n:02d}'
            for k in ['raw_original','filtered_original','normalized_pcg']:assert np.array_equal(d[k],reordered[k])
            assert np.isclose(d['normalized_pcg'].mean(),0,atol=1e-12) and np.isclose(d['normalized_pcg'].std(),1)
            cr=[c for c in cycles if c['original_id']==r['original_id'] and c['segment_id']==r['segment_id']]
            for c in cr:
                bounds=[int(c[k]) for k in ['s1_onset','s1_offset','s2_onset','s2_offset','next_s1_onset']]
                assert a<=bounds[0]<bounds[-1]<=b
                durations=np.diff(bounds)/fs
                for k,v in zip(['IntS1','IntSys','IntS2','IntDia'],durations):assert np.isclose(float(c[k]),v,rtol=0,atol=1e-12)
                assert np.isclose(sum(durations),float(c['RR']))
                amps={k:np.abs(d['normalized_pcg'][l-a:h-a]).mean() for k,l,h in zip(['S1','Sys','S2','Dia'],bounds,bounds[1:])}
                for k,v in amps.items():assert np.isclose(float(c['A_'+k]),v)
                for key,num,den in [('SysS1','Sys','S1'),('DiaS2','Dia','S2'),('S1S2','S1','S2'),('S1Dia','S1','Dia'),('SysDia','Sys','Dia'),('S2Sys','S2','Sys')]:assert np.isclose(float(c['Amp_'+key]),amps[num]/amps[den])
            for table,count in zip(tables,[20,12]):
                f=next(v for v in table if v['original_id']==r['original_id'] and v['segment_id']==r['segment_id'])
                keys=[k for k in f if k.startswith(('m_','sd_'))];assert len(keys)==count
                for k in keys:
                    prefix,metric=k.split('_',1);values=[float(c[metric]) for c in cr];expected=(np.mean(values) if values else np.nan) if prefix=='m' else (np.std(values,ddof=1) if len(values)>1 else np.nan)
                    assert np.isclose(float(f[k]),expected,equal_nan=True)
        with Image.open(ROOT/r['plot']) as im:im.verify()
    with np.load(edge/'suffix.npz') as d:chunks.append(d['raw_original'])
    assert np.array_equal(np.concatenate(chunks),raw)
report=dict(status='passed',participants=20,segments=120,plots=120,cycle_rows=len(cycles),checks=['source SHA256 unchanged','whole recording reconstructed exactly including preserved edges','contiguous nonoverlapping sample intervals','original/reordered array equivalence','full-record filter reproduction','independent duration and amplitude recalculation','all 32 feature columns independently aggregated','all plots decoded'])
(ROOT/'qc/verification_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
