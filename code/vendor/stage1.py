"""Stage 1 representations only. Input WAVs are read-only; no band-pass filtering."""
from pathlib import Path
from dataclasses import dataclass, asdict
import argparse
import csv
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d

BASE=Path(r'C:\SYMBIO-X\SCG_PCG_BP\PCG-BP_analysis\envolope_energy_30-500')
SOURCE=BASE/'data_36participant'
OUTPUT=BASE/'representations'
SELECTED_IDS=[3,8,9,11,16,17,18,19,23,24,25,28,32,33,37,38,39,41,42,43,44,49,52,53,54,55,57,59,62,65,66,67,68,72,75,78]

@dataclass(frozen=True)
class Parameters:
    smoothing_ms: float=25.0
    hilbert_reflection_ms: float=250.0
    epsilon: float=1e-12

def represent(signal, fs, params=Parameters()):
    """Same-length arrays; no detection, filtering of the PCG, or resampling.

    Peak normalization is chosen over z scoring to bound x² <= 1 for Shannon
    energy. It is not robust to isolated high-amplitude artifacts: scale is saved
    explicitly, and physical/amplitude features must use a non-normalized array.
    Centered odd-length smoothing has no causal time delay, but broadens events.
    25 ms is a provisional visualization setting, NOT a validated S1/S2 parameter.
    """
    original=np.array(signal,dtype=np.float64,copy=True)
    if original.ndim!=1 or len(original)<2 or not np.isfinite(original).all():
        raise ValueError('Expected finite, mono signal with at least two samples')
    if fs<=0 or params.smoothing_ms<=0 or params.hilbert_reflection_ms<0 or not 0<params.epsilon<1:
        raise ValueError('Invalid sampling rate or representation parameters')
    mean=float(original.mean()); centered=original-mean
    scale=float(np.max(np.abs(centered)))
    normalized=centered/scale if scale>0 else np.zeros_like(centered)
    # Reflection reduces the artificial periodic discontinuity of FFT Hilbert.
    # Crop back to original length; no invented samples enter saved time axes.
    pad=min(len(original)-1,int(round(params.hilbert_reflection_ms*fs/1000)))
    extended=np.pad(normalized,(pad,pad),mode='reflect') if pad else normalized
    he=np.abs(hilbert(extended))
    he=he[pad:pad+len(original)]
    power=normalized**2
    se=-power*np.log(power+params.epsilon)  # Natural log; exact requested formula.
    # At |x|=1 this can be ~-epsilon. Preserve that value, rather than silently
    # clipping the requested formula; zero input produces exactly zero energy.
    width=max(1,int(round(params.smoothing_ms*fs/1000)))
    if width%2==0: width+=1  # 51 samples at 2 kHz: 25.5 ms, +/-12.5 ms support.
    smooth=lambda x: uniform_filter1d(x,size=width,mode='reflect',origin=0)
    # Conservative review flag, not a guarantee of Hilbert error outside it.
    guard=min(len(original),pad+width//2)
    edge=np.zeros(len(original),dtype=bool);edge[:guard]=True
    if guard: edge[-guard:]=True
    return dict(filtered_original=original,filtered_dc_removed=centered,
                normalized_pcg=normalized,hilbert_envelope=he,
                hilbert_smoothed=smooth(he),shannon_energy=se,
                shannon_smoothed=smooth(se),time_s=np.arange(len(original))/fs,
                edge_review_mask=edge,mean_removed=np.array(mean),
                normalization_scale=np.array(scale),zero_dynamic_range=np.array(scale==0),
                fs_hz=np.array(fs),smoothing_samples=np.array(width),
                hilbert_padding_samples=np.array(pad))

def plot_qc(data,title,nominal_duration=10,window=None):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True,layout='constrained')
    labels=[('filtered_original','30–500 Hz filtered PCG (original amplitude)','WAV units','#176f8c'),
            ('hilbert_smoothed','Smoothed Hilbert amplitude envelope','Normalized amplitude','#b25824'),
            ('shannon_smoothed','Smoothed Shannon energy representation','Dimensionless','#7654a3')]
    t=data['time_s'];duration=len(t)/int(data['fs_hz'])
    for ax,(key,label,unit,color) in zip(axes,labels):
        ax.plot(t,data[key],lw=.65,color=color)
        ax.set_title(label,loc='left',fontsize=11);ax.set_ylabel(unit)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        if duration<nominal_duration: ax.axvspan(duration,nominal_duration,color='gray',alpha=.3)
        guard=(int(data['hilbert_padding_samples'])+int(data['smoothing_samples'])//2)/int(data['fs_hz'])
        ax.axvspan(0,min(guard,duration),color='#e6b84b',alpha=.12)
        ax.axvspan(max(0,duration-guard),duration,color='#e6b84b',alpha=.12)
    axes[-1].set(xlim=window or (0,nominal_duration),xlabel='Time within segment (s)')
    fig.suptitle(title+f'\n{duration:.3f} s | {int(data["fs_hz"]):,} Hz | smoothing: {int(data["smoothing_samples"])/int(data["fs_hz"])*1000:g} ms',fontsize=14)
    fig.supxlabel('Amber: boundary-review region. Gray: unavailable data_36participant. Hilbert is an amplitude envelope; Shannon is a dimensionless energy representation.',fontsize=9)
    return fig

def process_file(path,source,output,params):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    fs,x=wavfile.read(path)
    if fs!=2000 or x.dtype!=np.float64: raise ValueError(f'Unexpected format: {path}')
    match=re.fullmatch(r'segment_(\d+)_(\d+)-(\d+)s',path.stem)
    if not match: raise ValueError(f'Unexpected segment identifier: {path}')
    sid,start,end=map(int,match.groups())
    if start!=(sid-1)*5 or end-start!=10: raise ValueError('Unexpected segment boundaries')
    d=represent(x,fs,params)
    d.update(participant_id=np.array(path.parent.name),segment_id=np.array(sid),
             recording_start_s=np.array(start),nominal_end_s=np.array(end),
             source_sha256=np.array(before),source_path=np.array(str(path)),
             epsilon=np.array(params.epsilon),schema_version=np.array('1.0'))
    dest=output/path.parent.name;dest.mkdir(parents=True,exist_ok=True)
    npz=dest/(path.stem+'.npz');png=dest/(path.stem+'.png')
    np.savez_compressed(npz,**d)
    with np.load(npz,allow_pickle=False) as saved:
        assert np.array_equal(saved['filtered_original'],x)
        for key in ['normalized_pcg','hilbert_envelope','hilbert_smoothed','shannon_energy','shannon_smoothed']:
            assert saved[key].shape==x.shape and np.isfinite(saved[key]).all()
    fig=plot_qc(d,f'{path.parent.name} | Segment {sid:02d} | recording interval {start}–{end} s')
    fig.savefig(png,dpi=130);plt.close(fig)
    with Image.open(png) as im: im.verify()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    return dict(participant=path.parent.name,segment_id=sid,source=str(path),source_sha256=before,
                sample_rate_hz=fs,samples=len(x),duration_s=len(x)/fs,recording_start_s=start,
                nominal_end_s=end,complete=len(x)==(end-start)*fs,mean_removed=float(d['mean_removed']),
                normalization_scale=float(d['normalization_scale']),zero_dynamic_range=bool(d['zero_dynamic_range']),
                smoothing_samples=int(d['smoothing_samples']),npz=str(npz.relative_to(output)),figure=str(png.relative_to(output)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--smoothing-ms',type=float,default=25)
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.source.resolve()==args.output.resolve() or args.source.resolve() in args.output.resolve().parents:
        raise ValueError('Output must not be inside the selected input folder')
    files=sorted(args.source.glob('Participant_*/segment_*.wav'))
    if not files: raise ValueError('No segment WAVs found')
    expected={f'Participant_{i:02d}' for i in SELECTED_IDS}
    if {p.parent.name for p in files}!=expected or len(files)!=396:
        raise ValueError('Expected exactly the selected 36 participants and 396 segments')
    params=Parameters(smoothing_ms=args.smoothing_ms)
    args.output.mkdir(parents=True,exist_ok=True)
    settings=dict(parameters=asdict(params),source=str(args.source),selected_participants=SELECTED_IDS,sample_rate_hz=2000,
                  normalization='subtract mean then divide by peak absolute value; zero-safe',
                  smoothing='centered odd-length uniform moving average, reflect boundaries',
                  hilbert='abs(scipy.signal.hilbert(reflection-padded normalized signal)), cropped',
                  shannon='-x**2 * natural_log(x**2 + epsilon)',
                  bandpass_repeated=False,numpy_version=np.__version__,scipy_version=scipy.__version__)
    (args.output/'processing_settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(process_file,p,args.source,args.output,params) for p in files]
        for i,job in enumerate(as_completed(jobs),1):
            rows.append(job.result())
            if i%100==0 or i==len(files): print(f'Processed and verified {i}/{len(files)} segments',flush=True)
    rows.sort(key=lambda r:(r['participant'],r['segment_id']))
    with (args.output/'representation_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Completed {len(rows)} representations; incomplete: {sum(not r["complete"] for r in rows)}; zero dynamic range: {sum(r["zero_dynamic_range"] for r in rows)}',flush=True)

if __name__=='__main__': main()
