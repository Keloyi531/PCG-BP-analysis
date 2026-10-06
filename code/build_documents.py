from pathlib import Path
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Feature extraction_reorder 20 participants_resegment'
styles=getSampleStyleSheet()
for key in ['Normal','BodyText']:
    styles[key].fontSize=9;styles[key].leading=11
timing=[('RR',r'(e_i-a_i)/f_s'),('IntS1',r'(b_i-a_i)/f_s'),('IntS2',r'(d_i-c_i)/f_s'),('IntSys',r'(c_i-b_i)/f_s'),('IntDia',r'(e_i-d_i)/f_s'),('Ratio_SysRR',r'T_{Sys,i}/RR_i'),('Ratio_DiaRR',r'T_{Dia,i}/RR_i'),('Ratio_SysDia',r'T_{Sys,i}/T_{Dia,i}'),('Ratio_S1RR',r'T_{S1,i}/RR_i'),('Ratio_S2RR',r'T_{S2,i}/RR_i')]
amp=[('Amp_'+k,r'A_{'+a+r',i}/A_{'+b+r',i}') for k,a,b in [('SysS1','Sys','S1'),('DiaS2','Dia','S2'),('S1S2','S1','S2'),('S1Dia','S1','Dia'),('SysDia','Sys','Dia'),('S2Sys','S2','Sys')]]
def math_image(eq):
    fig=plt.figure(figsize=(6,.42));fig.text(.01,.3,'$'+eq+'$',fontsize=12)
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=200,bbox_inches='tight',pad_inches=.04);plt.close(fig);buf.seek(0)
    w,h=ImageReader(buf).getSize();return Image(buf,width=w*.30,height=h*.30)
def build(folder,title,rows,notes,extra):
    dest=BASE/folder;dest.mkdir(parents=True,exist_ok=True)
    story=[Paragraph(title,styles['Title']),Paragraph('PCG-BP analysis version 2 | Feature calculation specification',styles['Normal']),Spacer(1,12)]
    for note in notes:story += [Paragraph(note,styles['BodyText']),Spacer(1,6)]
    for eq in extra:story += [math_image(eq),Spacer(1,6)]
    table=[['Per-cycle quantity','Equation','Output columns']]
    for name,eq in rows:table.append([Paragraph(name,styles['Normal']),math_image(eq),Paragraph('m_'+name+'<br/>sd_'+name,styles['Normal'])])
    t=Table(table,colWidths=[125,190,175],repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff4')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LINEBELOW',(0,0),(-1,0),.6,colors.gray),('BOTTOMPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4)]));story += [t,Spacer(1,12)]
    story += [Paragraph('Aggregation within each segment',styles['Heading2']),math_image(r'm_q=\frac{1}{K}\sum_{i=1}^{K}q_i'),math_image(r'sd_q=\sqrt{\frac{1}{K-1}\sum_{i=1}^{K}(q_i-m_q)^2}'),Paragraph('Ratios are calculated per cycle, then averaged. Sample SD uses K-1 (an explicit implementation convention). Mean is NaN for K=0; SD is NaN for K&lt;2. Invalid or ambiguous cycles are excluded, never bridged or filled with zero.',styles['BodyText']),Spacer(1,8),Paragraph('Source: Tang et al. (2018), PCG Classification Using Multidomain Features and SVM Classifier, Tables 1-2 and Sections 2.3-2.5.3. DOI: 10.1155/2018/4205027. Feature definitions follow the paper; this project retains its 30-500 Hz Butterworth filter and timing-based detector rather than the paper\'s 10 Hz high-pass, spike removal and Springer HSMM.',styles['BodyText'])]
    SimpleDocTemplate(str(dest/'feature_equations.pdf'),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=38).build(story)
    def esc(s):return s.replace('_',r'\_').replace('&lt;','<')
    tex=r'''\documentclass[11pt]{article}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage{amsmath,booktabs,longtable,hyperref}
\begin{document}
\title{'''+title+r'''}\author{PCG-BP analysis version 2}\date{}\maketitle
'''
    tex+='\n\n'.join(esc(n) for n in notes)+'\n'
    for eq in extra:tex+='\\['+eq+'\\]\n'
    tex+=r'\begin{longtable}{lll}\toprule Quantity & Equation & Output columns\\\midrule'+'\n'
    for name,eq in rows:tex+=esc(name)+' & $'+eq+'$ & '+esc('m_'+name)+', '+esc('sd_'+name)+r'\\'+'\n'
    tex+=r'''\bottomrule\end{longtable}
\section*{Aggregation within each segment}
\[m_q=\frac{1}{K}\sum_{i=1}^{K}q_i,\qquad sd_q=\sqrt{\frac{1}{K-1}\sum_{i=1}^{K}(q_i-m_q)^2}.\]
Ratios are calculated per cycle, then averaged. Sample SD uses $K-1$, an explicit implementation convention. Mean is NaN for $K=0$; SD is NaN for $K<2$. Invalid or ambiguous cycles are excluded, never bridged or filled with zero.

\section*{Reference and adaptations}
Tang et al. (2018), \emph{PCG Classification Using Multidomain Features and SVM Classifier}, Tables 1--2 and Sections 2.3--2.5.3. \url{https://doi.org/10.1155/2018/4205027}.
The feature definitions follow the paper. This project retains its 30--500 Hz Butterworth filter and timing-based detector instead of the paper's 10 Hz high-pass, spike removal and Springer HSMM.
\end{document}
'''
    (dest/'feature_equations.tex').write_text(tex,encoding='utf-8');(dest/'feature_equations_overleaf.txt').write_text(tex,encoding='utf-8')
common=['For each complete adjacent S1-S2-S1 cycle, a, b, c, d, e denote S1 onset, S1 offset, S2 onset, S2 offset and next S1 onset, in zero-based recording samples. They must satisfy a &lt; b &lt; c &lt; d &lt; e. Sampling frequency is f_s. States use half-open sample intervals [a,b), [b,c), [c,d), [d,e).','Only complete cycles contained in a segment contribute. The next S1 onset may equal the exclusive segment end; its waveform belongs to the next segment. Detection is performed once over the whole recording.']
build('time domain features','Time-domain features: 20 equations',timing,common+['Intervals are in seconds; ratios are dimensionless. RR is a PCG cycle-period surrogate from consecutive S1 onsets, not an ECG-measured RR. T_S1, T_Sys, T_S2 and T_Dia denote IntS1, IntSys, IntS2 and IntDia. All ten quantities use the same complete-cycle pool.'],[r'RR_i=T_{S1,i}+T_{Sys,i}+T_{S2,i}+T_{Dia,i}'])
build('normalized amplitude features','Normalized amplitude features: 12 equations',amp,common+['Normalize the filtered waveform separately within each saved segment to zero mean and unit population SD. L is the number of segment samples. A_Q is the mean absolute normalized waveform amplitude over state Q, not an envelope peak or RMS. Exclude a cycle if any state amplitude is at most 1e-12. All six ratios use the same valid-cycle pool.'],[r'z[n]=\frac{x[n]-\mu_x}{\sigma_x},\quad \mu_x=\frac{1}{L}\sum_n x[n],\quad \sigma_x=\sqrt{\frac{1}{L}\sum_n(x[n]-\mu_x)^2}',r'A_{Q,i}=\frac{1}{|I_{Q,i}|}\sum_{n\in I_{Q,i}}|z[n]|'])
print('Created two PDF, LaTeX and Overleaf TXT specifications.')
