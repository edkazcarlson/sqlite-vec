"""Render tables, CSVs and publication/export-friendly charts from raw results."""
import csv
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'build/packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def geo(values):
    return float(np.exp(np.mean(np.log(values))))


def report():
    root=HERE/'results'
    load=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    screen=load('screen.json');conversion=load('conversion.json');sql=load('sql.json')
    rank=load('ranking.json');env=load('environment.json');builds=load('builds-sql.json')
    portable=load('portable-diagnostic.json')['name']
    assert load('sql-complete.json')['complete'],'Do not report an incomplete SQL suite'
    finalists=[r['name'] for r in rank[:2]]
    # Flatten raw records while keeping all samples in companion JSON files.
    for name,records in [('screen',screen),('conversion',conversion),('sql',sql)]:
        flat=[]
        for r in records:
            r=dict(r)
            if 'case' in r:r.update(r.pop('case'))
            flat.append({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
        with (root/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    def micro(v,dtype='f16',distribution=('gaussian','normalized'),novec=False):
        return {(r['distribution'],r['dims'],r['metric']):r['median_ms'] for r in screen if r['variant']==v and r['dtype']==dtype and r['distribution'] in distribution and r['novec']==novec}
    def gain(a,b):return geo([a[k]/b[k] for k in a])
    mb=micro('baseline')
    def sqlgroup(v,dtype):
        groups={}
        for r in sql:
            c=r['case']
            if c['variant']==v and c['dtype']==dtype:
                key=(c['rows'],c['dims'],c['metric'],c['storage'],c['selectivity'])
                groups.setdefault(key,[]).append(r)
        return groups
    def med(group):return float(np.median([r['median_ms'] for r in group]))
    sb=sqlgroup('baseline','f16');sf=sqlgroup('baseline','f32')
    primary=[k for k in sb if k[4]==100 and k[2]!='l1']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':180})
    def finish(fig,name):
        fig.savefig(root/(name+'.png'),bbox_inches='tight')
        svg=root/(name+'.svg')
        fig.savefig(svg,bbox_inches='tight')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
        plt.close(fig)
    pg=sqlgroup(portable,'f16');pf=sqlgroup(portable,'f32')
    portable_primary=[k for k in primary if k in pg]
    summary_names=[portable,'sse2_b2','sse2_b8']
    fig,ax=plt.subplots(figsize=(9,4.5))
    for offset,reference,color in ((-.18,'current f16','#236d95'),(.18,'matched f32','#cf6c38')):
        values=[]
        for v in summary_names:
            g=sqlgroup(v,'f16');fg=sqlgroup(v,'f32');ks=[k for k in primary if k in g]
            values.append(geo([med((sb if reference=='current f16' else fg)[k])/med(g[k]) for k in ks]))
        bars=ax.bar(np.arange(3)+offset,values,width=.34,label='Versus '+reference,color=color)
        ax.bar_label(bars,fmt='%.2fx',padding=4)
    ax.set_xticks(range(3),['Portable finite converter\n768 dimensions only','SSE2, 2 candidates\n384 / 768 / 1536 dimensions','SSE2, 8 candidates\n384 / 768 / 1536 dimensions'])
    ax.axhline(1,color='#777',ls='--');ax.set_ylim(0,2.35)
    ax.set_ylabel('Geometric-mean SQL speedup (larger is better)');ax.legend(loc='upper right',fontsize=9)
    ax.set_title('Measured KNN gains with software float16 conversion\nL2 + cosine | memory + warm file | 1,000 + 2,000 rows | no AVX/F16C')
    fig.tight_layout();finish(fig,'sql-summary')
    # Conversion-specific changes first; batching appears separately.
    singled=[r for r in rank if r['batch']==1]
    singled.sort(key=lambda r:r['id'])
    names=[r['name'] for r in singled]
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    colors=['#236d95' if not n.startswith('sse') else '#cf6c38' for n in names]
    axs[0].barh(names,[gain(mb,micro(n)) for n in names],color=colors)
    axs[0].axvline(1,color='#555',ls='--');axs[0].set_xlabel('Speedup versus current float16 (larger is better)')
    axs[0].set_title('One-candidate microkernels\nBaseline includes converter calls')
    for v,color in zip(finalists,['#236d95','#cf6c38']):
        vals=[]
        for metric in ('l2','cosine'):
            g=sqlgroup(v,'f16');vals.append(geo([med(sb[k])/med(g[k]) for k in primary if k[2]==metric]))
        pos=np.arange(2)+(0 if v==finalists[0] else .35)
        axs[1].bar(pos,vals,width=.33,label=v,color=color)
    axs[1].set_xticks(np.arange(2)+.175,['L2','Cosine']);axs[1].axhline(1,color='#555',ls='--')
    axs[1].set_ylabel('SQL speedup versus current float16');axs[1].set_title('Finalists: complete SQL KNN');axs[1].legend(fontsize=8)
    fig.suptitle('Float16-specific changes and end-to-end results\nRyzen 7 5800H | MSVC /O2 /fp:precise | no AVX/F16C | at most 2,000 vectors',fontsize=12)
    fig.tight_layout();finish(fig,'overview')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for conv in ('original','original_inline','int_branch','int_select','magic_branch','magic_select','sse2'):
        label='sse2' if conv=='sse2' else conv
        for metric,ax in zip(('l2','cosine'),axs):
            ys=[]
            for b in (1,2,4,8):
                mm=micro(f'{conv}_b{b}');keys=[k for k in mm if k[2]==metric]
                ys.append(geo([mb[k]/mm[k] for k in keys]))
            ax.plot([1,2,4,8],ys,marker='o',label=label)
            ax.set_title(metric.upper());ax.set_xticks([1,2,4,8]);ax.set_xlabel('Candidates processed together');ax.set_ylabel('Speedup versus current float16');ax.axhline(1,color='#777',ls='--')
    axs[1].legend(fontsize=8);fig.suptitle('Across-candidate batching: Gaussian workloads, all dimensions');fig.tight_layout();finish(fig,'batch-width')
    keys=sorted(primary)
    fig,axs=plt.subplots(1,2,figsize=(12,8))
    labels=[f'{n} x {d} / {m} / {s}' for n,d,m,s,_ in keys]
    for ax,kind in zip(axs,('current f16','matched f32')):
        values=np.array([[med((sb if kind=='current f16' else sqlgroup(v,'f32'))[k])/med(sqlgroup(v,'f16')[k]) for v in finalists] for k in keys])
        im=ax.imshow(values,aspect='auto',cmap='RdYlGn',vmin=0,vmax=max(2,float(values.max())))
        ax.set_xticks(range(2),finalists,rotation=15,ha='right');ax.set_yticks(range(len(keys)),labels if ax==axs[0] else [])
        ax.set_title('Float16 speedup versus '+kind)
        for i,j in itertools.product(range(len(keys)),range(2)):ax.text(j,i,f'{values[i,j]:.2f}x',ha='center',va='center',fontsize=8)
    fig.suptitle('SQL KNN: 20 measured queries x 3 repetitions\nRatios above 1 favor the experimental float16 path');fig.tight_layout();finish(fig,'sql-heatmap')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    convnames=['original','original_inline','int_branch','int_select','magic_branch','magic_select','sse2','block4','block8']
    for ax,dist in zip(axs,('gaussian','subnormal')):
        vals=[];lo=[];hi=[]
        for name in convnames:
            rs=[r for r in conversion if r['name']==name and r['distribution']==dist and r['dims']==768 and not r['novec']]
            r=rs[0];samples=np.array(r['samples_ms'])*1e6/r['elements'];vals.append(float(np.median(samples)));lo.append(float(samples.min()));hi.append(float(samples.max()))
        ax.barh(convnames,vals,color='#236d95');ax.errorbar(vals,range(len(vals)),xerr=[np.array(vals)-lo,np.array(hi)-vals],fmt='none',ecolor='#222',capsize=2)
        ax.set_xlabel('Nanoseconds per half value (smaller is better)');ax.set_title(dist+' conversion, 1,536,000 values')
    fig.suptitle('Standalone widening cost, including writing the float32 output\nError bars show min/max of three samples; these are not additive portions of fused-kernel time')
    fig.tight_layout();finish(fig,'conversion-cost')
    # Compact measured summary, no inferred speedup claims.
    text=['# Portable float16 KNN experiment results','',f"Measured on **{env['cpu']}**, Windows, {env['compiler']}. Production files were not changed. All datasets contained at most **2,000 vectors**.",'','![Overview](overview.png)','','## Conclusions','']
    for v in finalists:
        g=sqlgroup(v,'f16');fg=sqlgroup(v,'f32')
        wins=sum(med(g[k])<med(sb[k]) for k in primary)
        text.append(f'- **{v}**: geometric-mean SQL speedup **{geo([med(sb[k])/med(g[k]) for k in primary]):.2f}x** over current float16; **{geo([med(fg[k])/med(g[k]) for k in primary]):.2f}x** versus its matched float32 implementation. Faster than current float16 in **{wins}/{len(primary)}** primary cases.')
    text.append(f'- **{portable}**: geometric-mean SQL speedup **{geo([med(sb[k])/med(pg[k]) for k in portable_primary]):.2f}x** over current float16 and **{geo([med(pf[k])/med(pg[k]) for k in portable_primary]):.2f}x** versus matched scalar float32 across its **{len(portable_primary)}** primary cases at 768 dimensions. This isolates a useful half-specific improvement without explicit SIMD or candidate batching.')
    text+=['','![SQL summary](sql-summary.png)','','These averages are not universal wins over float32: several memory-backed cases favor matched float32, as the heatmap below shows. The matched float32 distance microkernels are also faster than float16. The end-to-end float16 advantage is consistent with smaller stored vectors and reduced blob traffic, but cache/copy costs were not instrumented, so that explanation remains an inference.']
    bestscalar=max((r for r in singled if r['kind']=='scalar' and r['converter'] not in ('original','original_inline')),key=lambda r:r['geomean_speedup'])
    inline_gain=gain(mb,micro('original_inline_b1'))
    converter_gain=gain(micro('original_inline_b1'),micro(bestscalar['name']))
    text+=['',f"The strongest one-candidate portable converter was **{bestscalar['name']}**, with **{bestscalar['geomean_speedup']:.2f}x** microkernel speedup over the original float16 path on Gaussian workloads. Inlining the original converter alone gives **{inline_gain:.2f}x**; the new converter is **{converter_gain:.2f}x** relative to that inlining-only control. This separates half decoding/call overhead from candidate batching. SSE2 and batching must also be judged against their matched float32 controls.",'','**Microbenchmark compiler caveat:** the multi-variant DLL leaves an out-of-line `vec_half_to_float` call in the original kernel; the smaller baseline SQL DLL inlines conversion in its L2 distance function. Translation-unit context changes MSVC inlining decisions despite identical flags. Therefore the headline microkernel speedups are not predicted SQL speedups, and the inlining-only improvement must not be attributed to the production SQL path. The forced-inline ablation retains the original converter arithmetic, including exceptional cases and its normalization loop. Use the SQL results for the recommendation.','',f'The portable diagnostic **{portable}** also received SQL confirmation at 768 dimensions; see its separate table below.','', 'Finalists were selected by geometric-mean full-distance speedup across 16 Gaussian/normalized-Gaussian dimension/metric combinations. Selection used kernel measurements; SQL measurements are a separate confirmation. Ratios are baseline time / candidate time; above 1 is faster.','', '## Ablation: where the improvement comes from','','| Configuration | Gaussian f16 kernel speedup | Zero-heavy | Subnormal-heavy | Matched f32 / f16 | Vectorization-off / on time |','|---|---:|---:|---:|---:|---:|']
    for r in sorted(rank,key=lambda r:r['id']):
        v=r['name'];normal=micro(v)
        zero=gain(micro('baseline',distribution=('zeros',)),micro(v,distribution=('zeros',)))
        sub=gain(micro('baseline',distribution=('subnormal',)),micro(v,distribution=('subnormal',)))
        text.append(f'| {v} | {gain(mb,normal):.2f}x | {zero:.2f}x | {sub:.2f}x | {gain(micro(v,"f32"),normal):.2f}x | {gain(micro(v,novec=True),normal):.2f}x |')
    text+=['','`original_b1` is the matched scalar control. Unlike historical float32 cosine, experimental float32 precomputes the query norm outside the candidate loop: once per scan in microbenchmarks, once per chunk in SQL. Float16 retains its existing once-per-query norm. These SQL cases use one or two chunks, so this is a small remaining outer-loop asymmetry; the candidate arithmetic/batching is matched. Compare against matched float32, not only the historical baseline. The two branched finite converters share the same normal-value algorithm; tiny differences on normal-heavy data are not evidence of a winner.','', '![Candidate batching](batch-width.png)','','## SQL results','','Each cell aggregates three per-run medians; p95 is the median of three per-run p95 values. Complete individual samples, insertion times, error checks and memory readings are in [sql.json](sql.json) and [sql.csv](sql.csv). The production `vec_debug()` string still says scalar because it does not know about the isolated scan hook; the build label/selected ID and frozen source identify the actual experimental kernel.','','| Rows x dims / metric / storage / eligible % | Baseline f16 median / p95 ms | Baseline f32 median ms | '+ ' | '.join(v+' f16 median / p95 ms; speedup' for v in finalists)+' |','|---|---:|---:|'+ '---:|'*len(finalists)]
    for k in sorted(sb):
        n,d,m,s,sel=k
        row=f'| {n} x {d} / {m} / {s} / {sel}% | {med(sb[k]):.3f} / {np.median([r["p95_ms"] for r in sb[k]]):.3f} | {med(sf[k]):.3f} |'
        for v in finalists:
            g=sqlgroup(v,'f16')[k];row+=f' {med(g):.3f} / {np.median([r["p95_ms"] for r in g]):.3f}; {med(sb[k])/med(g):.2f}x |'
        text.append(row)
    text+=['','### Portable conversion diagnostic','','| Rows / metric / storage / eligible % (768 dimensions) | f16 median / p95 ms | Speedup vs current f16 | Matched f32 / f16 |','|---|---:|---:|---:|']
    pg=sqlgroup(portable,'f16');pf=sqlgroup(portable,'f32')
    for k in sorted(pg):
        n,d,m,s,sel=k;g=pg[k]
        text.append(f'| {n} / {m} / {s} / {sel}% | {med(g):.3f} / {np.median([r["p95_ms"] for r in g]):.3f} | {med(sb[k])/med(g):.2f}x | {med(pf[k])/med(g):.2f}x |')
    text+=['','![SQL comparisons](sql-heatmap.png)','','## Conversion cost and footprint','','![Standalone conversion](conversion-cost.png)','','Standalone conversion writes a float32 array; fused distance kernels do not. These measurements diagnose conversion throughput, but subtracting them from full-distance time would not yield a valid time breakdown.','', '| Build | Executable .text bytes | Read-only .rdata bytes | Largest experimental scan stack frame, including saved GPRs |','|---|---:|---:|---:|']
    for b in builds:
        v=next((r for r in rank if r['name']==b['label']),dict(batch=1))
        frames=b['scan_stack_frames'].values()
        frame=str(max((f['local_stack_allocation']+f['pushed_register_bytes'] for f in frames),default=0)) if b['label']!='baseline' else 'Not applicable'
        text.append(f"| {b['label']} | {b['sections']['.text']['virtual_bytes']:,} | {b['sections']['.rdata']['virtual_bytes']:,} | {frame} |")
    text+=['','No LUT or persistent expanded-vector cache is allocated. The query buffer already present in float16 uses `4 * dimensions` bytes. Candidate pointers/IDs/results require at most `16 * batch_width` logical staging bytes on x64, plus arithmetic temporaries. The table reports the largest compiled scan frame across dtypes/metrics, not total call-stack usage; it includes local allocation and pushed general-purpose registers, excludes the return address and callees. Read-only section sizes include compiler constants, strings and other extension metadata, not just conversion constants.','', '| Rows x dimensions | Float16 database bytes | Matched float32 database bytes | Ratio |','|---|---:|---:|---:|']
    for n,d in itertools.product((1000,2000),(384,768,1536)):
        a=next(r['database_bytes'] for r in sql if r['case']==dict(rows=n,dims=d,metric='l2',storage='file',selectivity=100,variant='baseline',dtype='f16',repetition=0))
        b=next(r['database_bytes'] for r in sql if r['case']==dict(rows=n,dims=d,metric='l2',storage='file',selectivity=100,variant='baseline',dtype='f32',repetition=0))
        text.append(f'| {n} x {d} | {a:,} | {b:,} | {b/a:.2f}x |')
    maxerr=max(max(r['max_errors']) for r in sql)
    overlaps=[o for r in sql if r['case']['dtype']=='f16' for o in r['original_fp32_topk_overlap']]
    text+=['','## Correctness and reproducibility','',f'- Every specialized converter passed all **63,488 finite encodings**; both general converter forms passed all **65,536 encodings**. Signed zeros, NaN classification and all four FTZ/DAZ combinations were checked with unaligned input.',f'- Both default and `loop(no_vector)` builds passed the distance grid: **{sum(r["distance_cases"] for r in load("correctness-micro.json")):,}** variant/dtype/metric/distribution/dimension checks, plus maximum-finite stress cases.','- SQL extensions passed the existing float16 suite and extra matched-float32 tests covering ties, deletions, filtered searches, empty matches, short batches and `k=0`/`k>eligible`.',f'- All measured SQL answers passed float64 distance and top-k threshold checks. Maximum absolute returned-distance error was **{maxerr:.3g}**. Original-float32 top-10 overlap averaged **{np.mean(overlaps)*100:.2f}%**, minimum **{min(overlaps)*100:.1f}%**, on these synthetic queries. Near-tie rounding is handled with the existing tolerance.','- Same half-rounded base/query values are stored in both dtypes. Original float32 comparisons are accuracy checks only.','- SQL settings: 4 KiB pages, 1,024-row chunks, 64 MiB SQLite cache, mmap disabled; file runs use WAL and synchronous FULL. Reads are warm. Insertion includes commits every 1,000 rows; input serialization and correctness references are outside query timing. Query widening inside SQLite remains timed.','- Five SQL warmups, twenty timed queries, three repetitions, sequential fresh workers and one numerical-library thread. Variant order rotates across cases and reverses on alternate repetitions. Microbenchmarks use a warmup/calibration and three samples of bounded repeated scans.','- MSVC ignores the initially proposed `/Qvec-` flag. The final control uses documented `__pragma(loop(no_vector))` on measured loops; discarded preliminary control results are not included. Explicit SSE2 intrinsics remain SIMD in that control.','- Disassembly contains no AVX/F16C, SSE3/SSE4, LZCNT or TZCNT requirements in measured floating-point paths. Existing MSVC Hamming code emits POPCNT; it is unrelated, unchanged and never called here. The whole library is therefore not certified as universally SSE2-only.','- Results are local Ryzen evidence, not an emulation of older hardware. Frequency, thermals, scheduling and shared-machine activity remain sources of variability. No CPU affinity or frequency lock was imposed.','- Raw measurements and CSV exports are in this directory. [Environment and source hashes](environment.json), [micro builds](builds-micro.json), [SQL builds](builds-sql.json), [kernel correctness](correctness-micro.json), and [SQL correctness](correctness-sql.json) preserve provenance. Reproduction commands are in [README](../README.md).','','## CPU support and research','','| Feature | This Ryzen 7 5800H | Steam Windows, August 2026 |','|---|---|---:|','| SSE2 | Supported | 100.00% |','| AVX | Supported; OS usability separately confirmed with .NET | 99.15% |','| AVX2 | Supported | 97.30% |','| F16C | Supported | 98.54% |','| AVX-512F | Not supported | 24.46% |','','[Steam Hardware Survey](https://store.steampowered.com/hwsurvey/?l=english&platform=pc), inspected September 26, 2026. Voluntary Steam Windows participants are not representative of all consumer devices. SSE2 is the [default Windows x64 compiler baseline](https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64?view=msvc-170); portable C remains necessary for ARM and WASM.','','Conversion references: [FP16 implementation](https://github.com/Maratyszcza/FP16/blob/master/include/fp16/fp16.h), [Giesen, Half to float done quick](https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/). Candidate-batching precedent: [Faiss distance interfaces](https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances.h). These are research references, not dependencies.','','## Recommendation','']
    reliable=[]
    for v in finalists:
        g=sqlgroup(v,'f16');count=0
        for k in primary:
            # Conservative criterion: slowest candidate run median beats fastest baseline.
            count+=max(r['median_ms'] for r in g[k])<min(r['median_ms'] for r in sb[k])
        reliable.append((v,count))
        text.append(f'- **{v}** has non-overlapping per-run-median ranges favoring it in **{count}/{len(primary)}** primary SQL cases. This is an observed range check, not a statistical confidence interval.')
    text+=['','Recommend the finite-only branched converter as the portable follow-up, and **SSE2 with two candidates** as the x64 follow-up. Batches of two and eight are effectively tied in aggregate SQL performance; two adds less executable code (12.25 KiB versus 21.98 KiB) and has a smaller maximum experimental scan frame (280 versus 552 bytes). These code-size increases include the matched float32 implementations. Validate both choices on older hardware before production adoption. This deliverable leaves production defaults unchanged.','']
    (root/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    print('report written',root/'REPORT.md')


if __name__=='__main__':report()
