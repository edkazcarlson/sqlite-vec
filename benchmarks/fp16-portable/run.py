"""Bounded, Windows-native float16 experiments. See README.md for commands."""
import argparse
import ctypes as C
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / 'build'
RESULTS = HERE / 'results'
sys.path.insert(0, str(BUILD / 'packages'))
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR=str(BUILD / 'mpl-cache'))
import numpy as np
from kernels import VARIANTS, CONVERTERS, generate, exports

CAP = 2000
METRICS = {'l2':1, 'cosine':2, 'l1':3}
VCVARS = Path(os.environ.get('VEC_VCVARS', r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
FLAGS = ['/nologo','/O2','/fp:precise','/MD','/LD','/std:c11','/Gy','/utf-8']


def cap(rows):
    if not 1 <= rows <= CAP:
        raise ValueError(f'Experiment limit: 1 <= rows <= {CAP}; got {rows}')


def save(name, value):
    RESULTS.mkdir(parents=True, exist_ok=True)
    dest = RESULTS / name
    temp = dest.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    # Windows readers/indexers can briefly hold the destination without sharing
    # delete access. Preserve atomic checkpoints and retry only this operation.
    for attempt in range(9):
        try:
            temp.replace(dest)
            break
        except PermissionError:
            if attempt==8:raise
            time.sleep(min(0.02*2**attempt,1))


def read(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vsenv():
    p = subprocess.run(f'"{VCVARS}" >nul && set', shell=True, capture_output=True, text=True, check=True)
    env = {k.upper():v for k,v in os.environ.items()}
    for line in p.stdout.splitlines():
        if '=' in line:
            k,v = line.split('=',1)
            env[k.upper()]=v
    return env


def prepare():
    BUILD.mkdir(parents=True, exist_ok=True)
    src = BUILD/'source'
    src.mkdir(exist_ok=True)
    archive = BUILD/'sqlite-amalgamation-3450300.zip'
    if not archive.exists():
        urllib.request.urlretrieve('https://www.sqlite.org/2024/sqlite-amalgamation-3450300.zip',archive)
    with zipfile.ZipFile(archive) as z:
        for name in ('sqlite3.h','sqlite3ext.h'):
            (src/name).write_bytes(z.read('sqlite-amalgamation-3450300/'+name))
    for path in ROOT.glob('sqlite-vec*.c'):
        shutil.copy2(path,src/path.name)
    version = (ROOT/'VERSION').read_text(encoding='utf-8').strip()
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    header = (ROOT/'sqlite-vec.h.tmpl').read_text(encoding='utf-8')
    for key,val in dict(VERSION=version,DATE='experiment',SOURCE=commit,VERSION_MAJOR='0',VERSION_MINOR='1',VERSION_PATCH='10').items():
        header=header.replace('${'+key+'}',val)
    (src/'sqlite-vec.h').write_text(header,encoding='utf-8')
    return src


def compile_dll(label, selected=None, novec=False):
    src=BUILD/'source'
    dest=BUILD/label
    dest.mkdir(exist_ok=True)
    original=(src/'sqlite-vec.c').read_text(encoding='utf-8')
    if selected is None:
        experimental=generate()+'\n'+exports()
        if novec:
            experimental=re.sub(r'\bfor\s*\(', '__pragma(loop(no_vector)) for(', experimental)
            # Suppress loop vectorization in original measured kernels too.
            halfsource=(src/'sqlite-vec-fp16.c').read_text(encoding='utf-8')
            halfsource=re.sub(r'\bfor\s*\(', '__pragma(loop(no_vector)) for(',halfsource)
            (dest/'sqlite-vec-fp16-novec.c').write_text(halfsource,encoding='utf-8')
            original=original.replace('#include "sqlite-vec-fp16.c"','#include "sqlite-vec-fp16-novec.c"')
            for fn in ('l2_sqr_float','distance_l1_f32','distance_cosine_float'):
                match=re.search(r'\b'+fn+r'\([^;]*?\)\s*\{',original)
                assert match,fn
                lo=match.start();hi=match.end();depth=1
                while depth:
                    depth+=(original[hi]=='{')-(original[hi]=='}');hi+=1
                body=original[lo:hi]
                body=re.sub(r'\b(for|while)\s*\(',r'__pragma(loop(no_vector)) \1(',body)
                original=original[:lo]+body+original[hi:]
        source=original+'\n'+experimental
    elif selected == 0:
        source=original
    else:
        source=original.replace('int vec0Filter_knn_chunks_iter(',generate(selected)+'\nint vec0Filter_knn_chunks_iter(',1)
        start=source.index('int vec0Filter_knn_chunks_iter(')
        end=source.index('static int vec0Filter_knn_diskann',start)
        section=source[start:end]
        marker='    for (int i = 0; i < p->chunk_size; i++) {\n      if (!bitmap_get(b, i)) {'
        assert section.count(marker)==1
        hook=f'''    if(vector_column->element_type==SQLITE_VEC_ELEMENT_TYPE_FLOAT16 || vector_column->element_type==SQLITE_VEC_ELEMENT_TYPE_FLOAT32) {{
      int ishalf=vector_column->element_type==SQLITE_VEC_ELEMENT_TYPE_FLOAT16;
      const float*q=ishalf?halfQuery:(const float*)queryVector;
      float qn=vector_column->distance_metric==VEC0_DISTANCE_METRIC_COSINE?(ishalf?halfQueryNorm:vec_query_norm(q,vector_column->dimensions)):0;
      exp_run_{selected}(ishalf,vector_column->distance_metric,baseVectors,q,vector_column->dimensions,p->chunk_size,b,qn,chunk_distances);
    }}
'''
        section=section.replace(marker,hook+marker)
        lo=section.index('      case SQLITE_VEC_ELEMENT_TYPE_FLOAT16: {',section.index(hook))
        hi=section.index('      case SQLITE_VEC_ELEMENT_TYPE_INT8: {',lo)
        section=section[:lo]+'''      case SQLITE_VEC_ELEMENT_TYPE_FLOAT16:
      case SQLITE_VEC_ELEMENT_TYPE_FLOAT32:
        result=chunk_distances[i];break;
'''+section[hi:]
        source=source[:start]+section+source[end:]
    cfile=dest/'experiment.c'
    cfile.write_text(source,encoding='utf-8')
    args=['cl.exe',*FLAGS]
    args+=['/Qvec-report:2']
    args += ['/I'+str(src),'/Fo'+str(dest/'experiment.obj'),'/Fe'+str(dest/'vec0.dll'),'/FAcs','/Fa'+str(dest/'assembly.asm'),str(cfile),'/link','/OPT:REF','/OPT:ICF']
    env=vsenv()
    args[0]=str(Path(env['VCTOOLSINSTALLDIR'])/'bin/Hostx64/x64/cl.exe')
    done=subprocess.run(args,env=env,cwd=dest,text=True,capture_output=True)
    (dest/'compile.log').write_text(done.stdout+done.stderr)
    assert 'D9002' not in done.stdout+done.stderr,'Compiler ignored an option'
    if done.returncode:
        raise RuntimeError((done.stdout+done.stderr)[-12000:])
    dump=subprocess.run([str(Path(env['VCTOOLSINSTALLDIR'])/'bin/Hostx64/x64/dumpbin.exe'),'/disasm',str(dest/'vec0.dll')],env=env,capture_output=True,text=True,check=True).stdout
    (dest/'disassembly.txt').write_text(dump)
    # Match actual disassembly lines, not identifiers or raw byte values.
    ins=[]
    for line in dump.splitlines():
        match=re.match(r'\s*[0-9A-Fa-f]{8,16}:\s+(?:(?:[0-9A-Fa-f]{2})\s+)+([a-z][a-z0-9]*)\b',line)
        if match:
            ins.append(match.group(1))
    forbidden=sorted({op for op in ins if op.startswith('v') or op in {'lzcnt','tzcnt','popcnt','pblendw','blendps','blendvps','pblendvb','ptest','pmulld','roundps','dpps','pshufb','haddps','hsubps','movddup','movshdup','movsldup','insertps','extractps','phaddw','phaddd','pmaddubsw'}})
    assert ins, 'Disassembler parser found no instructions'
    # Existing MSVC binary-vector Hamming code explicitly uses __popcnt64.
    # It is outside the float16/float32 paths and is not modified here.
    assert not set(forbidden)-{'popcnt'}, forbidden
    asm=(dest/'assembly.asm').read_text(encoding='utf-8',errors='replace')
    frames={}
    for fn,body in re.findall(r'^(\w+)\s+PROC[^\n]*\n(.*?)^\1\s+ENDP',asm,re.M|re.S):
        if fn.startswith(('exp_','l2_sqr_float','l1_float16','cosine_float16','distance_cosine_float','distance_l1_f32')):
            assert not re.search(r'\t(?:popcnt|lzcnt|tzcnt)\s',body),fn
        if fn.startswith('exp_scan_'):
            allocations=[int(n) for n in re.findall(r'\bsub\s+rsp,\s*(\d+)',body)]
            pushes=len(re.findall(r'\bpush\s+r\w+',body))
            frames[fn]=dict(local_stack_allocation=max(allocations,default=0),pushed_register_bytes=pushes*8,conversion_calls=len(re.findall(r'\bcall\s+vec_half_to_float\b',body)))
    binary=(dest/'vec0.dll').read_bytes()
    pe=struct.unpack_from('<I',binary,0x3c)[0]
    nsects=struct.unpack_from('<H',binary,pe+6)[0]
    opt=struct.unpack_from('<H',binary,pe+20)[0]
    sections={}
    for i in range(nsects):
        offset=pe+24+opt+40*i
        name=binary[offset:offset+8].split(b'\0')[0].decode()
        virtual,_,raw=struct.unpack_from('<III',binary,offset+8)
        sections[name]={'virtual_bytes':virtual,'file_bytes':raw}
    info=dict(label=label,selected=selected,novec=novec,vectorization_control='loop(no_vector) on measured loops' if novec else 'default',command=args,source_sha256=sha(cfile),binary_sha256=sha(dest/'vec0.dll'),sections=sections,scan_stack_frames=frames,instruction_count=len(ins),instruction_mnemonics=sorted(set(ins)),forbidden_in_float_paths=[],existing_unrelated_instructions=forbidden)
    (dest/'build.json').write_text(json.dumps(info,indent=2))
    print('built',label,'instructions',len(ins),flush=True)
    return info


def library(novec=False):
    lib=C.CDLL(str(BUILD/('micro_novec' if novec else 'micro')/'vec0.dll'))
    lib.exp_distance.argtypes=[C.c_int,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p,C.c_void_p,C.c_int]
    lib.exp_convert.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
    lib.exp_mxcsr.argtypes=[C.c_uint,C.c_int]
    lib.exp_mxcsr.restype=C.c_uint
    lib.exp_cpuid.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_int)]
    return lib


def ptr(a):
    return None if a is None else a.ctypes.data


def native_distance(lib,vid,half,metric,x,q,mask=None,repeat=1):
    cap(len(x))
    out=np.full(len(x),np.nan,dtype=np.float32)
    lib.exp_distance(vid,half,METRICS[metric],ptr(x),ptr(q),x.shape[1],len(x),ptr(mask),ptr(out),repeat)
    return out


def reference(x,q,metric):
    x=x.astype(np.float64);q=q.astype(np.float64)
    if metric=='l2':return np.sqrt(np.sum((x-q)**2,axis=1))
    if metric=='l1':return np.sum(np.abs(x-q),axis=1)
    return 1-np.sum(x*q,axis=1)/(np.sqrt(np.sum(x*x,axis=1))*np.sqrt(np.sum(q*q)))


def dataset(rows,dims,distribution='gaussian',seed=20260926,queries=1):
    cap(rows)
    rng=np.random.default_rng(seed)
    original=rng.standard_normal((rows,dims),dtype=np.float32)
    oq=rng.standard_normal((queries,dims),dtype=np.float32)
    if distribution=='normalized':
        original/=np.linalg.norm(original,axis=1,keepdims=True)
        oq/=np.linalg.norm(oq,axis=1,keepdims=True)
    elif distribution=='zeros':
        original[rng.random(original.shape)<0.8]=0
        oq[rng.random(oq.shape)<0.8]=0
        original[:,0]=1;oq[:,0]=1
    elif distribution=='subnormal':
        # All values nonzero; mixed signs and the full subnormal mantissa range.
        original=(rng.integers(1,1024,size=original.shape)*rng.choice([-1,1],size=original.shape)*2.0**-24).astype(np.float32)
        oq=(rng.integers(1,1024,size=oq.shape)*rng.choice([-1,1],size=oq.shape)*2.0**-24).astype(np.float32)
    h=original.astype(np.float16);qh=oq.astype(np.float16)
    return h,h.astype(np.float32),qh,qh.astype(np.float32),original,oq


def verify_micro():
    results=[]
    for novec in (False,True):
        lib=library(novec)
        bits=np.arange(65536,dtype=np.uint16)
        finite=(bits&0x7c00)!=0x7c00
        for state in (0,0x40,0x8000,0x8040):
            old=lib.exp_mxcsr(0,0)
            try:
                lib.exp_mxcsr((old&~0x8040)|state,1)
                for converter in range(9):
                    data=bits if converter in (0,8) else bits[finite]
                    # Exercise deliberately unaligned input as well as final tails.
                    raw=np.zeros(data.nbytes+1,dtype=np.uint8)
                    raw[1:]=data.view(np.uint8)
                    out=np.empty(len(data),dtype=np.float32)
                    lib.exp_convert(converter,raw.ctypes.data+1,ptr(out),len(data),1)
                    expected=data.view(np.float16).astype(np.float32)
                    good=~np.isnan(expected)
                    np.testing.assert_array_equal(out[good].view(np.uint32),expected[good].view(np.uint32))
                    assert np.isnan(out[~good]).all()
            finally:
                lib.exp_mxcsr(old,1)
        checks=0
        for dims in (1,3,7,8,15,16,17,31,33,128,384,769,1536):
            for dist in ('gaussian','zeros','subnormal'):
                h,f,qh,q,_,_=dataset(19,dims,dist)
                mask=np.packbits(np.arange(19)%3!=1,bitorder='little')
                active=np.arange(19)%3!=1
                for half,x in ((1,h),(0,f)):
                    raw=np.zeros(x.nbytes+1,dtype=np.uint8);raw[1:]=x.view(np.uint8).reshape(-1)
                    unaligned=np.ndarray(x.shape,dtype=x.dtype,buffer=raw,offset=1)
                    for metric in METRICS:
                        expected=reference(f,q[0],metric)
                        for v in VARIANTS:
                            got=native_distance(lib,v['id'],half,metric,unaligned,q[0],mask)
                            np.testing.assert_allclose(got[active],expected[active],rtol=2e-5,atol=2e-5,err_msg=str((novec,dims,dist,half,metric,v['name'])))
                            assert np.isnan(got[~active]).all()
                            checks+=1
        # Maximum finite values and cancellation mixed with signed zero/subnormals.
        h=np.tile(np.array([65504,-65504,0,-0.0,2**-24,-2**-24],dtype=np.float16),(11,1))
        q=np.array([-65504,65504,0,-0.0,-2**-24,2**-24],dtype=np.float32)
        for metric in METRICS:
            expected=reference(h,q,metric)
            for v in VARIANTS:
                np.testing.assert_allclose(native_distance(lib,v['id'],1,metric,h,q),expected,rtol=2e-5,atol=2e-5)
        results.append(dict(novec=novec,finite_patterns=63488,general_patterns=65536,mxcsr_modes=4,distance_cases=checks,status='passed'))
        print('verified micro',novec,checks,flush=True)
    save('correctness-micro.json',results)


def timing(call,repetitions=3):
    call(1)
    start=time.perf_counter_ns();call(1);duration=(time.perf_counter_ns()-start)/1e9
    repeat=max(1,min(16,math.ceil(0.006/max(duration,1e-6))))
    samples=[]
    for _ in range(repetitions):
        start=time.perf_counter_ns();call(repeat);samples.append((time.perf_counter_ns()-start)/repeat/1e6)
    return samples,repeat


def screen(novec_only=False):
    records=[r for r in read('screen.json') if not r['novec']] if novec_only else []
    conversions=[r for r in read('conversion.json') if not r['novec']] if novec_only else []
    rng=np.random.default_rng(19)
    for novec in ((True,) if novec_only else (False,True)):
        lib=library(novec)
        for distribution,dims in itertools.product(('gaussian','normalized','zeros','subnormal'),(128,384,768,1536)):
            h,f,qh,q,_,_=dataset(CAP,dims,distribution)
            for metric in ('l2','cosine'):
                order=rng.permutation(len(VARIANTS))
                for vid in order:
                    v=VARIANTS[int(vid)]
                    for half,x in ((1,h),(0,f)):
                        out=np.empty(CAP,dtype=np.float32)
                        call=lambda repeat:lib.exp_distance(v['id'],half,METRICS[metric],ptr(x),ptr(q[0]),dims,CAP,None,ptr(out),repeat)
                        samples,repeat=timing(call)
                        np.testing.assert_allclose(out,reference(f,q[0],metric),rtol=2e-5,atol=2e-5)
                        records.append(dict(variant=v['name'],id=v['id'],novec=novec,dtype='f16' if half else 'f32',distribution=distribution,rows=CAP,dims=dims,metric=metric,samples_ms=samples,median_ms=float(np.median(samples)),inner_repeats=repeat))
                print('screen',novec,distribution,dims,metric,flush=True)
            flat=h.reshape(-1);out=np.empty(flat.size,dtype=np.float32)
            for converter in rng.permutation(9):
                samples,repeat=timing(lambda repeat:lib.exp_convert(int(converter),ptr(flat),ptr(out),flat.size,repeat))
                conversions.append(dict(converter=int(converter),name=(CONVERTERS+['sse2','block4','block8','original_inline'])[int(converter)],novec=novec,distribution=distribution,dims=dims,elements=int(flat.size),samples_ms=samples,median_ms=float(np.median(samples)),inner_repeats=repeat))
            save('screen.json',records);save('conversion.json',conversions)
    baseline={(r['distribution'],r['dims'],r['metric']):r['median_ms'] for r in records if r['variant']=='baseline' and not r['novec'] and r['dtype']=='f16'}
    ranking=[]
    for v in VARIANTS[1:]:
        rs=[r for r in records if r['id']==v['id'] and not r['novec'] and r['dtype']=='f16' and r['distribution'] in ('gaussian','normalized')]
        speedups=[baseline[(r['distribution'],r['dims'],r['metric'])]/r['median_ms'] for r in rs]
        ranking.append(dict(**v,geomean_speedup=math.exp(np.mean(np.log(speedups)))))
    ranking.sort(key=lambda v:v['geomean_speedup'],reverse=True)
    save('ranking.json',ranking)
    print('finalists',[(v['name'],v['geomean_speedup']) for v in ranking[:2]],flush=True)


def connect(path,extension):
    db=sqlite3.connect(str(path))
    db.enable_load_extension(True)
    db.load_extension(str(extension),entrypoint='sqlite3_vec_init')
    db.enable_load_extension(False)
    db.execute('pragma page_size=4096');db.execute('pragma cache_size=-65536');db.execute('pragma mmap_size=0')
    db.execute('pragma synchronous=FULL')
    if str(path)!=':memory:':db.execute('pragma journal_mode=WAL')
    return db


def rss():
    class Counters(C.Structure):
        _fields_=[('cb',C.c_ulong),('PageFaultCount',C.c_ulong)]+[(name,C.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters=Counters();counters.cb=C.sizeof(counters)
    kernel=C.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=C.c_void_p
    psapi=C.WinDLL('psapi',use_last_error=True);psapi.GetProcessMemoryInfo.argtypes=[C.c_void_p,C.POINTER(Counters),C.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),C.byref(counters),C.sizeof(counters)):raise C.WinError(C.get_last_error())
    return int(counters.PeakWorkingSetSize)


def sql_worker(case):
    n,d=case['rows'],case['dims'];cap(n)
    h,f,qh,q,original,oq=dataset(n,d,'gaussian',queries=25)
    half=case['dtype']=='f16';data=h if half else f;queries=qh if half else q
    work=RESULTS/'work';work.mkdir(exist_ok=True)
    path=':memory:' if case['storage']=='memory' else work/f'{os.getpid()}.db'
    db=connect(path,BUILD/case['variant']/'vec0.dll')
    column='float16' if half else 'float';wrap='vec_f16(?)' if half else '?'
    db.execute(f"create virtual table v using vec0(e {column}[{d}] distance_metric={case['metric']}, category integer, chunk_size=1024)")
    blobs=[row.tobytes() for row in data];qb=[row.tobytes() for row in queries]
    start=time.perf_counter_ns()
    for first in range(0,n,1000):
        with db:db.executemany(f'insert into v(rowid,e,category) values (?,{wrap},?)',[(i+1,blobs[i],i%100) for i in range(first,min(first+1000,n))])
    insert_ms=(time.perf_counter_ns()-start)/1e6
    db.execute('pragma wal_checkpoint(TRUNCATE)').fetchall()
    assert db.execute('select count(*) from v').fetchone()[0]==n
    selectivity=case.get('selectivity',100)
    pred='' if selectivity==100 else f' and category<{selectivity}'
    sql=f'select rowid,distance from v where e match {wrap} and k=10{pred} order by distance'
    for query in qb[:5]:db.execute(sql,(query,)).fetchall()
    samples=[];answers=[]
    for query in qb[5:]:
        start=time.perf_counter_ns();rows=db.execute(sql,(query,)).fetchall();samples.append((time.perf_counter_ns()-start)/1e6);answers.append(rows)
    allowed=np.flatnonzero(np.arange(n)%100<selectivity)
    errors=[];overlap=[];exact=[]
    for i,answer in enumerate(answers):
        ref=reference(f,q[i+5],case['metric'])
        ids=np.array([r[0]-1 for r in answer]);values=np.array([r[1] for r in answer])
        assert len(ids)==min(10,len(allowed)) and set(ids)<=set(allowed)
        np.testing.assert_allclose(values,ref[ids],rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(values,np.sort(ref[allowed])[:len(ids)],rtol=2e-5,atol=2e-5)
        errors.append(float(np.max(abs(values-ref[ids]),initial=0)))
        exact.append(len(set(ids)&set(allowed[np.argsort(ref[allowed])[:10]]))/max(1,len(ids)))
        orig=reference(original,oq[i+5],case['metric'])
        overlap.append(len(set(ids)&set(allowed[np.argsort(orig[allowed])[:10]]))/max(1,len(ids)))
    size=db.execute('pragma page_count').fetchone()[0]*4096
    settings={p:db.execute('pragma '+p).fetchall() for p in ('page_size','cache_size','mmap_size','journal_mode','synchronous')}
    debug=db.execute('select vec_debug()').fetchone()[0]
    db.close()
    file_bytes=None
    if path!=':memory:':
        file_bytes=path.stat().st_size;path.unlink()
    return dict(case=case,samples_ms=samples,median_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),insert_ms=insert_ms,database_bytes=size,file_bytes=file_bytes,peak_process_rss_bytes=rss(),max_errors=errors,stored_topk_overlap=exact,original_fp32_topk_overlap=overlap,settings=settings,debug=debug)


def verify_sql(labels):
    results=[]
    for label in labels:
        extension=BUILD/label/'vec0.dll'
        env=dict(os.environ,VEC_TEST_EXTENSION=str(extension),PYTHONPATH=str(BUILD/'packages'))
        # This suite owns all its fixtures. Repository conftest unconditionally
        # opens dist/vec0, which is not one of our isolated experimental DLLs.
        p=subprocess.run([sys.executable,'-m','pytest',str(ROOT/'tests/test-fp16.py'),'-q','--noconftest','--basetemp',str(BUILD/('pytest-'+label)),'-o','cache_dir='+str(BUILD/'pytest-cache')],env=env,text=True,capture_output=True,cwd=ROOT)
        (RESULTS/f'{label}-tests.log').write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(p.stdout+p.stderr)
        # Existing tests cover rowid filters, partitions, mutations, reopen,
        # all conversion patterns, near-boundary dimensions and invalid input.
        db=connect(':memory:',extension)
        for metric in METRICS:
            for half in (False,True):
                typ='float16' if half else 'float';wrap='vec_f16(?)' if half else '?'
                db.execute(f'create virtual table v using vec0(e {typ}[7] distance_metric={metric}, category integer, chunk_size=8)')
                h,f,qh,q,_,_=dataset(19,7)
                f[1]=f[0];h[1]=h[0] # exact ties
                a=h if half else f;query=(qh if half else q)[0].tobytes()
                db.executemany(f'insert into v(rowid,e,category) values(?,{wrap},?)',[(i+1,row.tobytes(),i%2) for i,row in enumerate(a)])
                db.execute('delete from v where rowid in (3,7,15)')
                ref=reference(a,q[0],metric)
                for predicate,allowed in [('',[i for i in range(19) if i not in (2,6,14)]),('and category=1',[i for i in range(19) if i%2 and i not in (2,6,14)]),('and rowid in (1,2,4,19)',[0,1,3,18]),('and category=99',[])]:
                    for k in (0,1,4,100):
                        rows=db.execute(f'select rowid,distance from v where e match {wrap} and k=? {predicate} order by distance',(query,k)).fetchall()
                        np.testing.assert_allclose([r[1] for r in rows],np.sort(ref[allowed])[:k],rtol=2e-5,atol=2e-5)
                db.execute('drop table v')
        db.close()
        results.append(dict(label=label,pytest=p.stdout.strip(),additional_sql='ties, deleted rows, partial chunks, filters, k=0/k>eligible: passed'))
        print('verified SQL',label,flush=True)
    save('correctness-sql.json',results)


def sql_suite(resume=False):
    selected=read('ranking.json')[:2]
    portable=next(v for v in read('ranking.json') if v['kind']=='scalar' and v['batch']==1 and v['converter'] not in ('original','original_inline'))
    save('portable-diagnostic.json',portable)
    labels=['baseline']+[v['name'] for v in selected]
    records=[]
    if resume:
        builds=read('builds-sql.json')
        assert {b['label'] for b in builds}==set(labels+[portable['name']])
        for b in builds:assert sha(BUILD/b['label']/'vec0.dll')==b['binary_sha256']
        candidates=[]
        for p in (RESULTS/'sql.json',RESULTS/'sql.tmp'):
            if p.exists():
                try:candidates.append(json.loads(p.read_text(encoding='utf-8')))
                except json.JSONDecodeError:pass
        records=max(candidates,key=len,default=[])
        print('resuming',len(records),'completed SQL workers',flush=True)
    else:
        builds=[compile_dll('baseline',0)]
        for v in selected:builds.append(compile_dll(v['name'],v['id']))
        builds.append(compile_dll(portable['name'],portable['id']))
        save('builds-sql.json',builds)
        verify_sql(labels+[portable['name']])
    save('sql-complete.json',dict(complete=False,row_cap=CAP))
    save('sql-harness.json',dict(command=sys.argv,run_sha256=sha(__file__),kernels_sha256=sha(HERE/'kernels.py'),resume=resume))
    completed={json.dumps(r['case'],sort_keys=True) for r in records}
    cases=[dict(rows=n,dims=d,metric=m,storage=s,selectivity=100) for n,d,m,s in itertools.product((1000,2000),(384,768,1536),('l2','cosine'),('memory','file'))]
    cases += [dict(rows=2000,dims=768,metric=m,storage='memory',selectivity=sel) for m,sel in [('l1',100),('l2',10),('l2',50),('cosine',10),('cosine',50)]]
    for basecase in cases:
        for repetition in range(3):
            active_labels=labels+([portable['name']] if basecase['dims']==768 else [])
            combos=list(itertools.product(active_labels,('f16','f32')))
            # Rotate by workload and reverse alternating repetitions.
            offset=cases.index(basecase)%len(combos);combos=combos[offset:]+combos[:offset]
            if repetition%2:combos.reverse()
            for variant,dtype in combos:
                case=dict(basecase,variant=variant,dtype=dtype,repetition=repetition)
                if json.dumps(case,sort_keys=True) in completed:continue
                p=subprocess.run([sys.executable,__file__,'worker','--case',json.dumps(case)],capture_output=True,text=True)
                if p.returncode:raise RuntimeError(str(case)+'\n'+p.stdout+p.stderr)
                records.append(json.loads(p.stdout));save('sql.json',records)
        print('SQL complete',basecase,flush=True)
    save('sql-complete.json',dict(complete=True,cases=len(records),queries_per_case=20,row_cap=CAP))


def metadata():
    lib=library();leaves={}
    for leaf,sub in ((0,0),(1,0),(7,0),(0x80000002,0),(0x80000003,0),(0x80000004,0)):
        out=(C.c_int*4)();lib.exp_cpuid(leaf if leaf<2**31 else leaf-2**32,sub,out)
        leaves[f'{leaf:x}:{sub}']=[v&0xffffffff for v in out]
    brand=b''.join(struct.pack('<4I',*leaves[f'{leaf:x}:0']) for leaf in (0x80000002,0x80000003,0x80000004)).decode().strip('\x00 ')
    c=leaves['1:0'][2];d=leaves['1:0'][3];b=leaves['7:0'][1]
    env=vsenv()
    cc=subprocess.run([str(Path(env['VCTOOLSINSTALLDIR'])/'bin/Hostx64/x64/cl.exe')],env=env,capture_output=True,text=True)
    return dict(cpu=brand,cpuid=leaves,flags=dict(sse2=bool(d&(1<<26)),avx=bool(c&(1<<28)),f16c=bool(c&(1<<29)),osxsave=bool(c&(1<<27)),avx2=bool(b&(1<<5)),fma=bool(c&(1<<12)),avx512f=bool(b&(1<<16))),compiler=cc.stderr.splitlines()[0],python=sys.version,numpy=np.__version__,sqlite=sqlite3.sqlite_version,platform=platform.platform(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={str(p.relative_to(ROOT)):sha(p) for p in [*ROOT.glob('sqlite-vec*.c'),ROOT/'sqlite-vec.h.tmpl',HERE/'run.py',HERE/'kernels.py']},sqlite_archive_sha256=sha(BUILD/'sqlite-amalgamation-3450300.zip'),row_cap=CAP,command=sys.argv)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','build','verify','screen','sql','report','all','worker'])
    parser.add_argument('--case')
    parser.add_argument('--resume',action='store_true',help='Resume SQL checkpoints with unchanged native binaries')
    args=parser.parse_args()
    if args.stage=='worker':
        print(json.dumps(sql_worker(json.loads(args.case))));return
    RESULTS.mkdir(parents=True,exist_ok=True)
    if args.stage in ('prepare','all'):prepare()
    if args.stage in ('build','all'):
        if not (BUILD/'source/sqlite-vec.c').exists():prepare()
        save('builds-micro.json',[compile_dll('micro'),compile_dll('micro_novec',novec=True)])
        save('environment.json',metadata())
    if args.stage in ('verify','all'):verify_micro()
    if args.stage in ('screen','all'):screen()
    if args.stage in ('sql','all'):sql_suite(resume=args.resume)
    if args.stage in ('report','all'):
        from report import report
        report()


if __name__=='__main__':main()
