"""Generate isolated C kernels. Production sources are never edited.

The bit-bias/subtraction identities are described by Fabian Giesen (2012),
https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/ and
Marat Dukhan's FP16 library, https://github.com/Maratyszcza/FP16.
Implementations below are written for this experiment; no dependency or LUT.
"""
CONVERTERS = ['original', 'int_branch', 'int_select', 'magic_branch', 'magic_select']
VARIANTS = [dict(name='baseline', kind='baseline', batch=1, converter='original')]
for conv in CONVERTERS:
    for batch in (1, 2, 4, 8):
        VARIANTS.append(dict(name=f'{conv}_b{batch}', kind='scalar', batch=batch, converter=conv))
for width in (4, 8):
    VARIANTS.append(dict(name=f'block{width}', kind='block', batch=1, converter='magic_branch', width=width))
for batch in (1, 2, 4, 8):
    VARIANTS.append(dict(name=f'sse2_b{batch}', kind='sse', batch=batch, converter='magic_select'))
VARIANTS.append(dict(name='sse2_candidates4', kind='across', batch=4, converter='magic_select'))
for batch in (1,2,4,8):
    VARIANTS.append(dict(name=f'original_inline_b{batch}',kind='scalar',batch=batch,converter='original_inline'))
for i, v in enumerate(VARIANTS):
    v['id'] = i

PREAMBLE = r'''
/* Generated benchmark code. SSE2 is conditional; portable C needs no SIMD. */
#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#define EXP_SSE2 1
#endif
#ifdef _MSC_VER
#define EXP_INLINE static __forceinline
#else
#define EXP_INLINE static inline
#endif
EXP_INLINE float exp_from_bits(uint32_t b) {float x; memcpy(&x,&b,4);return x;}
EXP_INLINE uint32_t exp_bits(float x) {uint32_t b;memcpy(&b,&x,4);return b;}
EXP_INLINE uint16_t exp_loadh(const void *p,size_t i) {uint16_t h;memcpy(&h,(const unsigned char*)p+2*i,2);return h;}
EXP_INLINE float exp_loadf(const void *p,size_t i) {float x;memcpy(&x,(const unsigned char*)p+4*i,4);return x;}
EXP_INLINE float exp_original(uint16_t h) {return vec_half_to_float(h);}
/* Inlining-only ablation: the original converter's arithmetic, unchanged. */
EXP_INLINE float exp_original_inline(uint16_t h) {
 uint32_t sign=((uint32_t)h&0x8000)<<16;
 uint32_t exponent=(h>>10)&31,mantissa=h&1023,bits;
 if(exponent==0){if(!mantissa)bits=sign;else{int e=-14;
 while(!(mantissa&1024)){mantissa<<=1;e--;}
 bits=sign|((uint32_t)(e+127)<<23)|((mantissa&1023)<<13);}}
 else if(exponent==31)bits=sign|0x7f800000|(mantissa<<13);
 else bits=sign|((exponent+112)<<23)|(mantissa<<13);
 return exp_from_bits(bits);
}
EXP_INLINE uint32_t exp_normal_bits(uint16_t h) {return (((uint32_t)h&32767)<<13)+0x38000000u;}
EXP_INLINE float exp_normal(uint16_t h) {return exp_from_bits(exp_normal_bits(h)|(((uint32_t)h&32768)<<16));}
EXP_INLINE float exp_int_branch(uint16_t h) {
 uint32_t b;
 if(h&0x7c00) b=exp_normal_bits(h);
 else b=exp_bits((float)(h&1023)*0x1p-24f);
 return exp_from_bits(b|(((uint32_t)h&32768)<<16));
}
EXP_INLINE float exp_magic_branch(uint16_t h) {
 uint32_t b;
 if(h&0x7c00) b=exp_normal_bits(h);
 else b=exp_bits(exp_from_bits(0x3f000000u|(h&1023))-0.5f);
 return exp_from_bits(b|(((uint32_t)h&32768)<<16));
}
EXP_INLINE float exp_int_select(uint16_t h) {
 uint32_t mask=0u-(uint32_t)((h&0x7c00)!=0);
 uint32_t small=exp_bits((float)(h&1023)*0x1p-24f);
 return exp_from_bits(((exp_normal_bits(h)&mask)|(small&~mask))|(((uint32_t)h&32768)<<16));
}
EXP_INLINE float exp_magic_select(uint16_t h) {
 uint32_t mask=0u-(uint32_t)((h&0x7c00)!=0);
 uint32_t small=exp_bits(exp_from_bits(0x3f000000u|(h&1023))-0.5f);
 return exp_from_bits(((exp_normal_bits(h)&mask)|(small&~mask))|(((uint32_t)h&32768)<<16));
}
#ifdef EXP_SSE2
/* Input: four zero-extended binary16 encodings in 32-bit integer lanes. */
EXP_INLINE __m128 exp_sse_decode(__m128i h) {
 __m128i sign=_mm_slli_epi32(_mm_and_si128(h,_mm_set1_epi32(32768)),16);
 __m128i mag=_mm_and_si128(h,_mm_set1_epi32(32767));
 __m128i normal=_mm_add_epi32(_mm_slli_epi32(mag,13),_mm_set1_epi32(0x38000000));
 __m128i zeroexp=_mm_cmpeq_epi32(_mm_and_si128(h,_mm_set1_epi32(0x7c00)),_mm_setzero_si128());
 __m128 small=_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_and_si128(h,_mm_set1_epi32(1023)),_mm_set1_epi32(0x3f000000))),_mm_set1_ps(0.5f));
 return _mm_castsi128_ps(_mm_or_si128(sign,_mm_or_si128(_mm_andnot_si128(zeroexp,normal),_mm_and_si128(zeroexp,_mm_castps_si128(small)))));
}
EXP_INLINE __m128 exp_sse_load(const void *p,size_t i) {
 __m128i h=_mm_loadl_epi64((const __m128i*)((const unsigned char*)p+2*i));
 return exp_sse_decode(_mm_unpacklo_epi16(h,_mm_setzero_si128()));
}
EXP_INLINE float exp_reduce(__m128 v) {float a[4];_mm_storeu_ps(a,v);float s=0;for(int j=0;j<4;j++)s+=a[j];return s;}
#endif
'''


def update(j, x, q, metric):
    if metric == 1:
        return f'float diff{j}=({x})-({q}); s{j}+=diff{j}*diff{j};'
    if metric == 3:
        return f's{j}+=fabsf(({x})-({q}));'
    return f's{j}+=({x})*({q}); t{j}+=({x})*({x});'


def batch_function(v, half, metric, batch):
    name = f'exp_batch_{v["id"]}_{half}_{metric}_{batch}'
    sig = f'EXP_INLINE void {name}(const void **a,const float*q,size_t n,float qn,float*out) {{'
    if v['kind'] == 'baseline':
        if half:
            call = {1:'l2_sqr_float16(a[0],q,n)', 2:'cosine_float16(a[0],q,n,qn)', 3:'l1_float16(a[0],q,n)'}[metric]
        else:
            call = {1:'distance_l2_sqr_float(a[0],q,&n)', 2:'distance_cosine_float(a[0],q,&n)', 3:'distance_l1_f32(a[0],q,&n)'}[metric]
        return name, sig + f'out[0]={call};}}\n'
    s = [sig, '(void)qn;']
    for j in range(batch):
        s.append(f'float s{j}=0,t{j}=0;')
    load = lambda j, idx='i': f'exp_{v["converter"]}(exp_loadh(a[{j}],{idx}))' if half else f'exp_loadf(a[{j}],{idx})'
    if v['kind'] == 'sse':
        s.append('#ifdef EXP_SSE2')
        for j in range(batch):
            s.append(f'__m128 vs{j}=_mm_setzero_ps(), vt{j}=_mm_setzero_ps();')
        s.append('size_t i=0;for(;i+4<=n;i+=4){__m128 y=_mm_loadu_ps(q+i);')
        for j in range(batch):
            x = f'exp_sse_load(a[{j}],i)' if half else f'_mm_loadu_ps((const float*)a[{j}]+i)'
            s.append(f'__m128 x{j}={x};')
            if metric == 2:
                s.append(f'vs{j}=_mm_add_ps(vs{j},_mm_mul_ps(x{j},y));vt{j}=_mm_add_ps(vt{j},_mm_mul_ps(x{j},x{j}));')
            else:
                s.append(f'__m128 diff{j}=_mm_sub_ps(x{j},y);')
                val = f'_mm_mul_ps(diff{j},diff{j})' if metric == 1 else f'_mm_andnot_ps(_mm_set1_ps(-0.0f),diff{j})'
                s.append(f'vs{j}=_mm_add_ps(vs{j},{val});')
        s.append('}')
        for j in range(batch):
            s.append(f's{j}=exp_reduce(vs{j});t{j}=exp_reduce(vt{j});')
        s.append('for(;i<n;i++){float y=q[i];')
        for j in range(batch):
            s.append(f'float x{j}={load(j)};' + update(j,f'x{j}','y',metric))
        s.append('}\n#else\n#error SSE2 variant requires SSE2 target\n#endif')
    elif v['kind'] == 'across' and batch == 4:
        s.append('#ifdef EXP_SSE2\n__m128 vs=_mm_setzero_ps(),vt=_mm_setzero_ps();for(size_t i=0;i<n;i++){')
        if half:
            s.append('__m128 x=exp_sse_decode(_mm_setr_epi32(exp_loadh(a[0],i),exp_loadh(a[1],i),exp_loadh(a[2],i),exp_loadh(a[3],i)));')
        else:
            s.append('__m128 x=_mm_setr_ps(exp_loadf(a[0],i),exp_loadf(a[1],i),exp_loadf(a[2],i),exp_loadf(a[3],i));')
        s.append('__m128 y=_mm_set1_ps(q[i]);')
        if metric == 2:
            s.append('vs=_mm_add_ps(vs,_mm_mul_ps(x,y));vt=_mm_add_ps(vt,_mm_mul_ps(x,x));')
        else:
            val = '_mm_mul_ps(diff,diff)' if metric == 1 else '_mm_andnot_ps(_mm_set1_ps(-0.0f),diff)'
            s.append(f'__m128 diff=_mm_sub_ps(x,y);vs=_mm_add_ps(vs,{val});')
        s.append('}float ss[4],tt[4];_mm_storeu_ps(ss,vs);_mm_storeu_ps(tt,vt);')
        for j in range(4):
            s.append(f's{j}=ss[{j}];t{j}=tt[{j}];')
        s.append('#else\n#error SSE2 variant requires SSE2 target\n#endif')
    elif v['kind'] == 'block' and half:
        w = v['width']
        s.append(f'size_t i=0;for(;i+{w}<=n;i+={w}){{uint16_t h[{w}];int normal=1;')
        for z in range(w):
            s.append(f'h[{z}]=exp_loadh(a[0],i+{z});normal &= ((h[{z}]&0x7c00)!=0);')
        s.append('if(normal){')
        for z in range(w):
            s.append('{float x=exp_normal(h['+str(z)+']);'+update(0,'x',f'q[i+{z}]',metric)+'}')
        s.append('}else{')
        for z in range(w):
            s.append('{float x=exp_magic_branch(h['+str(z)+']);'+update(0,'x',f'q[i+{z}]',metric)+'}')
        s.append('}}for(;i<n;i++){float x=exp_magic_branch(exp_loadh(a[0],i));'+update(0,'x','q[i]',metric)+'}')
    else:
        s.append('for(size_t i=0;i<n;i++){float y=q[i];')
        for j in range(batch):
            s.append(f'float x{j}={load(j)};'+update(j,f'x{j}','y',metric))
        s.append('}')
    for j in range(batch):
        result = f'sqrtf(s{j})' if metric == 1 else (f'1-s{j}/(sqrtf(t{j})*qn)' if metric == 2 else f's{j}')
        s.append(f'out[{j}]={result};')
    s.append('}')
    return name, '\n'.join(s)+'\n'


def generate(selected=None):
    variants = VARIANTS if selected is None else [VARIANTS[selected]]
    result = [PREAMBLE]
    for v in variants:
        vid, b = v['id'], v['batch']
        for half in (0,1):
            for metric in (1,2,3):
                name, source = batch_function(v,half,metric,b)
                result.append(source)
                tail = name
                if b != 1:
                    tail, source = batch_function(v,half,metric,1)
                    result.append(source)
                result.append(f'''static void exp_scan_{vid}_{half}_{metric}(const void*base,const float*q,size_t n,int rows,const unsigned char*mask,float qn,float*out) {{
 const void*a[{b}];int ids[{b}],used=0;float d[{b}];
 for(int r=0;r<rows;r++){{
  if(mask && !(mask[r>>3]&(1u<<(r&7))))continue;
  a[used]=(const unsigned char*)base+(size_t)r*n*{2 if half else 4};ids[used++]=r;
  if(used=={b}){{{name}(a,q,n,qn,d);for(int j=0;j<{b};j++)out[ids[j]]=d[j];used=0;}}
 }}
 for(int j=0;j<used;j++){{{tail}(a+j,q,n,qn,d);out[ids[j]]=d[0];}}
}}
''')
        result.append(f'EXP_INLINE void exp_run_{vid}(int half,int metric,const void*base,const float*q,size_t n,int rows,const unsigned char*mask,float qn,float*out){{\n')
        for half in (0,1):
            result.append(('if(half){' if half else 'if(!half){')+'switch(metric){')
            for metric in (1,2,3):
                result.append(f'case {metric}:exp_scan_{vid}_{half}_{metric}(base,q,n,rows,mask,qn,out);return;')
            result.append('}}')
        result.append('}\n')
    return '\n'.join(result)


def exports():
    s = [r'''
#include <intrin.h>
#define EXPORTED __declspec(dllexport)
EXPORTED unsigned exp_mxcsr(unsigned value,int set){unsigned old=_mm_getcsr();if(set)_mm_setcsr(value);return old;}
EXPORTED void exp_cpuid(int leaf,int subleaf,int*out){__cpuidex(out,leaf,subleaf);}
EXPORTED void exp_distance(int variant,int half,int metric,const void*base,const float*q,size_t n,int rows,const unsigned char*mask,float*out,int repeat){
 if(rows<0||rows>2000)return;
 float qn=vec_query_norm(q,n);
 for(int rep=0;rep<repeat;rep++){
 switch(variant){
''']
    for v in VARIANTS:
        s.append(f'case {v["id"]}:exp_run_{v["id"]}(half,metric,base,q,n,rows,mask,qn,out);break;')
    s.append('}}}\nEXPORTED void exp_convert(int variant,const void*data,float*out,size_t n,int repeat){for(int r=0;r<repeat;r++){switch(variant){')
    for i,c in enumerate(CONVERTERS):
        s.append(f'case {i}:for(size_t i=0;i<n;i++)out[i]=exp_{c}(exp_loadh(data,i));break;')
    s.append('case 5:{size_t i=0;for(;i+4<=n;i+=4)_mm_storeu_ps(out+i,exp_sse_load(data,i));for(;i<n;i++)out[i]=exp_magic_select(exp_loadh(data,i));break;}')
    for cid,w in ((6,4),(7,8)):
        s.append(f'case {cid}:{{size_t i=0;for(;i+{w}<=n;i+={w}){{int normal=1;uint16_t h[{w}];for(int j=0;j<{w};j++){{h[j]=exp_loadh(data,i+j);normal&=((h[j]&0x7c00)!=0);}}if(normal){{for(int j=0;j<{w};j++)out[i+j]=exp_normal(h[j]);}}else{{for(int j=0;j<{w};j++)out[i+j]=exp_magic_branch(h[j]);}}}}for(;i<n;i++)out[i]=exp_magic_branch(exp_loadh(data,i));break;}}')
    s.append('case 8:for(size_t i=0;i<n;i++)out[i]=exp_original_inline(exp_loadh(data,i));break;')
    s.append('}}}\n')
    return '\n'.join(s)
