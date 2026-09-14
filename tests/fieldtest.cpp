// Verify the canonical design against an exact reference over adversarial inputs.
#include <stdio.h>
typedef unsigned long long u64; typedef unsigned __int128 u128; typedef unsigned u32;
#define P 0xFFFFFFFF00000001ULL
// exact reference: full 128-bit mod
static u64 ref(u128 x){ return (u64)(x % (u128)P); }
// candidate: canonical-output reduce
static u64 rd(u128 x){
    u64 lo=(u64)x, hi=(u64)(x>>64);
    u32 hh=(u32)(hi>>32), hl=(u32)hi;
    u64 t = lo - (u64)hh;
    if (lo < (u64)hh) t -= 0xFFFFFFFFULL;        // borrowed: drop the extra 2^64 == 2^32-1
    u64 b = ((u64)hl<<32) - (u64)hl;             // hl*(2^32-1)
    u64 r = t + b;
    if (r < t) r += 0xFFFFFFFFULL;               // carried: fold once
    if (r >= P) r -= P;                          // canonical
    return r;
}
int main(){
    u64 bad=0, n=0;
    u64 probes[]={0,1,2,P-1,P,P+1,0xFFFFFFFFULL,0x100000000ULL,0xFFFFFFFFFFFFFFFFULL,
                  0xFFFFFFFF00000000ULL,0x7FFFFFFFFFFFFFFFULL,0x8000000000000000ULL,12345678901234567ULL};
    int m=sizeof(probes)/sizeof(probes[0]);
    for(int i=0;i<m;i++)for(int j=0;j<m;j++){
        u128 prod=(u128)probes[i]*probes[j];
        if(rd(prod)!=ref(prod)){ if(bad<4) printf("MUL MISMATCH %016llx*%016llx\n",probes[i],probes[j]); bad++; } n++;
        u128 sum=(u128)probes[i]+probes[j];
        if(rd(sum)!=ref(sum)){ if(bad<4) printf("ADD MISMATCH %016llx+%016llx\n",probes[i],probes[j]); bad++; } n++;
    }
    // 12-way sums (the int_lin accumulator shape)
    for(int i=0;i<m;i++){ u128 acc=0; for(int k=0;k<12;k++) acc+=(u128)probes[(i+k)%m];
        if(rd(acc)!=ref(acc)){ if(bad<6) printf("SUM12 MISMATCH i=%d\n",i); bad++; } n++; }
    printf("field test: %llu cases, %llu mismatches -> %s\n", n, bad, bad? "FAIL":"PASS");
    return bad?1:0;
}
