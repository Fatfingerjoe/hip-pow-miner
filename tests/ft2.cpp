// Validate candidate reduces against exact %P. Output need NOT be canonical, but must be
// congruent mod P and < 2^64 (so downstream u128 accumulation stays exact).
#include <stdio.h>
typedef unsigned long long u64; typedef unsigned __int128 u128; typedef unsigned u32;
#define P 0xFFFFFFFF00000001ULL
static u64 ref(u128 x){ return (u64)(x % (u128)P); }

// current: 3 conditionals, canonical output
static u64 rd3(u128 x){
  u64 lo=(u64)x,hi=(u64)(x>>64); u32 hh=(u32)(hi>>32),hl=(u32)hi;
  u64 t=lo-(u64)hh; if(lo<(u64)hh) t-=0xFFFFFFFFULL;
  u64 b=((u64)hl<<32)-(u64)hl;
  u64 r=t+b; if(r<t) r+=0xFFFFFFFFULL; if(r>=P) r-=P; return r;
}
// candidate: drop the canonicalising subtract (2 conditionals, non-canonical output)
static u64 rd2(u128 x){
  u64 lo=(u64)x,hi=(u64)(x>>64); u32 hh=(u32)(hi>>32),hl=(u32)hi;
  u64 t=lo-(u64)hh; if(lo<(u64)hh) t-=0xFFFFFFFFULL;
  u64 b=((u64)hl<<32)-(u64)hl;
  u64 r=t+b; if(r<t) r+=0xFFFFFFFFULL; return r;
}
int main(){
  u64 pr[]={0,1,2,P-1,P,P+1,0xFFFFFFFFULL,0x100000000ULL,0xFFFFFFFFFFFFFFFFULL,
            0xFFFFFFFF00000000ULL,0x7FFFFFFFFFFFFFFFULL,0x8000000000000000ULL,
            0xFFFFFFFEFFFFFFFFULL,12345678901234567ULL,0xDEADBEEFCAFEBABEULL};
  int m=sizeof(pr)/sizeof(pr[0]); u64 n=0,b3=0,b2=0,nc=0;
  for(int i=0;i<m;i++)for(int j=0;j<m;j++){
    u128 cases[3]={(u128)pr[i]*pr[j], (u128)pr[i]+pr[j], ((u128)pr[i]<<64)|pr[j]};
    for(int c=0;c<3;c++){ u128 x=cases[c]; n++;
      u64 e=ref(x);
      if(rd3(x)!=e) b3++;
      u64 v=rd2(x); if(v%P!=e) b2++;          // congruent mod P is the requirement
      if(v>=P) nc++;                           // how often output is non-canonical
    }
  }
  // 12-way accumulator shape
  for(int i=0;i<m;i++){ u128 a=0; for(int k=0;k<12;k++) a+=(u128)pr[(i+k)%m];
    u64 e=ref(a); n++; if(rd3(a)!=e) b3++; if(rd2(a)%P!=e) b2++; }
  printf("cases=%llu  rd3(3-cond) bad=%llu   rd2(2-cond) bad=%llu   rd2 non-canonical outputs=%llu\n",n,b3,b2,nc);
  printf("verdict: rd2 %s\n", b2? "*** INCORRECT ***":"congruent mod P on every case");
  return b2?1:0;
}
