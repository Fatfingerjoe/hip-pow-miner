// gf_red_s: for x < 2^96 the high-half term and its borrow branch are provably dead.
// Validate congruence mod P, and confirm every non-gf_mul site stays under that bound.
#include <stdio.h>
typedef unsigned long long u64; typedef unsigned __int128 u128; typedef unsigned u32;
#define P 0xFFFFFFFF00000001ULL
static u64 ref(u128 x){ return (u64)(x % (u128)P); }
static u64 rd2(u128 x){                       // current: 2 conditionals
  u64 lo=(u64)x,hi=(u64)(x>>64); u32 hh=(u32)(hi>>32),hl=(u32)hi;
  u64 t=lo-(u64)hh; if(lo<(u64)hh) t-=0xFFFFFFFFULL;
  u64 b=((u64)hl<<32)-(u64)hl; u64 r=t+b; if(r<t) r+=0xFFFFFFFFULL; return r; }
static u64 rds(u128 x){                       // candidate: 1 conditional, needs x < 2^96
  u64 lo=(u64)x; u32 hl=(u32)(u64)(x>>64);
  u64 b=((u64)hl<<32)-(u64)hl; u64 r=lo+b; if(r<lo) r+=0xFFFFFFFFULL; return r; }
int main(){
  u64 pr[]={0,1,2,P-1,P,P+1,0xFFFFFFFFULL,0x100000000ULL,0xFFFFFFFFFFFFFFFFULL,
            0xFFFFFFFF00000000ULL,0x7FFFFFFFFFFFFFFFULL,0x8000000000000000ULL,
            0xFFFFFFFEFFFFFFFFULL,12345678901234567ULL,0xDEADBEEFCAFEBABEULL};
  int m=sizeof(pr)/sizeof(pr[0]); u64 n=0,bad=0,over=0; u128 worst=0;
  // every shape the kernel actually feeds to a reduce OUTSIDE gf_mul
  for(int i=0;i<m;i++){
    for(int j=0;j<m;j++){
      u128 shapes[4]={ (u128)pr[i]+pr[j],                                  // gf_add
                       (u128)pr[i]+pr[j]+pr[(i+1)%m],                      // ext_lin final / int_lin final
                       (u128)pr[i]+pr[j]+pr[(i+2)%m]+pr[(i+3)%m],          // mat4 S
                       (u128)pr[i]+pr[j]+(u128)pr[j]+pr[(i+1)%m] };        // mat4 S+x+2t shape
      for(int c=0;c<4;c++){ u128 x=shapes[c]; n++;
        if(x>worst) worst=x;
        if((u64)(x>>64) >= 0x100000000ULL) over++;      // would break the bound
        if(rds(x)%P != ref(x)) bad++; }
    }
    u128 acc=0; for(int k=0;k<12;k++) acc+=(u128)pr[(i+k)%m];              // int_lin 12-way
    n++; if(acc>worst) worst=acc;
    if((u64)(acc>>64) >= 0x100000000ULL) over++;
    if(rds(acc)%P != ref(acc)) bad++;
  }
  printf("cases=%llu  rds bad=%llu  over-2^96=%llu  worst_hi=%llu (limit 4294967296)\n",
         n, bad, over, (unsigned long long)(worst>>64));
  printf("verdict: %s\n", (bad||over)? "*** UNSAFE ***" : "rds congruent mod P and bound holds everywhere");
  return (bad||over)?1:0;
}
