// Derive M_E columns for lanes 3 and 7, and verify M_E*v = 35*(e3-e7).
#include <stdio.h>
typedef long long i64;
static void mat4(i64*x){ i64 s=x[0]+x[1]+x[2]+x[3],o[4];
  for(int i=0;i<4;i++){ o[i]=s+x[i]+2*x[(i+1)&3]; } for(int i=0;i<4;i++)x[i]=o[i]; }
static void ext(i64*s){ for(int c=0;c<12;c+=4) mat4(s+c);
  i64 sum[4]; for(int i=0;i<4;i++) sum[i]=s[i]+s[4+i]+s[8+i];
  for(int i=0;i<12;i++) s[i]=s[i]+sum[i&3]; }
int main(){
  for(int lane=3; lane<=7; lane+=4){
    i64 e[12]={0}; e[lane]=1; ext(e);
    printf("M_E column for lane %d: {",lane);
    for(int i=0;i<12;i++) printf("%lld%s",e[i], i<11?",":"");
    printf("}\n");
  }
  // confirm the direction: v=(a,-a,0), a=(17,-11,3,-4) over the 8 absorbed felts
  i64 a[4]={17,-11,3,-4};
  i64 v[12]={a[0],a[1],a[2],a[3], -a[0],-a[1],-a[2],-a[3], 0,0,0,0};
  ext(v);
  printf("M_E * v = {");
  for(int i=0;i<12;i++) printf("%lld%s",v[i], i<11?",":"");
  printf("}   (want 35 at lane3, -35 at lane7, 0 elsewhere)\n");
  return 0;
}
