// Minimal pool client for the HIP miner. Protocol mirrors cuda-miner/pool_client.inc.
#include "easywsclient.hpp"
#include <string>
#include <vector>
#include <chrono>
#include <random>
using easywsclient::WebSocket;

static std::string jstr(const std::string& s, const char* key){
    std::string k=std::string("\"")+key+"\""; size_t p=s.find(k); if(p==std::string::npos) return "";
    p=s.find(':',p+k.size()); if(p==std::string::npos) return "";
    size_t a=s.find('"',p); if(a==std::string::npos) return ""; size_t b=s.find('"',a+1);
    if(b==std::string::npos) return ""; return s.substr(a+1,b-a-1);
}
static void hexbytes(const std::string& h, unsigned char* o, int n){
    for(int i=0;i<n;i++){ auto hv=[&](char c){return c>='0'&&c<='9'?c-'0':(c>='a'&&c<='f'?c-'a'+10:(c>='A'&&c<='F'?c-'A'+10:0));};
        o[i]=(unsigned char)(hv(h[2*i])*16+hv(h[2*i+1])); } }
static std::string tohex(const unsigned char* b, int n){
    static const char* d="0123456789abcdef"; std::string s; s.reserve(2*n);
    for(int i=0;i<n;i++){ s+=d[b[i]>>4]; s+=d[b[i]&15]; } return s; }
// 64-byte BE target = (2^512-1)/difficulty
static bool build_target(const std::string& dd, unsigned char* t){
    unsigned long long diff=strtoull(dd.c_str(),nullptr,10); if(!diff) return false;
    unsigned long long rem=0;
    for(int i=0;i<64;i++){ unsigned long long cur=(rem<<8)|0xFFULL; t[i]=(unsigned char)(cur/diff); rem=cur%diff; }
    return true;
}
static int run_pool(const char* url, const char* payout, const char* worker, u32 per_thread, int tpb){
    setvbuf(stdout,NULL,_IONBF,0);
    std::mt19937_64 rng(std::random_device{}());
    unsigned char header[32], target[64], high[32];
    unsigned char *dh,*dnh,*dtg; u64* dms; FoundR* dres;
    HC(hipMalloc(&dh,32)); HC(hipMalloc(&dnh,32)); HC(hipMalloc(&dtg,64)); HC(hipMalloc(&dms,12*sizeof(u64)));
    HC(hipMalloc(&dres,sizeof(FoundR)));
    hipDeviceProp_t pr; HC(hipGetDeviceProperties(&pr,0));
    int blocks = pr.multiProcessorCount*8;
    printf("pool miner: %s  url=%s payout=%s worker=%s  blocks=%d tpb=%d per_thread=%u\n",
           pr.name, url, payout, worker, blocks, tpb, per_thread);
    long long submitted=0, accepted=0, rejected=0, throttled=0; unsigned long long hashes=0;
    const double MIN_SUBMIT_GAP=0.20;   // seconds, only enforced while difficulty < 1e8
    auto last_submit=std::chrono::steady_clock::now();
    auto t_start=std::chrono::steady_clock::now(), t_log=t_start;
    while(true){
        WebSocket::pointer ws = WebSocket::from_url(url);
        if(!ws){ printf("connect failed, retry in 5s\n"); sleep_ms(5000); continue; }
        { char hello[512]; snprintf(hello,sizeof hello,
            "{\"Hello\":{\"payout_address\":\"%s\",\"worker_name\":\"%s\",\"client\":\"qpow-hip/0.1\"}}",payout,worker);
          ws->send(hello); ws->poll(); ws->dispatch([](const std::string&){}); }
        bool have=false; std::string job, diff; u64 group=0;
        while(ws->getReadyState()!=WebSocket::CLOSED){
            ws->poll(0);
            ws->dispatch([&](const std::string& msg){
                if(msg.find("mining_hash")!=std::string::npos){
                    std::string mh=jstr(msg,"mining_hash"), df=jstr(msg,"difficulty"), jid=jstr(msg,"job_id");
                    if(mh.size()<64||df.empty()) return;
                    hexbytes(mh,header,32);
                    if(!build_target(df,target)) return;
                    for(int i=0;i<32;i++) high[i]=(unsigned char)(rng()&0xFF);
                    HC(hipMemcpy(dh,header,32,hipMemcpyHostToDevice));
                    HC(hipMemcpy(dnh,high,32,hipMemcpyHostToDevice));
                    HC(hipMemcpy(dtg,target,64,hipMemcpyHostToDevice));
                    hipLaunchKernelGGL(k_midstate,dim3(1),dim3(1),0,0,dh,dnh,dms);
                    HC(hipDeviceSynchronize());
                    job=jid; diff=df; group=0; have=true;
                    printf("[job %s] diff=%s target=%02x%02x%02x%02x...\n",jid.c_str(),df.c_str(),target[0],target[1],target[2],target[3]);
                } else if(msg.find("accepted")!=std::string::npos||msg.find("Accepted")!=std::string::npos) accepted++;
                else if(msg.find("rejected")!=std::string::npos||msg.find("Rejected")!=std::string::npos){
                    rejected++; printf("  REJECTED: %s\n", msg.substr(0,160).c_str()); }
            });
            if(!have){ sleep_ms(50); continue; }
            HC(hipMemset(dres,0,sizeof(FoundR)));
            hipLaunchKernelGGL(k_mine_sparse,dim3(blocks),dim3(tpb),0,0,dms,group,per_thread,dtg,dres);
            HC(hipDeviceSynchronize());
            hashes += (unsigned long long)blocks*tpb*per_thread;
            FoundR r; HC(hipMemcpy(&r,dres,sizeof(FoundR),hipMemcpyDeviceToHost));
            unsigned nh = r.n<8?r.n:8;
            for(unsigned k=0;k<nh;k++){
                auto nsub=std::chrono::steady_clock::now();
                if(atof(diff.c_str())<1e8 &&
                   std::chrono::duration<double>(nsub-last_submit).count()<MIN_SUBMIT_GAP){ throttled++; continue; }
                unsigned char nonce[64]; memcpy(nonce,high,32);
                nd_low_bytes(r.group[k], r.local[k], nonce+32);
                std::string nx=tohex(nonce,64);
                char res[1024]; snprintf(res,sizeof res,
                  "{\"JobResult\":{\"status\":\"completed\",\"job_id\":\"%s\",\"nonce\":\"%s\",\"work\":\"%s\",\"hash_count\":%llu,\"elapsed_time\":0.0,\"miner_id\":null}}",
                  job.c_str(), nx.c_str(), nx.c_str(), (unsigned long long)hashes);
                ws->send(res); ws->poll(0); submitted++; last_submit=nsub;
            }
            group += blocks;
            auto now=std::chrono::steady_clock::now();
            double since=std::chrono::duration<double>(now-t_log).count();
            if(since>=10.0){
                double el=std::chrono::duration<double>(now-t_start).count();
                printf("[%.0fs] %.1f MH/s | diff=%s submitted=%lld accepted=%lld rejected=%lld\n",
                       el, hashes/el/1e6, diff.c_str(), submitted, accepted, rejected);
                t_log=now;
            }
        }
        printf("disconnected — reconnecting\n"); delete ws; sleep_ms(1000);
    }
    return 0;
}
