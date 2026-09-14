# hip-pow-miner
HIP/ROCm Quantus QPoW miner for AMD RDNA3 GPUs (RX 7900 XTX / gfx1100).

## Build
```bash
mkdir build && cd build
hipcc -O3 --offload-arch=gfx1100 ../src/bench.cpp -o hip_miner
```

## Benchmark
```bash
./hip_miner bench <per_thread> <tpb> <blocks> <seconds>
./hip_miner bench 16 256 384 10
```
