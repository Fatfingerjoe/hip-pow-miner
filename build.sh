#!/usr/bin/env bash
# Build qpow-hip for AMD RDNA3 (gfx1100).
#
# Two published builds exist because the ROCm *runtime* version changes the speed:
#   rocm5 runtime -> ~175 MH/s   (distro hipcc, Ubuntu 24.04 `apt install hipcc`)
#   rocm6 runtime -> ~130 MH/s   (AMD amdgpu-install)
# The kernel source is identical. See README.md.
set -euo pipefail

HIPCC="${HIPCC:-/usr/bin/hipcc}"
ARCH="${ARCH:-gfx1100}"
OUT="${OUT:-qpow_hip}"

# -mcode-object-version=4 is REQUIRED when a 6.x hipcc wrapper is linking the 5.x
# runtime (the common case after an amdgpu-install on a box that had distro ROCm).
# Without it the binary compiles but dies at startup with:
#   "create kernel metadata map using COMgr" / "shared object initialization failed"
"$HIPCC" --offload-arch="$ARCH" -O3 -mcode-object-version=4 qpow_hip.cpp easywsclient.cpp -o "$OUT"

echo "built: $OUT"
"./$OUT" version || true
