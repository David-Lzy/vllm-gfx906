# Triton 3.7.1 gfx906 patch series

Upstream Triton revision: `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`
(`v3.7.1`). This series is deliberately maintained outside the vLLM source
tree. Build products, source checkouts, and compiler caches are local-only.

## Apply

```bash
git -C <triton-3.7.1-source> apply \
  <this-repository>/patches/triton-3.7-gfx906/0001-amd-register-vega20-gfx906.patch \
  <this-repository>/patches/triton-3.7-gfx906/0002-amd-enable-vega20-dpp-broadcast.patch
```

## Patch intent

`0001` restores the small Vega20/gfx906 support surface present in the
validated Triton 3.6 gfx906 fork:

- classify `gfx906` as its own `VEGA20` ISA family;
- retain a 64-lane wavefront;
- keep direct-to-LDS loads limited to 32-bit accesses;
- expose only the existing conservative VDot/CDNA fallback path.

It does not pretend that Vega20 supports CDNA3/4 instructions, tensor-memory
operations, async-copy specializations, FP8 instructions, or multi-CTA
launches. Each further compiler error must become a separate patch with its
own numerical evidence.

`0002` retains the gfx9 DPP broadcast reduction path already used by the
validated Triton 3.6 fork. Without it, the newer feature helper directs Vega20
through an RDNA-only `permlanex16` intrinsic that gfx906 cannot select.

## Build and package gate

Use the Phase 19 scripts to produce a relocatable wheel before putting this
compiler underneath vLLM. The source checkout and all build artifacts remain
outside the repository.

```bash
TRITON_SOURCE_DIR=<patched-triton-3.7.1-source> \
TRITON_ARTIFACT_ROOT=<external-artifact-root> \
tools/build-gfx906-phase19-triton37-wheel.sh

TRITON_WHEEL_DIR=<printed-wheel-dir> \
TRITON_ARTIFACT_ROOT=<external-artifact-root> \
TRITON_GPU_INDEX=2 \
tools/run-gfx906-phase19-triton37-wheel-smoke.sh
```

The second command installs the wheel into a fresh environment and runs the
gfx906 elementwise, reduction, and FP16 matmul smoke suite. Only a successful
packaged-wheel smoke may be used as the Triton input for the v0.27 image build.
