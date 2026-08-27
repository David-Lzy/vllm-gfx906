# v0.28 Qwen 27B TP4 Standard-AWQ Baseline

## Scope

Phase 118 validated the retained gfx906 SplitKV path on all four MI50 GPUs for
Qwen3.6 27B AWQ and Qwen3.8 27B AWQ standard weights. It is a development
compatibility baseline, not a production replacement evaluation.

The fixed contract was TP4, explicit `gfx906_gptq`, FP16 KV cache, 100K model
length, eight sequences, 8,192 batched tokens, no MTP, and the default-off
eight-query-row gfx906 SplitKV selector. Routine validation used text, one/two
256-square images, and JSON constrained output. It deliberately did not run
large-media capacity workloads.

## Result

Both checkpoints became healthy and passed text, image, and JSON 3/3 gates.
The servers finished with no waiting or running requests, and fatal-log scans
were empty.

| Checkpoint | Startup | Fixed-128 C1 | Fixed-128 C8 aggregate | 32K cache-hit fixed-128 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 27B AWQ | 633.464 s | 52.65 tok/s | 216.53 tok/s | 17.21 tok/s |
| Qwen3.8 27B AWQ standard weights | 633.601 s | 52.13 tok/s | 217.75 tok/s | 17.32 tok/s |

For Qwen3.6, the short-decode result is +0.7% C1 and +1.3% C8 against the
retained v0.27 TP4 reference. Historical long-context records used a forced
29-split profile, while this screen intentionally used the current eight-row
selector. Those values are configuration-different evidence, not a strict
regression comparison.

## Follow-up

The standard Qwen3.8 AWQ checkpoint is functional in v0.28 TP4. A distinct
copy-on-write Qwen3.8 checkpoint quantizes only its embedding and output head
to packed INT8. Its initial v0.28 load exposed a narrow model-construction
gap: `Qwen3_5Model` did not pass its compressed-tensors configuration to the
embedding layer, so the expected packed parameter was never created. Phase 119
restored that construction and proved the checkpoint can load, but rejected the
candidate after text, image, and JSON quality all failed. This standard-AWQ
baseline remains valid regardless of that optional checkpoint.

The temporary TP4 server was removed and the Qwen3.5 production Router and
workers were restored and smoke-tested before phase closeout.
