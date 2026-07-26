# JoyAI-Echo GGUF nodes — multishot fixes + automation patch


> **Before you dive in - a word on expectations.** This is a community-built,
> bleeding-edge pipeline: a 22B audio+video model with cross-shot memory,
> running on consumer hardware. It is powerful, but it is not one-click - your
> first clean render will likely take some tuning to YOUR machine (VRAM, system
> RAM and pagefile, and which model build: bf16 / fp8 / GGUF / INT8). No two
> rigs behave identically. `INSTRUCTIONS.md` covers install, per-VRAM settings,
> and every failure mode reported so far. **If you get stuck, [open an issue](../../issues)
> - I answer, I troubleshoot, and most of the fixes in this pack exist because
> someone reported something.** You will not be left hanging.


## Quick fixes — read this first

Nearly every problem reported with this pack is one of these.

| symptom | what's actually wrong | fix |
|---|---|---|
| **Renders crawl; VRAM pinned at 100%** | Without `fp8_scaled_mm` the DiT runs in bf16 — ~40 GB staged. On a 32 GB card that streams over PCIe every step. | Turn **`fp8_scaled_mm` ON** (with a **bf16 or INT8** checkpoint) and **`sequential_offload` OFF** — ~22 GB resident, native fp8 matmul on RTX 40/50-series. It is **off by default** and is the single biggest speed setting in this pack. |
| **`fp8_scaled_mm` errors — "needs the bf16 checkpoint as its source"** | The toggle downcasts the attention/FF linears *itself*, so it must start from bf16. An fp8 FILE would load every tensor as fp8 — norms, tables and adalns included — and crash the denoise pipeline. | Point `model_file` at the **bf16** (or **INT8**) build. Do not pair the fp8 file with this toggle. |
| **The fp8 file didn't speed anything up** | With the toggles off, an fp8 checkpoint **upcasts to bf16 at load**. It saves download size, not memory or time. | Use **bf16 or INT8 + `fp8_scaled_mm`** for the actual win. |
| **The INT8 file is slow** | You ran it without `fp8_scaled_mm`. INT8 reconstructs to bf16 at load, so on its own it is the bf16 path plus ~40 s of reconstruction. | Turn **`fp8_scaled_mm` ON**. INT8 then behaves exactly like bf16 as a source — same ~22 GB resident — at roughly 60 % of the download. |
| **Lip sync drifts apart ~10 s into every shot** | The wrapper's video RoPE clock was hardcoded to 24 fps while audio RoPE runs in true seconds — a 25 fps render drifts ~4 %/s. Looks like a model limit; it is not. | Apply this patch (**Bug fix #0**). No checkpoint can fix it. With the patch, 60–105 s multishot masters hold sync. |
| **`memory_size=0` every shot; a new face each shot** | With `enable_audio_memory` off, the video memory-bank save was gated on the audio latent, so the bank never filled. | Fixed in this patch (**Bug fix #1**). Console `memory_size=` should climb 0,1,2,… up to your cap. |
| **Quality degrades over a long run** — waxy skin, smearing by the late shots | The memory-bank trim was a no-op when `memory_max_size <= num_fix_frames`, so the bank grew unbounded. | Fixed in this patch (**Bug fix #1b**). `memory_size=` now freezes at your cap. |
| **A GGUF errors about VAEs** | A GGUF is **DiT-only**. | Keep a full bf16 checkpoint in `checkpoint_path` — it supplies the VAEs, vocoder and connectors. |
| **Burned-in subtitles or captions** | The DMD pipeline has no CFG, so a plain negative does nothing. | Use `negative_prompt_video` / `negative_scale_video` (~0.5). Above ~0.8 it locks every shot to shot 1's composition. |
| **The voice comes out British** | LTX drifts British unless told otherwise. | Name it in the *positive*: "in a casual American accent". |
| **Occasional robotic voice on long runs** | JoyAI-Echo's finetune is what suppresses it; the e50 merge keeps half of it. | For long multishot runs prefer the full-Echo surgical merge; use e50 for talking heads. |
| **Second shot won't lip-sync in a hand-built chain** | Guiding shot 2 with shot 1's decoded last frame is broken on *any* checkpoint — the guide is pixel-continuable, so the sampler reproduces it. | Extend with real audio+video latent context. See [Multishot Lite v2](https://huggingface.co/joeygambino/ltx23-multishot-lite). |

## Also here: Multishot Lite (core ComfyUI + one KJNodes node)

A second, much simpler workflow for chained talking shots, with **no node pack
to install**.

**Its own repo:** https://github.com/jlucasmcrell/ltx23-multishot-lite
(standalone docs + issues) · HF: https://huggingface.co/joeygambino/ltx23-multishot-lite

Mirrored here: [MULTISHOT_LITE.md](MULTISHOT_LITE.md) · [Releases](../../releases/latest).

* **Core ComfyUI nodes plus one from ComfyUI-KJNodes** (`LTXVAudioVideoMask`,
  which powers the extension). Nothing else to install.
* **Shot 2 is a true audio+video extension of shot 1**, not a cut: the last ~3 s
  of shot 1's video *and audio* become latent context, so the model generates
  forward from an ongoing utterance and the voice carries over by construction.
* **Mode 1 (default):** character speaks in a **reference voice you supply**.
  **Mode 2:** bypass one node per shot; the model invents a voice.
* Both shots joined and refined into one `FINAL` file.

> **v1.x is superseded.** It chained shots on shot 1's decoded last frame. That
> guide is pixel-continuable, so the sampler reproduced it instead of
> lip-syncing — chained shots came out as voiceover over a barely-moving face,
> and it got *worse* on more strongly distilled checkpoints. No setting fixed
> it. If you are on v1.x, update.

**Not a replacement for the patch below.** Lite has *no memory bank* — identity
continuity comes only from the extension context, so it drifts over many shots.
That is exactly why the node pack exists.

---

## Start here - which download do I need?

**Download the zip from [Releases](../../releases/latest).** That is the whole
thing: every patch file, the workflow, an example prompt file, and step-by-step
instructions. You do not need to clone this repo.

The files in this repo are the same files, unpacked, for browsing and diffing.

**This is a patch, not a standalone node pack.** Install RealRebelAI's
`ComfyUI_JoyAI_Echo_GGUF_Nodes` first, then MERGE these files over it (replace
when prompted) - do not delete or replace the whole folder. The pack prints a
clear startup error if it detects a replace-instead-of-merge install.

Models are not here; they are on Hugging Face: https://huggingface.co/joeygambino

---


> **This is a patch, not a one-click ComfyUI Manager install.**
> It layers on top of an existing `ComfyUI_JoyAI_Echo_GGUF_Nodes` install
> (RealRebelAI's Rebels GGUF loader stack). Install that first, then **merge**
> these files over it - do not replace the folder. The pack prints a clear
> startup error if it detects a replace-instead-of-merge install.
>
> Models are **not** in this repo. They live on Hugging Face:
> https://huggingface.co/joeygambino


A set of bug fixes and features layered on top of the community
`ComfyUI_JoyAI_Echo_GGUF_Nodes` pack (the Rebels GGUF loader stack around
JoyAI-Echo). Everything here targets the **multi-shot** path (`JoyEcho_Generate`
+ the discrete Rebels loaders / `JoyEcho_ModelLoader`).

This is a **patch drop**, not a standalone pack: copy these files over a working
install of the same pack (back up first). The files are interdependent — in
particular `nodes.py` calls new signatures added to the two `libs/` files, so
apply them together.

Tested on an RTX 5090 (32 GB) and a 3090 (24 GB), ComfyUI 0.26–0.27,
torch 2.8–2.11, with the JoyAI-Echo bf16 release and self-built Q8 GGUFs.

---

## Files in this package

```
nodes.py                                     # JoyEcho_TextEncode / _Generate / _ModelLoader / _LLMEnhance
__init__.py                                  # registrations for the new nodes
rebels_loaders.py                            # discrete GGUF loaders (text-encoder fixes)
joyecho_prompt_source.py   (new node)        # one dropdown: .txt briefs + .json scripts
joyecho_ref_picker.py      (new node)        # auto reference-image picker by character name
joyecho_ref_batch.py       (new node)        # None-tolerant image batcher
joyecho_script_picker.py   (new node)        # JSON dropdown (superseded by PromptSource)
libs/ltx_distillation/models/ltx_wrapper.py  # fp8 quantization passthrough
libs/ltx_core/loader/fuse_loras.py           # kohya-LoRA fusion + alpha scaling + fuse telemetry
libs/ltx_core/quantization/policy.py         # fp8_scaled_mm_torch policy (+ sm_89 gate)
libs/ltx_core/quantization/fp8_torch_mm.py   # native-fp8 Linear forward (torch._scaled_mm)
libs/ltx_distillation/utils.py               # tiled VAE decode
libs/ltx_distillation/inference/memory_multishot.py            # memory bank TRIM FIX (critical)
libs/ltx_distillation/inference/bidirectional_pipeline.py      # dtype hardening vs fp8 params
libs/ltx_distillation/inference/memory_bidirectional_pipeline.py # dtype hardening vs fp8 params
prompts/long_story_writer_system_prompt.md   # (optional) de-musicked + character-age edits
```

The files are interdependent - apply the whole set together, never cherry-pick
(a nodes.py newer than its libs/ raises AttributeError at load).

---

## Bug fixes

### 0. RoPE clock hardcoded to 24 fps — the ~10 s lip-sync cliff (CRITICAL)
Lip sync held for the first several seconds of a shot then progressively fell
apart, the mouth running steadily **ahead** of the audio, with the break
crossing visibility around **9.6 s into every shot** regardless of prompt,
model, or reference. This is the reason the pack's practical dialogue limit was
believed to be ~241 frames.

Root cause: `LTX2DiffusionWrapper.VIDEO_FPS` was a hardcoded class constant of
`24.0`, used to convert the video RoPE temporal coordinate from frames into
seconds. The **audio** RoPE is built in true seconds. Rendering at 25 fps
therefore ran the video positional clock 25/24 ≈ **4 % fast** against audio —
a linear divergence of ~0.04 s per second of runtime, i.e. roughly half a frame
of drift per second, accumulating without bound. At ~9–10 s it passes the
threshold where a viewer reads it as "not lip syncing".

Fix: the generate nodes now stamp the actual render fps onto the generator
before sampling — `JoyEcho_Generate` (main pass and the hires refine pass) and
`JoyEcho_SingleShotGenerate`. Any render at a consistent fps is now
rope-coherent end to end.

**Consequence: there is no ~241-frame shot limit.** Verified with 69 s and
105 s multishot masters. This was a pipeline bug, not a model property — no
LTX-2.3 or JoyAI-Echo checkpoint carries a short training-length cap here; the
temporal RoPE range is `positional_embedding_max_pos[0] = 20` (seconds),
identical in JoyAI-Echo, `ltx-2.3-22b-dev`, and `ltx-2.3-22b-distilled-1.1`.
Because the fix is in coordinate math rather than weights, it applies to every
checkpoint loaded through these nodes, merges included.
(`nodes.py` + `libs/ltx_distillation/models/ltx_wrapper.py`)

### 1. `enable_audio_memory=False` silently disabled ALL cross-shot memory
The pack computed `audio_memory_latent=None` when audio memory was off, and the
video **memory-bank save was gated on that latent being non-None** — so with
audio memory off (the standard anti-drone setting) the bank never filled and
cross-shot **identity** silently died (symptom: `memory_size=0` every shot even
with `memory_max_size=7`; a new face each shot).
Fix: memory storage is now unconditional; `enable_audio_memory` gates only the
audio-memory **injection** path. Verify: console `memory_size=` should climb
0,1,2,… capped at your `memory_max_size`. (`nodes.py`)

### 1b. Memory bank trim was a NO-OP whenever `memory_max_size <= num_fix_frames` (CRITICAL)
`PairedAudioVideoMemoryBank._trim()` computed `tail[-keep_tail:]` - and when
`keep_tail == 0` (e.g. the common max_size=3 / num_fix_frames=3 combo),
`tail[-0:]` is the WHOLE list, so the bank grew unbounded: every shot
conditioned on EVERY prior shot. Symptom: console `memory_size=` climbing
0,1,2,...,N-1 past your cap, and severe compounding quality degradation over
long runs (waxy skin, contrast crush, smearing by the late shots - the "gets
worse as it goes" failure). Fixed with a proper zero-tail branch + anchor
clamp; `memory_size=` now freezes at your cap. This one fix eliminated the entire
long-run degradation in our tests. (`libs/.../memory_multishot.py`)

### 2. GGUF text-encoder loader (`RebelsJE_TextEncoder`)
Two fixes so a text-only Gemma-3 GGUF loads cleanly:
- **meta-strip**: drop `vision_tower` / `multi_modal_projector` / `lm_head`
  (the text-only GGUF has no weights for them → "Cannot copy out of meta tensor").
- **device-unify**: pin the embeddings-processor to the encoder's actual device
  (GGUF Gemma runs on CPU while the connector was on cuda → addmm device mismatch).
- **fp8 gemma scale-key layouts**: the `our_fp8` swap only recognized its own
  export layout (bare module names + `.scale_weight`); standard HF/comfy-style
  fp8 gemma files (`<module>.weight` + `.weight_scale`, e.g. community
  abliterated builds) silently loaded with **zero modules swapped** — the
  encoder stayed bf16 with no indication. Both layouts are now accepted
  (per-tensor scalar scales; per-channel scales are skipped and those modules
  stay bf16), and a loud warning prints if a file matches neither.
(`rebels_loaders.py`)

---

## Features

### 3. Split per-domain negative lever (`JoyEcho_TextEncode`)
The DMD pipeline has no CFG, so the only steering lever is embedding-space.
Instead of one `negative_prompt`/`negative_scale` that steers both branches,
this splits it:
- `negative_prompt_video` / `negative_scale_video` — kills burned-in
  captions/subtitles. Working value ~0.5. **Above ~0.8 it over-rotates the
  video context and locks every shot to shot 1's composition** (scene-lock).
- `negative_prompt_audio` / `negative_scale_audio` — kills invented
  music/score. Keep ≤ ~0.4 or dialogue suffers.
Steering is norm-preserving (RescaleCFG-style): `cond' = renorm(cond + s*(cond − neg))`.
Old single-widget names still work as a fallback. (`nodes.py`)

### 4. Passthrough mode (`JoyEcho_LLMEnhance`)
`mode = "passthrough (raw JSON, skip LLM)"` — feed a finished
`{"prompts":[...]}` script straight through with no LLM call / no API key.
Auto-detects when `story_idea` already parses as that JSON. (`nodes.py`)

### 5. Reference-image conditioning — I2V-as-reference (`JoyEcho_Generate`)
New `reference_image` (IMAGE batch, up to 4). Identity references are prepended
as **video-only conditioning clips** at the memory-encode step — they are
**never** written into the paired audio/video bank. (An earlier attempt that
seeded refs into the bank with zero-filled audio latents injected loud
background noise with 2+ refs; video-only conditioning avoids it entirely.)
Also new: `head_trim_frames` (auto 8 with refs) drops the first N frames of each
shot, where the model morphs out of the reference/memory content. The trim is
applied once right after decode, so the final output, the per-shot preview
files, and any external concat of them stay frame-identical. (`nodes.py`)

### 6. Shot transitions (`JoyEcho_Generate`)
`transition`: `cut` (original) / `dissolve` (overlap cross-dissolve + equal-power
audio crossfade) / `vhs_glitch` (analog static burst at each boundary: snow,
tear bands, dropout lines + a raised-cosine tape-noise audio bed).
`transition_frames`, `glitch_intensity` tune it. (`nodes.py`)

### 7. fp8 transformer quantization (`JoyEcho_ModelLoader`)
New `fp8_transformer` toggle. Quantizes the DiT's attention/FF linear weights to
`float8_e4m3fn` **at load, from the normal bf16 checkpoint** (uses the vendored
`ltx_core.quantization.QuantizationPolicy.fp8_cast()` — upcasts per-layer at
inference). Roughly halves DiT weight memory and halves sequential-offload PCIe
traffic; keeps memory training + all tensors; VAEs/text-encoder/non-linears stay
bf16. Ignored when a GGUF DiT is selected (already quantized).
(`nodes.py` + `libs/ltx_distillation/models/ltx_wrapper.py` — new `quantization`
param; the quantized build path skips the post-load dtype cast that would
otherwise silently upcast fp8 back to bf16.)

### 8. Tiled VAE decode (`JoyEcho_Generate`)
Decoding a long high-res shot (e.g. 241f @ 1280×736) in one pass hard-aborts the
VAE decode on a 24–32 GB card (fatal cuDNN abort mid-conv, not a catchable OOM).
New `decode_tiling` (`auto`/`on`/`off`) routes decode through the vendored
`VideoDecoder.tiled_decode` — **temporal-only** 64-frame chunks with 24-frame
blended overlap (no spatial tiles → no spatial seams), streaming each chunk to
CPU. `auto` engages only above a size threshold, so small renders keep the
original single-pass decode bit-for-bit.
(`nodes.py` + `libs/ltx_distillation/utils.py` — `decode_benchmark_sample` gains
a `video_tiling_config` kwarg + `_decode_video_tiled_uint8`.)

### 9. Model dropdown (`JoyEcho_ModelLoader`)
New `model_file` combo lists every `.safetensors` / `.gguf` under the ComfyUI
`checkpoints` / `diffusion_models` / `unet` dirs. Pick a `.safetensors` → full
checkpoint (replaces `checkpoint_path`); pick a `.gguf` → DiT loaded from GGUF
while `checkpoint_path` still supplies the VAEs / vocoder / text connectors.
`"(use checkpoint_path)"` keeps the old typed-path behavior. A matching
`lora_file` dropdown lists every `.safetensors` under `models/loras`
(applied at `lora_strength` on the safetensors DiT path; ignored for GGUF). Plus a clear
early error if `gemma_path` is a `.gguf`/file/sidecar-less dir (this loader
needs the HF `gemma-3-12b-it` folder; GGUF Gemma only works via
`RebelsJE_TextEncoder`). (`nodes.py`)

### 10. LoRA loading hardening (`JoyEcho_ModelLoader` + `libs/.../fuse_loras.py`)
- A `lora_file` dropdown picks LoRAs from `models/loras` (existing
  `lora_strength` widget applies).
- Fusion now supports **kohya naming** (`lora_down`/`lora_up`) in addition to
  PEFT (`lora_A`/`lora_B`), with standard `alpha/rank` scaling — previously a
  kohya-named LoRA silently did NOTHING (zero keys matched, no warning).
- Fusion prints how many weights fused, and WARNS LOUDLY when a provided LoRA
  matched zero keys.
- The loader refuses **ComfyUI-quantized checkpoints** (`.comfy_quant` marker
  tensors, e.g. "fp8mixed learned" builds) with a clear error: this loader
  never applies their weight scales (the model would silently load mis-scaled)
  and LoRA fusion on them crashes with shape errors. Use bf16 checkpoints.

### 11. Automation / batching nodes (new)
- **`JoyEcho_PromptSource`** — one dropdown listing LPFF-style `.txt` briefs
  (from the inspire-pack prompts tree) **and** passthrough `.json` scripts
  (`input/joyecho_prompts/`). Multi-block briefs fan out like
  LoadPromptsFromFile. Emits `story_idea` (→ LLMEnhance) + `character`
  (→ RefPicker) + `count`. Replaces the LPFF→UnzipPrompt chain and lets you
  switch prompt sources with one dropdown instead of rewiring.
- **`JoyEcho_RefPicker`** — auto-selects a character reference image from a
  folder tree keyed by character name (a `character_pick` dropdown of the
  folder names, a typed/wired `character` string, or a prompt scan — dialogue
  mentions are stripped so only the on-screen subject wins). The dropdown
  survives model refreshes, an explicitly named character that matches no
  folder refuses to fall back to the prompt scan (a wiped/typo'd name can't
  silently become the wrong character's face), and the cache signature
  includes the prompt text (without it, ComfyUI could serve a cached pick
  from a previous queue item). `on_no_match=no_reference` returns nothing so
  a batch keeps running.
- **`JoyEcho_RefBatch`** — None-tolerant image batcher: combines up to 4
  optional IMAGE inputs (e.g. two RefPickers for a two-character shot), skips
  missing refs, resizes mismatched sizes to the first image, outputs `None` if
  all are missing (Generate then just skips identity seeding). The stock KJNodes
  `ImageBatchMulti` crashes with `'NoneType' has no attribute 'shape'` on a
  missing ref; this replaces it.
- **`JoyEcho_ScriptPicker`** — JSON dropdown (superseded by PromptSource; kept
  for compatibility).

### 12. GPU encode hot-swap (`JoyEcho_TextEncode`)
With `low_vram` the Gemma encoder used to encode every shot on CPU (~10s+ per
shot). The encode pass now borrows the (idle) GPU when the encoder fits free
VRAM - with a fits-check, an OOM fallback to CPU, and a move-back before the
denoise phase. 20-shot encodes drop from minutes to seconds. (`nodes.py`)

### 13. `encoder_fp8` (`JoyEcho_ModelLoader`)
Stores the Gemma encoder's linear weights as float8_e4m3fn with per-layer
upcast at encode (encode runs once per item, so the upcast tax that makes
fp8 slow on the DiT is irrelevant here). Wrapper drops ~24GB -> ~21GB and the
GPU hot-swap engages on 32GB cards; JD's connector projections stay bf16.

### 14. `fp8_scaled_mm` (`JoyEcho_ModelLoader`) - native fp8 compute
Stores the DiT's attention/FF linears as fp8 AND runs the matmuls natively
via `torch._scaled_mm` - no per-layer upcast tax (measured x2.8 raw kernel /
x1.5 end-to-end vs bf16 on an RTX 5090). ~22GB resident enables
`sequential_offload=False` at moderate resolutions. REQUIREMENTS: sm_89+
GPU (RTX 40/50 - clear error on older cards, with a per-device runtime
fallback to upcast), and a **bf16 source checkpoint** (an fp8 FILE would load
every tensor fp8 with the cast skipped and crash the noise path - guarded
with a clear error). Tensorwise dynamic activation quant: A/B your content
before adopting.

### 15. `resident_blocks` (`JoyEcho_Generate`)
Sequential offload middle ground: pin the first N of 48 transformer blocks
permanently on GPU, stream the rest. N=24 halves the per-step PCIe traffic;
raise until VRAM is nearly full. Composes with fp8 modes (fp8 blocks are
half the bytes both resident and streamed).

### 16. Hires-fix second pass (`JoyEcho_Generate`)
`hires_factor` (>1.0) + `hires_denoise`: after all shots render, each shot is
bicubic-upscaled, VAE re-encoded, re-noised at a tail sigma and re-denoised
through the DMD ladder at the TARGET resolution - the model synthesizes real
detail (RTX-class upscalers only sharpen what exists). Runs in 65-frame
windows with cross-fade (a 24GB card survives 1920x1088 refines); memory
bank and per-shot previews stay base-res; failures fall back to the base
frames. Audio is untouched.

### 17. Reference scheduling upgrades (`JoyEcho_RefPicker` + `_Generate`)
- Script-carried ref pinning: `{"prompts": [...], "refs": {"zara":
  "zara_file.png"}}` pins a scene-matched reference per character (a
  full-scene ref SETS the render's scene - match it to the script).
- Re-entry injection: a character returning after a 3+-shot absence gets
  their ref re-injected at the return shot automatically (the rolling memory
  window is 4; long absences otherwise re-invent the character).
- Generate's ref dedup is schedule-aware (the same image scheduled at two
  shots survives; cap 6 scheduled entries).

### 18. Robustness
- Pipelines no longer derive their working dtype from
  `next(parameters()).dtype` (an fp8 first-param crashed `torch.randn`);
  fp8 dtypes are skipped with a bfloat16 fallback.
- fp8 gemma swap accepts both `.scale_weight` and `.weight_scale` layouts
  and warns loudly on zero matches instead of silently staying bf16.

### 19. Finishing: who builds your master (READ THIS before touching hires)
`hires_factor` is a ROUTING switch, not a quality slider - it decides which
pipeline builds your final video:

| you want | hires_factor | hires_denoise | master comes from |
|---|---|---|---|
| **default: zero detail-shimmer** | **1.0** | (ignored) | base shots, upscaled by AutoFinish (bicubic + contrast-adaptive sharpen - deterministic, seconds per shot) |
| synthesized detail (pores/hair), accepts slight per-frame texture shimmer | 1.5 | subtle / medium / strong | your refined shots, used as-is (AutoFinish skips its own upscale) |
| deterministic upscale baked into the shot files | 1.5 | spatial | spatial-upscaled shots - EVEN latent grids only (height AND width /32 must be even: 768-height yes, 736 no) |
| the old RTX path | 1.0 | (ignored) | base shots via RTXBatchVideoUpscale (`upscale_mode: rtx (legacy)` on the AutoFinish node) |

- **Judge and publish only the `*_MASTER.mp4`.** The in-graph SaveVideo
  output (prefixed `PREVIEW` in the shipped workflow) is a convenience
  preview: its re-encode is bit-starved and shows artifacts the master does
  not have. This is a ComfyUI limitation, not a render problem.
- Resolutions: everything works at any /32 size; only the `spatial` hires
  mode additionally needs the /32 result EVEN on both axes (1280x768 and
  1344x768 qualify; 1280x736 and 1536x864 do not - spatial will smear one
  edge and warn in the console).
- All other widgets are genuinely free: seed, num_frames (long shots are
  fine - the old ~10s lip-sync limit was the fps bug, now fixed), fps,
  memory sizes, head_trim.
- The master pipeline re-encodes with `bf 0` + `tune grain` end to end, so
  masters never reintroduce B-frame pumping.

---

## Applying

1. Back up your existing pack folder.
2. Copy each file over the same relative path in
   `ComfyUI/custom_nodes/ComfyUI_JoyAI_Echo_GGUF_Nodes/`.
3. Restart ComfyUI. New widgets append at the **end** of existing nodes, so
   saved graphs keep their values; the four new nodes appear under the
   `JoyAI-Echo` category. Press `R` after adding model files to refresh the
   `model_file` dropdown.

The `libs/` files must match the vendored `ltx_core` / `ltx_distillation` in
your pack (same JoyAI-Echo release). If your `libs/` differ substantially,
cherry-pick the changes described above rather than overwriting.

Not included (intentionally): model weights, the `gemma_assets/` tokenizer
binaries, `.bak` snapshots, and `__pycache__`.

---

## Where the models live

Nothing in this repo is a model. Weights are on Hugging Face:

| what | where |
|---|---|
| Surgical merge (bf16 / fp8) | https://huggingface.co/joeygambino/joyai-echo-ltx23-echoVid-ltxAud-surgical |
| Surgical merge GGUF (Q8_0 / Q5_0 / Q4_0) | https://huggingface.co/joeygambino/joyai-echo-ltx23-echoVid-ltxAud-surgical-gguf |
| Surgical merge INT8 ConvRot (stock ComfyUI loaders, **not** this pack) | https://huggingface.co/joeygambino/joyai-echo-ltx23-echoVid-ltxAud-surgical-int8 |
| Gemma-3-12B text encoder GGUF (Q8_0 / Q4_0) | https://huggingface.co/joeygambino/joyai-echo-gemma3-12b-encoder-Q8_0-gguf |
| This pack, mirrored | https://huggingface.co/joeygambino/joyai-echo-multishot-workflow |
| Everything | https://huggingface.co/joeygambino |

Civitai mirrors: [GGUF builds](https://civitai.com/models/2796109) - [workflow](https://civitai.com/models/2780640)

## License and scope

This repository contains **only my patch files** - it is not a redistribution
of the underlying pack or of any model weights.

- The patched pack wraps **JoyAI-Echo**, which is **research / non-commercial**.
  That is the strictest term in the stack and it governs your outputs.
- **LTX-2 / LTX-2.3** is under the
  [LTX-2 Community License](https://huggingface.co/Lightricks/LTX-2/blob/main/LICENSE.txt).
- **Gemma 3** - the text encoder, and the `gemma_assets/` tokenizer sidecars
  bundled here because the loader requires them - is subject to the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- AI-generated content produced with this stack must be disclosed as such.

Not affiliated with Lightricks, JD, RealRebelAI, Comfy-Org, or Google.

## Credits

This patch stands on other people's work:

- **JD Joy Future Academy** - [JoyAI-Echo](https://huggingface.co/jdopensource/JoyAI-Echo), the multishot memory model this whole stack serves (research / non-commercial license).
- **Lightricks** - [LTX-2 / LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (LTX-2 Community License).
- **TenStrip** - the [LTX2.3 DMD LoRAs](https://huggingface.co/TenStrip/LTX2.3_DMD_Lora); the hires `strong (tenstrip 4-step)` mode uses his published upscale sigma ladder verbatim.
- **RealRebelAI** - the Rebels GGUF loader stack this patches, and the [Q6_K_RM GGUF](https://huggingface.co/realrebelai/JoyAI-Echo_GGUF) whose tensor canon the self-built GGUFs mirror.
- **Comfy-Org** - the comfy-quants `int8_tensorwise` + ConvRot export recipe behind the INT8 checkpoint.
- **Google** - Gemma 3 12B, the text encoder (Gemma license).

## Support

Everything here is free and stays free. If it saved you time, you can
[buy me a coffee](https://ko-fi.com/joeygambino) or
[support me on Liberapay](https://liberapay.com/joeygambino).
