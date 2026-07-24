# JoyAI-Echo Multishot — Full Instructions

This is the complete, step-by-step guide. If you only read one file, read
this one. It covers install, your first render, where the reference image
goes, LoRAs, performance settings, and every common failure with its fix.

---

## 1. What this workflow actually is (read this first — it prevents confusion)

This is a **multi-shot text-to-video** pipeline with **joint audio** (voice and
picture generated together, not dubbed). You write one or more SHOTS; each shot
renders separately, and a **cross-shot memory bank** carries your character's
face, wardrobe, and voice from shot to shot. The thing that makes it work is
simple but strict:

> **Repeat your character's description sentence WORD-FOR-WORD IDENTICAL in
> every shot.** The memory bank keys on it. Reword it even slightly and the
> face drifts.

It is **not** a classic image-to-video workflow. You do not have to provide an
image at all — text alone holds identity. You *can* add a reference image to
lock the face harder (section 6), but it works differently from normal i2v:
it pre-seeds the memory bank rather than becoming frame one.

---

## 2. What you need

| thing | where it goes | notes |
|---|---|---|
| RealRebelAI's ComfyUI_JoyAI_Echo_GGUF_Nodes | `ComfyUI/custom_nodes/` | the BASE pack — install it first |
| This patch (the folder in this zip) | copied OVER the base pack | replace files when asked |
| A model build (see table below) | `ComfyUI/models/diffusion_models/` (safetensors) or `models/unet/` (GGUF) | one of: bf16, fp8, Q8_0, Q5_0 |
| A FULL checkpoint (bf16 or fp8) | `ComfyUI/models/diffusion_models/` | **required even when using a GGUF** — the GGUF is the transformer only; the VAEs and vocoder load from the full checkpoint |
| Gemma-3-12B-it text encoder | `ComfyUI/models/text_encoders/` | use TRUE BASE Gemma — single-file safetensors or the clean Q8_0 encoder GGUF. **Do not** use the circulating "joyecho" Gemma GGUF: it is a different model and produces duplicated people |
| ComfyUI-Easy-Use + ComfyUI-Custom-Scripts | `custom_nodes/` | two helper packs used by two convenience nodes; or delete those two nodes |

Which model build:

| you have | use |
|---|---|
| RTX 40/50-series, 24 GB+ | fp8 (fast) or Q8_0 GGUF (highest quant fidelity) |
| RTX 30-series or older, 24 GB | Q8_0 GGUF |
| 16 GB card | Q5_0 GGUF |
| 32 GB+ and patience | bf16 |

## 3. Install

1. Install the base pack (RealRebelAI's) into `ComfyUI/custom_nodes/` if you
   have not already.
2. Copy this zip's `ComfyUI_JoyAI_Echo_GGUF_Nodes` folder into
   `ComfyUI/custom_nodes/`, REPLACING files when asked.
3. Restart ComfyUI.
4. Open `workflow/JoyEcho_Multishot_Workflow_PUBLIC.json` (drag it onto the
   ComfyUI canvas, or Workflows > Open).

## 4. First render in five minutes

1. Copy `workflow/example_multishot.json` into `ComfyUI/input/joyecho_prompts/`
   (create the folder if it does not exist).
2. In the **Model Loader** node: pick your DiT in `model_file` (GGUF or
   safetensors). If you picked a GGUF, ALSO set `checkpoint_path` to your full
   bf16/fp8 checkpoint. Pick your Gemma in `gemma_file`.
3. In the **Prompt Source** node: pick `example_multishot.json`.
4. Queue. First run encodes the prompts (a few minutes); every later run of the
   same prompts skips encoding entirely (you will see `Conditioning cache HIT`).

Where your output goes — THREE places, by design:

| file | what it is |
|---|---|
| `output/joyecho/shot_000.mp4`, `shot_001.mp4`, ... | each shot, saved the moment it finishes (watch progress live) |
| `output/output_#####.mp4` | the assembled full video from SaveVideo (native resolution, with shot transitions) |
| `output/joyecho/<name>_<timestamp>_MASTER.mp4` | the upscaled master, built automatically in the background a few minutes after the run ends |

## 5. Writing your own shots

Your prompt file is either a `.json` file: `{"prompts": ["shot 1 text",
"shot 2 text", ...]}` — or a `.txt` in the same format the example shows. Rules
that matter:

- **The identity sentence is byte-identical in every shot.** Copy-paste it.
  Include: appearance, wardrobe, and a voice description WITH an explicit
  accent ("in a casual American accent" — the model defaults to British if you
  do not say otherwise).
- One shot is ~12.5 seconds at the default 313 frames. A comfortable spoken
  line for one shot is **about 20-30 words**, in double quotes.
- For talking shots, keep the framing **medium close-up or tighter** and the
  resolution height 864+. In wide shots the mouth is smaller than the model
  can resolve and lip sync suffers no matter what you write.
- Describe sound affirmatively in every shot ("Quiet diegetic sound only:
  rain on the window and a ticking radiator") — otherwise the model invents
  drones and music.
- Do not write "no X" in a prompt — it renders the X. State what IS there.

## 6. THE REFERENCE IMAGE — "where do I insert the source image?"

As of v1.4 the graph ships with a **Reference Image (OPTIONAL)** node just
below the Generate node, deliberately unconnected - unconnected nodes never
run, so it is safe to ignore. To use it: load your image there and drag its
IMAGE output to the `reference_image` input on the left edge of
JoyEcho_Generate. If you are on an older graph without the node:

1. Right-click the canvas > **Add Node > image > Load Image**.
2. Load your character image (a portrait render or photo-style still works
   best — clear face, front or three-quarter angle).
3. Drag from the Load Image node's **IMAGE** output to the **`reference_image`**
   input socket on the left edge of the **JoyEcho_Generate** node.
4. Queue as normal.

What it does: the image is planted into the cross-shot memory bank as a
permanent anchor, so EVERY shot is conditioned on that face and look — on top
of (not instead of) your identity sentence. Keep the identity sentence anyway,
and make it describe the same person as the image.

The `reference_zoom` widget (default 1.2) crops the reference in ~20% before
conditioning. Why: a reference at exactly the render size is treated as
continuable content — the first shot opens ON your image like classic i2v,
which usually is not what you want in a multi-shot piece. 1.2 keeps the
identity and drops that behavior. If you WANT shot 1 to start from your image
(true i2v-style opening), set `reference_zoom` to 1.0.

Advanced: the RefPicker node can auto-load reference folders per character by
name (a folder `ComfyUI/input/joyecho_refs/alice/` is used whenever "alice"
appears in your prompts). For single-character work the Load Image route above
is simpler.

## 7. LoRAs

As of v1.4 the loader takes **multiple LoRAs**, and the workflow ships with
the node for it: the **LoRA Stack** node, already wired into the Model Loader.

- Pick up to three LoRAs in its dropdowns, each with its own strength slider.
  Slots left on `(none)` do nothing.
- Need more than three? Add another LoRA Stack node (Add Node > JoyAI-Echo >
  JoyEcho LoRA Stack) and chain it: first node's `lora_stack` output into the
  second node's `lora_stack` input, second node into the Model Loader.
- Everything stacks with the Model Loader's own `lora_file` pick.

Power users / automation: `lora_path` also accepts a text list — one entry per
line or comma-separated, each with an optional strength suffix
(`my_style.safetensors@0.7`). Same result, no extra nodes.

All LoRAs are fused into the model at load, so there is no per-step cost.
Note: LoRAs apply on the **safetensors DiT path** only — they are ignored when
a GGUF is selected as the DiT (existing limitation, stated in the tooltip).

## 8. Performance settings by VRAM

| widget | what to set |
|---|---|
| `sequential_offload` | ON below 32 GB VRAM. Streams transformer blocks over PCIe. |
| `resident_blocks` | With offload ON: how many of the 48 blocks stay on the GPU. Rough guide: 24 GB card + fp8/GGUF -> ~20-24; 16 GB -> 8-12; raise until VRAM is nearly full. |
| `fp8_scaled_mm` | RTX 40/50-series ONLY. Native fp8 compute, real speedup. Leave OFF on 30-series and older. |
| `low_vram` (encoder) | ON if Gemma encoding OOMs; slower but ~3 GB. |
| resolution | 544x960 is a fast preview. For dialogue, raise height to 864+. |
| `hires_factor` | 1.0 = off (default). The background AutoFinish upscale usually covers you. |
| `head_trim_frames` | Leave at 14 (default). Trims the unstable first ~0.5s of each shot, where identity morph and lip-sync error concentrate. |

## 9. After updating this patch — one-time step

Open your workflow, glance at the LAST few widgets on the Generate node
(reference_zoom / resident_blocks / hires), fix anything that looks wrong, and
SAVE. ComfyUI stores widget values by position; a graph saved under an older
pack version can load a few trailing widgets scrambled until you re-save it
once. If you ever find yourself re-entering the same two or three values after
every restart — this is why, and one save ends it.

## 10. Troubleshooting

| symptom | cause / fix |
|---|---|
| Error: "checkpoint_path is empty" | You picked a GGUF in model_file. GGUFs are the transformer only — set checkpoint_path to a full bf16/fp8 checkpoint too. |
| Two copies of your character / wrong wardrobe / garbled identity | Wrong text encoder. Use TRUE BASE Gemma-3-12B-it (safetensors or the clean Q8_0 encoder GGUF). The "joyecho" Gemma GGUF is a different model — replace it. |
| Face drifts between shots | Your identity sentence is not byte-identical across shots. Copy-paste it; do not reword. |
| Voice is British | State the accent in every shot: "in a casual American accent". |
| Random music / horror drones | Add an explicit sound line to every shot; leave `negative_scale_audio` at 0.3. |
| Lip sync poor in wide shots | Physics of resolution: mouth too small. Medium close-up + height 864+ for talking shots. |
| Widgets reset every restart | Section 9 — re-save the graph once. |
| First run very slow | Normal: prompt encoding + caches build. Subsequent runs with the same prompts skip encoding entirely. |
| Master video missing | It arrives a few minutes AFTER the run ends (background worker). Check `output/joyecho/` for `*_MASTER.mp4` and the `_autofinish_*.log` next to it. |
| OOM during generation | sequential_offload ON, lower resident_blocks, lower resolution, low_vram ON for the encoder. |

---

Model downloads, measurements, and a live demo Space are linked from the
model listings. AI-generated content must be disclosed as such (LTX-2
Community License; the JoyAI-Echo component is research / non-commercial).
