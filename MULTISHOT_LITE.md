---
license: other
license_name: ltx-2-community-license
license_link: https://huggingface.co/Lightricks/LTX-2/blob/main/LICENSE.txt
tags:
  - comfyui
  - workflow
  - ltx-video
  - ltx-2.3
  - text-to-video
  - talking-head
  - lip-sync
  - audio
---

# LTX-2.3 Multishot Lite

**Two-shot talking-character video on LTX-2.3 — shot 2 is a TRUE audio+video
extension of shot 1** (not a cut), the character
speaks in a reference voice you supply, and both shots are joined and refined
into a single final file.

Needs ComfyUI, the models, and ComfyUI-KJNodes (one node powers the extension). Everything else is core.

> **A word on expectations.** This is a bleeding-edge pipeline — a 22B
> audio+video model running on consumer hardware. It works, but your first
> clean render will likely take some tuning to YOUR machine (VRAM, system RAM,
> which model build). Every setting that matters is documented in note blocks
> *inside* the workflow, so the guidance travels with the graph. **If you get
> stuck, open a discussion — I answer.**

---

## Quick fixes — read this first

Nearly every problem reported with this workflow is one of these.

| symptom | what's actually wrong | fix |
|---|---|---|
| **Second shot doesn't lip-sync** — voiceover over a barely-moving face | You are on **v1.x**, which chained shots on shot 1's decoded last frame. That guide is pixel-continuable, so the sampler copies it instead of animating. Structural — no setting fixes it. | Update to **v2.0+**. Shot 2 is now a true audio+video latent extension. |
| **`LTXVAudioVideoMask` node is missing (red)** | v2 uses one node from **ComfyUI-KJNodes** for the extension. | Install ComfyUI-KJNodes. Everything else is core ComfyUI. |
| **Burned-in subtitles or captions appear** | The graph ships at **cfg 1**, and at cfg 1 negative prompts are **inert** on distilled models — your negative is doing nothing. | Raise cfg to about **1.3** on both `CFGGuider` nodes. |
| **Voices come out British or Australian and IGNORE accent wording** | Frame rate. LTX-2.3's joint AV prior is 24 fps-native, and this graph shipped at 25 through v2.0 — off-24 fps drifts every voice Commonwealth-ward (25 → southern British, 30 → broad Australian, A/B-verified 2026-07-30) and geometrically overrides accent prose. | Update to **v2.1** (ships at 24 fps), or set every fps widget in your graph to **24** — both `LTXVConditioning`, `LTXVPreprocess`, `LTXVEmptyLatentAudio`, all `CreateVideo` nodes, and the AV-extend mask (fps 24, times 3.04 / 13.04). THEN name the accent in the *positive*: "in a casual American accent". |
| **The model reads your scene description aloud** | Nothing told it which words are spoken. | Keep the exclusivity block: "She is the only person speaking, and the only voice on the audio track is hers … nothing else is read aloud." |
| **Audio and video durations drift apart after changing length** | In **v1** the audio latent did not follow the `length` primitive — a silent desync. | Fixed in v2. Change `length`, then set the AV-extend node's `video_end_time` / `audio_end_time` to **`(length + 72) / 24`**. At the shipped 241 that is 13.04; for ~20 s use length 265 and 14.04. |
| **Length change errors out** | LTX-2.3 hard constraints. | Frames must be **8n+1** (121, 241, 265, 329…) and both dimensions divisible by **32**. |
| **Output is mush, or the face wanders** | Wrong checkpoint class. | This graph's 8-step sigma ladder assumes a **distilled** LTX-2.x checkpoint. A non-distilled one will not resolve in 8 steps. |
| **Out of VRAM, or renders crawl** | A bf16 LTX-2.3 checkpoint is ~40 GB of weights. | Pick an **fp8** checkpoint in `UNETLoader` (or set `weight_dtype` to an fp8 option), and drop resolution before you drop steps. |

**The one people miss most:** cfg 1 makes your negative prompt do nothing. If you are fighting burned-in text, that is why.

## v2.1 — 24 fps: your frame rate was choosing your accents

The graph shipped at 25 fps through v2.0. It turns out LTX-2.3's joint
audio-video prior is 24 fps-native and render fps is a hidden **accent
dial**: at 25 fps the same prompt and seed render non-rhotic southern
British, at 30 fps broad Australian — and any off-24 setting overrides
accent wording in the prompt entirely (a few conditioning tokens cannot
outvote a geometric signal present in every attention operation at every
denoising step). Dose-response verified by A/B on identical configs,
2026-07-30: 24 fps = rhotic General American at full reviewer confidence.

v2.1 changes every fps-carrying widget to 24 (`LTXVConditioning` ×2,
`LTXVPreprocess`, `LTXVEmptyLatentAudio`, `CreateVideo` ×3) and rescales the
AV-extend mask times so the extension window stays on the same latent frames
(73 / 313 → 3.04 s / 13.04 s at 24 fps). Duration math is now `length / 24`;
the extend end-time formula is `(length + 72) / 24`.

Flip side, free feature: if you *want* an authentic British or Australian
character, render their scenes at 25 or 30 fps — it is more consistent than
any accent wording. And at any fps, still name the accent on every spoken
line: with nothing stated, young female voices lean Australian even at 24.

## v2.0 — the chained shot is now a TRUE extension (fixes chained lip sync)

**If you use two shots, this release matters more than everything before it
combined. v1.x's second shot was structurally broken and no setting could fix
it.** What we found, in code:

* `LTXVAddGuide` does not "set frame 0" — it **appends the guide as a competing
  token at the same timestamp** and, at `strength 1.0`, skips its attention
  mask entirely. The official Lightricks workflows use 0.7 for exactly this
  reason.
* v1 guided shot 2 with shot 1's own decoded last frame — VAE-round-tripped, at
  exactly the render size, **pixel-continuable**. Reproducing it verbatim is a
  cheaper solution for the sampler than lip-syncing. That is why chained shots
  came out as "voiceover over a barely-moving face", and why it was WORSE on
  strongly distilled checkpoints.
* Shot 2's audio restarted from pure noise mid-sentence while its video was
  pinned — two contradictory anchors every render.

**v2 replaces the handover with a real audio+video extension** (the pattern
community extend workflows and Lightricks' own hosted `/extend` use): the last
~3 s of shot 1 — video AND audio — are encoded as latent context, and the model
generates forward from an ongoing utterance. The voice carries over by
construction (the chained shot no longer needs a reference-audio node at all),
identity holds through the join, and the join is a continuous take instead of
a cut.

Measured on the shipped demo config: chained-shot audio/mouth correlation went
from ~0.1–0.4 (v1, wildly unstable) to **0.64** — and for the first time the
second shot syncs *better* than the first.

**One dependency change:** the extension node (`LTXVAudioVideoMask`) ships in
**ComfyUI-KJNodes**. Everything else is still core. If you work with LTX you
almost certainly have KJNodes already; "no custom nodes at all" is now
"core + one node from KJNodes", and it buys working multishot.

Also in v2.0: `strength` 0.7 everywhere (official value — 1.0 disables the
guide's attention mask), and every length-dependent value now follows the one
`length` primitive automatically (negative indexing + clamped trims), so a
20-second render is a one-widget change.

## Changes in v1.3

**v1.2 gave bad tuning advice. This corrects it.**

v1.2 called `img_compression` "the motion dial" and told you to raise it to
50-70 for more movement. Both halves were wrong, and here is the measurement:

* That widget is passed **straight to x264 as CRF**. x264 clamps CRF at 51, so
  every value from **51 to 100 is byte-identical** - if you ran it at 100 you
  were not at maximum motion, you were at CRF 51 and the rest of the slider did
  nothing.
* At CRF 51, fine high-frequency detail drops to ~83% of source. That is where
  patterned fabric, curtains, hair and skin texture go: the model receives mush
  where the pattern was and **invents a replacement**. It restyles the whole
  frame, not just the face.

It now ships at **25** (visually near-lossless) and is documented as guide
degradation, not a motion dial.

**The real first question is your checkpoint.** Lip sync does not live in the
audio branch - it lives in the video branch's cross-attention to the audio
conditioning, and video-side merges rewrite exactly those layers. Comparing one
such merge against stock `ltx-2.3-22b-distilled-1.1`: the audio branch was
**bit-identical**, but only 2 of 24 sampled video cross-attention tensors were
unchanged. A checkpoint can be excellent at scenery and markedly worse at mouths
with its audio weights untouched.

The workflow now defaults to stock **`ltx-2.3-22b-distilled-1.1`**. If your
mouths are poor, A/B a stock checkpoint before touching any widget - and compare
against a *distilled* build, never `-dev`, or the 8-step ladder here will
produce garbage and wrongly indict it.

**`strength` returns to 1.0.** On a healthy checkpoint a fully pinned frame 0
animates fine and gives the tightest identity hold. Needing to drop below ~0.6
to get any movement is a checkpoint symptom, not a strength setting.

## Changes in v1.2

Mouth motion, voice consistency, and the dials to trade them off.

* **`LTXVPreprocess` motion dial (new, per shot).** LTX is trained on VIDEO
  frames, which always carry codec artifacts. A pristine photo is
  out-of-distribution as a "video frame", so the model treats it as a perfect
  anchor and barely animates. The guide is now round-tripped through an H.264
  encode/decode. `img_compression` 35 by default; raise for more motion, lower
  (20-25) if the result looks soft or stylized.
* **The voice chain (new).** Shot 2's voice reference is now **shot 1's own
  generated audio**, not the external clip. Two shots referencing the same clip
  still sample independently, so the voice drifted between them; feeding shot
  1's output forward is the core-node equivalent of a memory bank.
* **Identity guidance is now gated to the early steps** (`end_percent` 0.5).
  Guidance amplifies its effect on the WHOLE denoised tensor - audio AND video -
  so at full range it restyles the face while it fixes the voice. Identity is
  decided early; detail forms late. Gating keeps the voice lock without
  touching the detail passes.

All still 100% core ComfyUI nodes. A **MOTION DIALS** note block in the graph
documents every knob and which way to turn it.

## Changes in v1.1

**v1.0 had a broken reference-image path — please re-download.** `LoadImage`
returns the file's original resolution, not the canvas, so the 1.2x reference
zoom cropped a small patch from the top-left corner of large images instead of
centring on the subject. The guide was effectively meaningless and renders
behaved like text-to-video. Fixed: scale to exactly 1.2x canvas with a centre
crop, then centre-crop back down.

Also in v1.1, all aimed at mouth motion and voice consistency:

* **Guide strength 1.0 -> 0.9.** At 1.0 the first frame's noise mask is 0.0 —
  frame 0 is *completely locked* to the still, and the model struggles to
  animate away from a closed mouth.
* **Head trim: 28 frames on shot 1, 14 on shot 2** (audio trimmed to match).
  The opening frames morph out of the guide and are where lip sync is worst;
  shot 1 opens practically *on* the reference, so it needs more.
* **Identity guidance on the voice reference.** Both builds now use core
  `LTXVReferenceAudio`, which patches the model with an extra forward pass
  without the reference and amplifies the speaker difference. The voice no
  longer drifts between shots.

---

## What it does

```
SETUP → SHOT 1 → [last frame] → SHOT 2 → FINISH (join + refine) → FINAL
```

- **Chained shots.** `ImageFromBatch` pulls shot 1's final frame into shot 2's
  `LTXVAddGuide` at frame 0. The shots genuinely continue each other rather
  than being two unrelated clips.
- **Mode 1 (ships enabled) — your reference voice.** Point `LoadAudio` at 3–5
  seconds of clean speech; the character speaks your prompt's quoted line in
  that voice.
- **Mode 2 — the model invents a voice.** Bypass one node per shot (Ctrl+B).
- **One final file.** Both shots joined — video *and* audio — then refined with
  a deterministic 1.5× bicubic upscale. Saved as `multishot_lite/FINAL`.

## Quick start

1. Download **[`LTX23_Multishot_Lite_v2.0.zip`](./LTX23_Multishot_Lite_v2.0.zip)**
   (or just the workflow JSON) and open it in ComfyUI.
2. Set the six loaders in the **SETUP** group — see the table in
   [INSTRUCTIONS.md](./INSTRUCTIONS.md). Each reads a *different* models folder.
3. Put a 3–5 second voice clip and a first-frame image in `ComfyUI/input/`.
4. Queue. Judge `multishot_lite/FINAL`.

Full walkthrough, per-VRAM settings and troubleshooting: **[INSTRUCTIONS.md](./INSTRUCTIONS.md)**

## What you need

| thing | where |
|---|---|
| `ltx-2.3-22b-distilled-1.1` (or another **distilled** LTX-2.x build) | `models/diffusion_models/` |
| LTX-2.3 video VAE | `models/vae/` |
| LTX-2.3 audio VAE + text projection | `models/checkpoints/` |
| Gemma-3-12B text encoder | `models/text_encoders/` |
| `LTX-2.3-ID-LoRA-TalkVid-3K.safetensors` | `models/loras/` — **Mode 1 needs this** |

A recent ComfyUI (needs `ManualSigmas` and `ComfySwitchNode` in core). Tested on
an RTX 5090; smaller cards should drop resolution before anything else.

**System RAM matters as much as VRAM** — loading copies the whole checkpoint
into host memory. On a 64 GB machine set a Windows pagefile of 64–128 GB on an
SSD, or the load can be killed by the OS with no Python error.

## The three things that most often go wrong

**Voice reference length.** The *entire* clip becomes conditioning tokens, so a
long sample hands the model extra words to echo — you get your line *plus*
garbled fragments. Keep the trim at 3–5 seconds of clean, solo, continuous
speech. This is the single most common cause of bad audio.

**Missing ID-LoRA.** Reference-audio identity transfer is a trained adapter,
not a base-model skill. Without it the model echoes your clip instead of
adopting the voice.

**The narrator prior.** Only text inside `"double quotes"` should be spoken,
but the model will happily read your scene description aloud too. The shipped
prompts carry an explicit exclusivity block — *"no narration, no voice-over,
nothing else is read aloud"*. Keep it when you rewrite them.

## Honest limits

There is **no memory bank** here. Continuity comes only from the handed-over
frame, so identity drifts over many shots. A paired audio+video memory bank
cannot run on a stock sampler — that is exactly why the full
**[JoyAI-Echo multishot pack](https://huggingface.co/joeygambino/joyai-echo-multishot-workflow)**
exists. Use Lite for a quick two-shot piece with no dependencies; use the pack
when one character has to hold across a whole scene.

Long-form video is heavy: a 1001-frame shot at 960×544 holds tens of GB of
frames in system RAM during the join. Start at the shipped defaults.

## Also available

- **GitHub:** https://github.com/jlucasmcrell/ltx23-multishot-lite
- **The full node pack** (memory bank, many shots):
  https://huggingface.co/joeygambino/joyai-echo-multishot-workflow
- **Models:** https://huggingface.co/joeygambino

## Credits

- **Lightricks** — [LTX-2 / LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3);
  the sigma ladder and AV latent conventions here follow their reference
  workflow.
- **Google** — Gemma 3 12B, the text encoder.

## License

LTX-2 Community License. Gemma components are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). AI-generated content
must be disclosed as such. Not affiliated with Lightricks or Google.

## Support

Everything here is free and stays free. If it saved you time, you can
[sponsor me on GitHub](https://github.com/sponsors/jlucasmcrell),
[buy me a coffee](https://ko-fi.com/joeygambino), or
[support me on Liberapay](https://liberapay.com/joeygambino).
