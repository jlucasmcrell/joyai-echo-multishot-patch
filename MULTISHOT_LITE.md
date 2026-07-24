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

**Two chained talking-character shots on LTX-2.3, using only core ComfyUI nodes.**
Shot 2 opens on shot 1's exact last frame so they cut together, the character
speaks in a reference voice you supply, and both shots are joined and refined
into a single final file.

Nothing to install beyond ComfyUI and the models. No custom node packs.

> **A word on expectations.** This is a bleeding-edge pipeline — a 22B
> audio+video model running on consumer hardware. It works, but your first
> clean render will likely take some tuning to YOUR machine (VRAM, system RAM,
> which model build). Every setting that matters is documented in note blocks
> *inside* the workflow, so the guidance travels with the graph. **If you get
> stuck, open a discussion — I answer.**

---

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

1. Download **[`LTX23_Multishot_Lite_v1.1.zip`](./LTX23_Multishot_Lite_v1.1.zip)**
   (or just the workflow JSON) and open it in ComfyUI.
2. Set the six loaders in the **SETUP** group — see the table in
   [INSTRUCTIONS.md](./INSTRUCTIONS.md). Each reads a *different* models folder.
3. Put a 3–5 second voice clip and a first-frame image in `ComfyUI/input/`.
4. Queue. Judge `multishot_lite/FINAL`.

Full walkthrough, per-VRAM settings and troubleshooting: **[INSTRUCTIONS.md](./INSTRUCTIONS.md)**

## What you need

| thing | where |
|---|---|
| Any LTX-2.x checkpoint | `models/diffusion_models/` |
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
