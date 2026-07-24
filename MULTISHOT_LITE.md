# Multishot Lite — chained talking-character shots on stock ComfyUI

A **100% core-node** LTX-2.3 workflow: two chained shots where the second
continues from the first's last frame, the character speaks in a voice you
supply, and both shots are joined and refined into one final file.

No custom node packs. Nothing to install beyond ComfyUI itself and the models.

> **A word on expectations.** This is a bleeding-edge pipeline — a 22B
> audio+video model on consumer hardware. It works, but your first clean
> render will take some tuning to YOUR machine. Every setting that matters is
> documented in note blocks *inside* the workflow. If you get stuck, open a
> discussion — I answer.

---

## What it does

```
SETUP → SHOT 1 → [last frame] → SHOT 2 → FINISH (join + refine) → FINAL
```

* **Chained shots.** `ImageFromBatch` takes shot 1's final frame and feeds it
  to shot 2's `LTXVAddGuide` at frame 0, so the two cut together seamlessly.
* **Two audio modes.** Ships in Mode 1: the character speaks your prompt's
  quoted line **in a reference voice you supply**. Bypass one node per shot
  for Mode 2, where the model invents a voice.
* **One final file.** Both shots are joined (video + audio) and refined with a
  deterministic 1.5× bicubic upscale, saved as `multishot_lite/FINAL`.

**What it is not:** there is no paired audio+video memory bank here. Continuity
comes only from the handed-over frame, so identity drifts over many shots. That
is the honest ceiling of the "lite" approach — the full
[JoyAI-Echo multishot pack](https://huggingface.co/joeygambino/joyai-echo-multishot-workflow)
exists because a real memory bank cannot run on a stock sampler.

## Setup — six things

| # | node | what to set |
|---|---|---|
| 1 | Diffusion model | any LTX-2.x checkpoint (`models/diffusion_models/`) |
| 1b | ID-LoRA | `LTX-2.3-ID-LoRA-TalkVid-3K.safetensors` (`models/loras/`) — **Mode 1 needs this** |
| 2 | Video VAE | `models/vae/` |
| 3 | Gemma + text projection | `models/text_encoders/` + `models/checkpoints/` |
| 4 | Audio VAE | `models/checkpoints/` |
| 5 | Voice reference | **3–5 seconds** of clean, solo speech |
| 6 | First frame | the image shot 1 opens on |

**Each loader reads a different folder.** A model in the wrong one simply will
not appear in its dropdown.

## The settings that actually matter

**Length and resolution are set once**, in the GLOBAL group, and feed both
shots — they must match or the chain breaks. `length` must be **8n+1** (97,
241, 497, 1001…); duration = `length ÷ 25`. Width and height must be divisible
by 32. Ships at 960×544 × 241 frames ≈ 9.6 s per shot.

Three things are still per-shot because core ComfyUI has no math nodes:
`LTXV Empty Latent Audio → frames_number`, `ImageFromBatch → batch_index`
(= length − 1), and each shot's seed.

**Voice reference length is the #1 cause of bad audio.** The whole clip becomes
conditioning tokens, so a long sample hands the model extra words to echo —
"my line plus garbled extra words". Keep the trim at 3–5 seconds.

**Prompting.** Only text inside `"double quotes"` should be spoken, but the
model has a narrator prior and will happily read your scene description aloud
too. The shipped prompts carry an explicit exclusivity block ("*no narration,
no voice-over, nothing else is read aloud*") — keep it when you rewrite them.
~26–32 spoken words per shot, one speaker per shot.

## Requirements

Stock ComfyUI (recent enough to have `ManualSigmas` and `ComfySwitchNode`), an
LTX-2.x checkpoint, the matching VAEs and Gemma encoder, and the ID-LoRA for
Mode 1. Tested on a 32 GB RTX 5090; smaller cards should drop resolution first.

System RAM matters as much as VRAM — loading copies the whole checkpoint into
host memory. On 64 GB set a Windows pagefile of 64–128 GB on an SSD.

## License

LTX-2 Community License. Gemma components are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). AI-generated content
must be disclosed as such. Not affiliated with Lightricks, JD, or Google.
