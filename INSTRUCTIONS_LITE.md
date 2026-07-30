# Multishot Lite — full instructions

Everything here is also written into note blocks inside the workflow itself, so
you can work from the graph alone. This file is the long version.

---

## 1. Install

There is nothing to install. Every node in this graph ships with ComfyUI.

If a node shows up red on load, your ComfyUI is older than `ManualSigmas` or
`ComfySwitchNode` — update ComfyUI and reopen.

Open the workflow: **Workflows → Open**, or drag
`workflow/LTX23_Multishot_Lite.json` onto the canvas.

## 2. Models — six loaders, five different folders

This trips up nearly everyone: **each loader reads a different folder**, and a
model in the wrong one simply will not appear in its dropdown.

| # | node | file | folder |
|---|---|---|---|
| 1 | Diffusion model | `ltx-2.3-22b-distilled-1.1` or another **distilled** build | `models/diffusion_models/` |
| 1b | ID-LoRA | `LTX-2.3-ID-LoRA-TalkVid-3K.safetensors` | `models/loras/` |
| 2 | Video VAE | LTX-2.3 video VAE | `models/vae/` |
| 3 | Gemma + text projection | Gemma-3-12B **and** the LTX text projection | `models/text_encoders/` + `models/checkpoints/` |
| 4 | Audio VAE | LTX-2.3 audio VAE | `models/checkpoints/` |

Node **3** takes two files on purpose: Gemma produces the embeddings, and LTX's
text projection maps them into the DiT's conditioning space. Gemma alone is not
enough.

The `weight_dtype` widget on node 1 will drop the model to fp8 without needing
a separate file — useful on 24 GB cards.

## 3. Your two inputs

- **`5. Voice reference`** — 3–5 seconds of clean, solo, continuous speech.
  No music, no second speaker, no long silences. Put it in `ComfyUI/input/`.
- **`6. Shot 1 first frame`** — the image shot 1 opens on. Also `ComfyUI/input/`.

Both ship as `<YOUR ...>` placeholders, so they will show as unset until you
pick your own — that is intentional.

## 4. First render

1. Set the six loaders and your two inputs.
2. Leave everything else at defaults (960×544, 241 frames ≈ 9.6 s per shot).
3. Queue.
4. Watch `multishot_lite/FINAL` — that is the joined, refined output. The
   per-shot files are previews.

## 5. Audio modes

**Mode 1 (default) — your reference voice.** The character speaks the quoted
line from your prompt in the voice from your sample. Requires the ID-LoRA.

**Mode 2 — the model invents a voice.** Select **both** reference nodes (one
per shot) and press **Ctrl+B**. They turn purple. Everything else is unchanged.

There is no track substitution in either mode — what you hear is always what
the model generated.

### If the audio garbles

Symptom: your line comes out *plus* extra garbled words.

1. **Shorten the voice reference toward 3 seconds.** The whole clip becomes
   conditioning tokens; a long one gives the model words to echo. This is
   usually the entire fix.
2. **Confirm the ID-LoRA is loaded** (node `1b`). Without it the model has no
   trained path for identity transfer and echoes fragments instead.
3. Lower `identity_guidance_scale` toward 1.5 where that widget exists.
   Note that `identity_guidance_scale = 0` is **not** a clean "no reference"
   test — it only skips the guidance pass; the reference tokens stay attached.
   Only Ctrl+B removes them.

## 5b. Tuning: checkpoint first, then widgets

**Start with your checkpoint, not a slider.** Lip sync lives in the *video*
branch's cross-attention to the audio conditioning - not in the audio branch.
Video-side merges and finetunes routinely rewrite exactly those layers, so a
checkpoint can be excellent at scenery and markedly worse at mouths while its
audio weights sit untouched. Measured on one merge vs stock
`ltx-2.3-22b-distilled-1.1`: audio branch **bit-identical**, only 2 of 24
sampled video cross-attention tensors unchanged.

If mouths are poor, A/B a stock checkpoint first. Change only the `UNETLoader`,
keep the seed, change nothing else. **Match the schedule:** the sigmas here are
the 8-step distilled ladder at cfg 1, so compare against a *distilled* build -
never `-dev`, which needs a many-step cfg>1 schedule and will produce garbage on
this ladder and have you blaming the wrong thing.

**`strength` (guide node) - ships at 1.0.** Sets frame 0's noise mask to
`1.0 - strength`, i.e. how free the model is to move that frame. 1.0 pins it to
your guide and gives the tightest identity hold; on a healthy checkpoint it
still animates.

* Mouth frozen -> 0.8, then 0.7.
* Needing below ~0.6 for any movement -> checkpoint problem, see above.
* Face drifting off your reference / looks like T2V -> back toward 1.0.

**`img_compression` (LTXVPreprocess) - ships at 25. Not a motion dial.** LTX is
trained on *video* frames, which carry codec artifacts; a pristine photo is
out-of-distribution as a "frame", so a light H.264 round-trip helps the model
read it as one. But the value is passed **straight to x264 as CRF**, so it is a
quality-destruction knob:

* **x264 clamps CRF at 51 - every value from 51 to 100 is byte-identical.**
  Running it at 100 is not "maximum motion", it is CRF 51.
* At CRF 51 fine detail drops to ~83% of source. Patterned fabric, curtains,
  hair and skin texture turn to mush and the model invents replacements - it
  restyles the whole frame, not just the face.

Leave it at 25. If you are tempted past 40, the problem is elsewhere.

**`identity_guidance_scale` + `end_percent` - ship at 3.0 / 0.5.** Guidance
amplifies the reference across the WHOLE denoised tensor - audio *and* video -
so run across every step it restyles the face while fixing the voice. Identity
is decided in the early high-noise steps; detail forms late. `end_percent 0.5`
switches guidance off halfway.

* Face stylized -> lower `end_percent` (0.3-0.4), **not** the scale.
* Voice not holding -> raise the scale (4-5), leave `end_percent`.

**Change one thing at a time.** Every wrong turn in this workflow's history came
from moving two dials at once and crediting the wrong one.

## 6. Length and resolution

Both are set **once** in the GLOBAL group. `length` must be **8n+1** (97, 145,
241, 265, 329...), duration = `length / 24`, dimensions divisible by 32.

Everything downstream of `length` now follows it automatically (the last-frame
picker uses negative indexing, the trims clamp). **The one manual value:** the
AV-extend node's `video_end_time`/`audio_end_time` = `(length + 72) / 24`.
At the shipped 241 that is 13.04; for a ~20 s shot set length 265 and end times
14.04.

## 7. Prompting

Only text inside `"double quotes"` should be spoken. But the model has a
narrator prior and will read your scene description aloud unless told not to —
quoting alone is not enough.

The shipped prompts carry an exclusivity block:

> They are the only person speaking, and the only voice on the audio track is
> theirs. One continuous line with natural breaths and brief pauses. **No
> narration, no voice-over, no other speech, and nothing else is read aloud.**

Keep that when you rewrite. Other rules that carry over:

- ~26–32 spoken words per shot. Longer and delivery drifts.
- **One speaker per shot.** Two quoted lines in one prompt tend to come out
  simultaneously, in the same voice.
- Name the accent explicitly ("a casual American accent") — it drifts
  otherwise.
- If you are using a first-frame image, describe what should **happen**, not a
  fixed person or place. A prompt that contradicts the image makes the model
  fight the guide, which reads as the image being ignored.

## 8. The chain (v2: a true extension)

Shot 2 does not restart — it **extends** shot 1. The last 73 frames (3.04 s) of
shot 1's video and audio are encoded as latent context on the AV-extend node
(`LTXVAudioVideoMask`, `pad` mode), the model generates forward from an ongoing
utterance, and the context region is dropped from the output before the join.

Because the audio never restarts, the voice carries over automatically — the
chained shot has **no reference-audio node** and needs none.

**To add a shot 3:** copy the whole SHOT 2 group and feed its two context nodes
from shot 2's decodes instead of shot 1's. Identity drifts slowly over many
extensions; for long chains use the full JoyAI-Echo pack (memory bank).

## 9. The FINISH group

Both shots are joined (video and audio) and refined with a deterministic 1.5×
bicubic upscale plus a gentle sharpen. Deterministic on purpose — generative
refiners re-invent detail per frame, which reads as texture shimmer.

- Turn refine off: set `scale_by` to 1.0.
- **The sharpen node ships bypassed.** Core `ImageSharpen` moves the whole
  batch to the GPU and allocates two more full-size copies, so it needs roughly
  3× the batch in VRAM — fine for a few hundred frames, an OOM on a 2000-frame
  run at any resolution. Ctrl+B to enable it on short renders. For long ones,
  sharpen the finished file instead:

```
ffmpeg -i FINAL.mp4 -vf "cas=0.55" -c:v libx264 -crf 16 -preset medium \
       -tune grain -bf 0 -c:a copy FINAL_sharp.mp4
```

- Shot 2's first frame **is** shot 1's last frame, so the joint holds one
  duplicated frame — 1/24th of a second, invisible at a cut.

## 10. Troubleshooting

| symptom | cause |
|---|---|
| A loader dropdown is empty / red | model is in the wrong folder — see §2 |
| Line spoken *plus* garbled words | voice reference too long, or ID-LoRA missing — §5 |
| The model reads your description aloud | exclusivity block removed from the prompt — §7 |
| Mouth barely moves | A/B a stock distilled checkpoint first — see §5b |
| Face soft / stylized / "smoothed" | lower `img_compression`, then `end_percent` — §5b |
| Shot 2 looks unrelated to shot 1 | `ImageFromBatch → batch_index` ≠ `length − 1` |
| ComfyUI dies during load, RAM at 100%, VRAM idle | system RAM — raise the Windows pagefile to 64–128 GB |
| OOM in `ImageSharpen` on a long render | expected; leave it bypassed and use ffmpeg — §9 |
| Video looks soft | `scale_by` is doing a plain upscale; enable sharpen on short runs |
