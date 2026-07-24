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
| 1 | Diffusion model | any LTX-2.x checkpoint | `models/diffusion_models/` |
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

## 5b. Tuning motion vs. fidelity

Two dials trade against each other. Both ship at sane defaults; touch them only
if the symptom below matches.

**`img_compression` (on `LTXVPreprocess`, one per shot) — ships at 35.**
LTX is trained on *video* frames, which always carry codec artifacts. A pristine
photo is out-of-distribution as a "video frame", so the model reads it as a
perfect anchor and barely animates — that is the real cause of a frozen, barely
moving mouth on shot 1. This node round-trips the guide through an H.264
encode/decode so it looks like a frame the model recognises.

* Mouth still too static -> raise toward 50-70.
* You look soft, smoothed or "stylized" -> lower to 20-25. Compression degrades
  the guide, so a high value costs real facial detail.
* 0 disables it; expect a near-frozen opening.

**`identity_guidance_scale` + `end_percent` (on `LTXVReferenceAudio`) — ship at
3.0 / 0.5.** Guidance amplifies the reference's pull on the WHOLE denoised
tensor — audio *and* video — so run across every step it restyles your face
while it is fixing your voice. Identity is decided in the early, high-noise
steps; fine detail forms in the late ones. `end_percent 0.5` switches guidance
off halfway, keeping the voice lock away from the detail passes.

* Face still stylized -> lower `end_percent` (0.3-0.4), **not** the scale.
* Voice not holding -> raise the scale (4-5), leave `end_percent`.

**If you change both at once you will not know which one did it.** To isolate:
set `img_compression` to 0 on both shots and change nothing else. A sharp face
with a frozen mouth confirms compression is your softness, and you tune up from
there.

## 6. Length and resolution

Both are set **once**, in the GLOBAL group, and feed both shots. They must
match across shots or the chain breaks — shot 2 starts from shot 1's last
frame, and the model cannot bridge a resolution change.

- **`length` must be 8n+1**: 97, 145, 241, 329, 497, 1001…
  Duration = `length ÷ 25`. So 241 ≈ 9.6 s, 497 ≈ 19.9 s, 1001 ≈ 40 s.
- **Width and height must be divisible by 32.** Landscape: 960×544, 1280×704,
  1344×768. Portrait: 544×960, 768×1344.

Three things stay per-shot because core ComfyUI has no math nodes — if you
change `length`, change these too:

1. `LTXV Empty Latent Audio → frames_number` = the same number
2. `ImageFromBatch → batch_index` = `length − 1`
3. each shot's seed (only if you want them to differ)

Cost scales with width × height × length. Get a shot right at the defaults
before scaling up.

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

## 8. The chain

`ImageFromBatch` takes the last frame of shot 1 and hands it to shot 2's
`LTXVAddGuide` at `frame_idx 0`. That is the whole trick.

**To add a shot 3:** copy the SHOT 2 group, add another `ImageFromBatch` fed
from shot 2's `VAEDecode`, and wire it into the new shot's `LTXVAddGuide.image`.
Then extend the FINISH group's `ImageBatch` / `AudioConcat` chain.

Remember: no memory bank. Identity drifts the further you get from the first
frame.

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
  duplicated frame — 1/25th of a second, invisible at a cut.

## 10. Troubleshooting

| symptom | cause |
|---|---|
| A loader dropdown is empty / red | model is in the wrong folder — see §2 |
| Line spoken *plus* garbled words | voice reference too long, or ID-LoRA missing — §5 |
| The model reads your description aloud | exclusivity block removed from the prompt — §7 |
| Mouth barely moves | raise `img_compression` — see §5b |
| Face soft / stylized / "smoothed" | lower `img_compression`, then `end_percent` — §5b |
| Shot 2 looks unrelated to shot 1 | `ImageFromBatch → batch_index` ≠ `length − 1` |
| ComfyUI dies during load, RAM at 100%, VRAM idle | system RAM — raise the Windows pagefile to 64–128 GB |
| OOM in `ImageSharpen` on a long render | expected; leave it bypassed and use ffmpeg — §9 |
| Video looks soft | `scale_by` is doing a plain upscale; enable sharpen on short runs |
