# .joypack — portable video-character cartridge (spec v1.0)

One file = one character: face, voice, canon description, and optionally the
LoRAs and environments that define them. Drop it into a JoyEcho workflow and
render new scenes with that exact character — same face, same voice — with
zero training. Designed to extend the roleplay ecosystem's Character Card
V3/CHARX lineage into photoreal joint audio-video generation.

## Container

A `.joypack` is a ZIP archive (store or deflate) with this layout:

```
manifest.json                REQUIRED  spec + inventory (see below)
persona/card.json            optional  CharacterCardV3-compatible JSON (chat interop)
voice/anchor.(mp4|wav)       REQUIRED  4s+ clip of the character speaking.
                                       Seeds the JoyEcho memory bank before
                                       shot 1 - the voice continues instead of
                                       being re-rolled. mp4 preferred (pairs
                                       audio with a moving reference face).
refs/*.png|jpg               REQUIRED  1-8 MSR identity reference stills,
                                       front-lit face clearly visible.
prompts/dna.txt              REQUIRED  the canonical DNA sentence(s): the
                                       verbatim identity text to carry in every
                                       shot prompt, plus voice scaffold line
                                       (timbre + accent binding).
loras/zimage/*.safetensors   optional  face LoRA for first-frame/still work
loras/ltx/*.safetensors      optional  LTX-2.3 LoRA(s) (ID-LoRA, style)
environment/*.png|jpg        optional  canonical location stills
environment/rooms.txt        optional  verbatim room description(s) - the
                                       location-lock prose, one per paragraph
```

## manifest.json

```json
{
  "joypack": "1.0",
  "name": "MARA",
  "speaker_tag": "mara",
  "display_name": "Detective Mara",
  "authors": ["you"],
  "license": "CC-BY-4.0",
  "notes": "freeform",
  "voice": {"file": "voice/anchor.mp4", "accent_line": "speaking in a casual American accent"},
  "refs": ["refs/mara_01.png", "refs/mara_02.png"],
  "dna": "prompts/dna.txt",
  "loras": {
    "zimage": [{"file": "loras/zimage/mara_z.safetensors", "trigger": "mara_z", "strength": 0.7}],
    "ltx":    [{"file": "loras/ltx/mara_id.safetensors", "strength": 1.0}]
  },
  "environment": {"stills": ["environment/apartment_01.png"], "rooms": "environment/rooms.txt"},
  "render_law": {"video_fps": 24},
  "sha256": {"voice/anchor.mp4": "..."}
}
```

Only `joypack`, `name`, `speaker_tag`, `voice`, `refs`, and `dna` are required.
Unknown keys MUST be ignored (forward compatibility). `render_law` records
generation constraints the character was authored under - loaders should warn
when the live workflow violates them (the fps/accent law above all).

## Loader behavior (reference implementation: JoyEcho_CartridgeLoader)

Loading MATERIALIZES the cartridge into the host's existing conventions -
the render path itself needs no changes:

1. Unpack to a cache dir keyed by content hash.
2. Copy `voice/anchor.*` to `input/joyecho_voices/<speaker_tag>/` (folder
   auto-casting picks it up; every script whose speaker tag matches is voiced
   by the anchor).
3. Copy `refs/*` to the workflow's refs root under `<speaker_tag>/` (RefPicker
   finds them by character scan).
4. Copy LoRAs to `models/loras/joypack/<name>/` and report their names for the
   LoRA stack.
5. Output the DNA text, accent line, room text, and trigger words as node
   outputs for prompt assembly.
6. Verify sha256 entries when present; refuse a cartridge whose manifest is
   missing required keys.

Removal = delete the materialized folders; a cartridge never modifies the
host beyond those drop-in locations.

## Security

A cartridge is DATA. Loaders MUST NOT execute anything from the archive, MUST
reject path traversal (entries containing `..` or absolute paths), and should
treat persona/card.json purely as chat-client payload. LoRA files are weights;
standard safetensors-only policy applies (no pickle formats).

## Versioning

`joypack: "1.0"` this document. Planned 1.1: serialized memory-bank state
(`bank/state.joybank`) for scene-warm characters; multi-character packs.
