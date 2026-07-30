"""RiftCast Generator - design a character from dropdowns, render their
audition tape, and pack them into a .riftcast cartridge in one queue.

Two nodes:
  RiftCast_CharacterDesigner - dropdowns -> canonical DNA paragraph + an
      audition-shot script (neutral casting room, front-lit close-up,
      ~45 words of non-falsifiable dialogue with timbre + accent binding -
      the verified voice recipe). Feed its script output into LLMEnhance
      (passthrough) exactly like a PromptSource.
  RiftCast_Packer - takes the rendered frames + audio from Generate, cuts
      the voice anchor and reference stills, writes NAME.riftcast into
      input/riftcast/ and materializes it immediately. The character is
      castable in the very next render.

Re-queue with a new seed to re-audition; pack when you like them. Once
packed, the voice and face are locked for every future render.
"""
import json
import os
import tempfile

try:
    import folder_paths
except Exception:
    folder_paths = None

from .riftcast import pack
from .joyecho_cartridge import _packs_dir

GENDER = {
    "woman": ("woman", "she", "her", "hers"),
    "man": ("man", "he", "his", "his"),
    "androgynous person": ("person with an androgynous presentation", "they", "their", "theirs"),
}
AGE = ["teens", "early twenties", "mid twenties", "thirties", "forties",
       "fifties", "sixties", "seventies"]
SKIN = ["pale", "fair freckled", "light olive", "tan", "golden brown",
        "brown", "deep brown", "dark"]
ETHNICITY = ["(unspecified)", "of East Asian descent", "of South Asian descent",
             "of Southeast Asian descent", "of Middle Eastern descent",
             "of Mexican descent", "of Puerto Rican descent",
             "of West African descent", "of Mediterranean descent",
             "of Eastern European descent", "of Irish descent",
             "of Indigenous American descent"]
HAIR_COLOR = ["black", "dark brown", "chestnut brown", "auburn", "dirty blonde",
              "blonde", "red", "gray-streaked", "silver", "white"]
HAIR_STYLE = ["long and loose", "long and wavy", "shoulder-length and straight",
              "shoulder-length and curly", "in a tight bun", "in braids",
              "short and cropped", "buzzed close", "short and tousled",
              "under a wool cap"]
HEIGHT = ["petite", "average height", "tall"]
BUILD = ["slim", "average build", "athletic", "sturdy", "heavyset"]
VOICE = ["a clear voice, natural and unforced",
         "a low voice with a slight rasp",
         "a warm mid-toned voice",
         "a bright quick voice",
         "a soft-spoken voice",
         "a gravelly voice",
         "a deep steady voice"]
ACCENT = ["casual American", "casual American with flat Midwestern vowels",
          "soft Southern American", "Boston-flavored American",
          "British", "Australian"]


def _an(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def _realism_block(age, pronoun_pos):
    if age == "teens":
        return "clear matte skin and natural unstyled brows"
    if age in ("sixties", "seventies"):
        return ("weathered skin with deep creases at the eyes, visible pores, "
                "dry lips, no makeup")
    return ("matte skin with visible pores across the nose and cheeks, faint "
            "shadows under the eyes, dry lips, no makeup")


class RiftCast_CharacterDesigner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "name": ("STRING", {"default": "NOVA"}),
            "gender": (list(GENDER.keys()),),
            "age": (AGE, {"default": "mid twenties"}),
            "ethnicity": (ETHNICITY,),
            "skin": (SKIN, {"default": "pale"}),
            "hair_color": (HAIR_COLOR, {"default": "dark brown"}),
            "hair_style": (HAIR_STYLE, {"default": "long and loose"}),
            "height": (HEIGHT, {"default": "average height"}),
            "build": (BUILD, {"default": "average build"}),
            "voice": (VOICE, {"default": "a clear voice, natural and unforced"}),
            "accent": (ACCENT, {"default": "casual American", "tooltip":
                       "American variants are enforced by the accent LoRA at "
                       "24fps. British/Australian render natively at 25/30fps "
                       "(the fps-accent law) - or at 24fps they lean on the "
                       "wording alone."}),
            "wardrobe": ("STRING", {"default": "a plain dark crewneck shirt"}),
            "distinguishing": ("STRING", {"default": "", "tooltip":
                               "optional: thin-framed glasses, a small nose "
                               "stud, freckles across the nose..."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("audition_script", "dna_text", "character_name")
    FUNCTION = "design"
    CATEGORY = "JoyAI-Echo/RiftCast"

    def design(self, name, gender, age, ethnicity, skin, hair_color, hair_style,
               height, build, voice, accent, wardrobe, distinguishing):
        name = (name or "NOVA").strip().upper()
        noun, pro, pos, _ = GENDER[gender]
        nationality = ("British" if accent == "British" else
                       "Australian" if accent == "Australian" else "American")
        eth = "" if ethnicity == "(unspecified)" else f" {ethnicity}"
        extra = f", {distinguishing.strip()}" if distinguishing.strip() else ""
        teen_word = "teenaged " if age == "teens" else ""
        bare_build = build.replace(" build", "")
        dna = (f"{name} is {_an(teen_word or nationality)} {teen_word}{nationality} {noun}{eth} in "
               f"{pos} {age}, {height} with {_an(bare_build)} {bare_build} build, "
               f"with {skin} {_realism_block(age, pos)}, "
               f"{pos} {hair_color} hair {hair_style}{extra}, wearing {wardrobe}. "
               f"{pos.capitalize()} voice is {voice}, speaking in a "
               f"{accent} accent.")

        script = {
            "speakers": [name.lower()],
            "prompts": [(
                "Consumer mirrorless camera video, neutral color, modest dynamic "
                "range: a locked static medium close-up at eye level in a small "
                "casting room, head and shoulders filling most of the frame and "
                "the mouth fully visible, one continuous unbroken take that never "
                "cuts and never changes angle. The backdrop is matte gray seamless "
                "paper with a soft vertical curl at its edge, taped at the top "
                "corners, and the floor edge shows scuffed pale concrete. The "
                "light is one large soft key from the front, even and flattering. "
                f"{dna} "
                f"Looking into the lens, calm and personable, {name} is talking, "
                f"saying in a {accent} accent, \"Hi, my name's "
                f"{name.title()}. This is my audition tape, so, here's a little "
                f"about how I sound when I'm just talking. I'll read whatever "
                f"you've got, whenever you're ready.\" "
                f"{name} shifts weight once, tucks a strand of "
                f"hair back, and gives one small easy smile as the take ends. "
                f"{pos.capitalize()} lips move naturally in tight sync with every "
                f"word. {name} is the only person in frame and "
                f"{pos} voice is the only voice on the audio track. The only "
                f"sounds are the room's quiet air, a faint camera-handling sound "
                f"at the start, fabric shifting, and {pos} voice close and clean "
                f"on the mic."
            )],
        }
        return (json.dumps(script, ensure_ascii=True), dna, name)


class RiftCast_Packer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "audio": ("AUDIO",),
            "character_name": ("STRING", {"forceInput": True}),
            "dna_text": ("STRING", {"forceInput": True}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            "anchor_start_sec": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0}),
            "anchor_dur_sec": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 12.0}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "pack_it"
    CATEGORY = "JoyAI-Echo/RiftCast"
    OUTPUT_NODE = True

    def pack_it(self, images, audio, character_name, dna_text, fps=24,
                anchor_start_sec=1.0, anchor_dur_sec=5.0):
        import av
        import numpy as np
        import torch
        from PIL import Image

        name = (character_name or "NOVA").strip().upper()
        frames = images
        if isinstance(frames, torch.Tensor):
            frames = (frames.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        wav = audio["waveform"]
        sr = int(audio["sample_rate"])
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu()
        if wav.dim() == 3:
            wav = wav[0]
        F = frames.shape[0]

        with tempfile.TemporaryDirectory() as td:
            for sub in ("voice", "refs", "prompts"):
                os.makedirs(os.path.join(td, sub))

            # --- voice anchor: av mux of the requested slice
            f0 = max(0, int(round(anchor_start_sec * fps)))
            f1 = min(F, int(round((anchor_start_sec + anchor_dur_sec) * fps)))
            if f1 - f0 < fps * 2:
                f0, f1 = 0, min(F, int(fps * 5))
            s0 = int(round(f0 / fps * sr))
            s1 = int(round(f1 / fps * sr))
            apath = os.path.join(td, "voice", "anchor.mp4")
            out = av.open(apath, "w")
            vs = out.add_stream("h264", rate=fps)
            vs.width = int(frames.shape[2]); vs.height = int(frames.shape[1])
            vs.pix_fmt = "yuv420p"
            vs.options = {"crf": "18"}
            asr = out.add_stream("aac", rate=sr)
            for i in range(f0, f1):
                fr = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
                for pkt in vs.encode(fr):
                    out.mux(pkt)
            for pkt in vs.encode():
                out.mux(pkt)
            seg = wav[..., s0:s1]
            seg16 = (seg.clamp(-1, 1) * 32767).to(torch.int16).numpy()
            if seg16.ndim == 1:
                seg16 = seg16[None, :]
            af = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(seg16), format="s16p",
                layout="stereo" if seg16.shape[0] == 2 else "mono")
            af.sample_rate = sr
            for pkt in asr.encode(af):
                out.mux(pkt)
            for pkt in asr.encode():
                out.mux(pkt)
            out.close()

            # --- refs: three spread frames from inside the anchor window
            for i, t in enumerate((f0 + fps, (f0 + f1) // 2, max(f0, f1 - fps)), 1):
                Image.fromarray(frames[min(F - 1, int(t))]).save(
                    os.path.join(td, "refs", f"{name.lower()}_{i:02d}.png"))

            open(os.path.join(td, "prompts", "dna.txt"), "w",
                 encoding="ascii", errors="replace").write(dna_text.strip() + "\n")
            manifest = {"riftcast": "1.0", "name": name,
                        "speaker_tag": name.lower(),
                        "display_name": name.title(),
                        "voice": {"file": "voice/anchor.mp4"},
                        "render_law": {"video_fps": 24}}
            json.dump(manifest, open(os.path.join(td, "manifest.json"), "w"),
                      indent=1)

            out_path = os.path.join(_packs_dir(), f"{name}.riftcast")
            pack(td, out_path)

        # materialize immediately - castable without a restart
        from .joyecho_cartridge import auto_materialize_all
        auto_materialize_all()
        report = (f"packed {name}.riftcast ({f1-f0} anchor frames, 3 refs) and "
                  f"installed - speaker tag '{name.lower()}' casts in the next "
                  f"render. Re-queue the audition with a new seed to recast; "
                  f"delete the .riftcast to retire.")
        print(f"[RiftCast] {report}", flush=True)
        return (report,)


NODE_CLASS_MAPPINGS = {
    "RiftCast_CharacterDesigner": RiftCast_CharacterDesigner,
    "RiftCast_Packer": RiftCast_Packer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RiftCast_CharacterDesigner": "RiftCast Character Designer",
    "RiftCast_Packer": "RiftCast Packer (audition -> cartridge)",
}
