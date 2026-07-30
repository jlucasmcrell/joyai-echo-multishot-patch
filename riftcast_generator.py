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
        },
        "optional": {
            "template": ("STRING", {"forceInput": True, "tooltip":
                         "Wire a RiftCast Audition Script node here to "
                         "customize the casting call. Unwired = the "
                         "built-in default."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("audition_script", "dna_text", "character_name")
    FUNCTION = "design"
    CATEGORY = "JoyAI-Echo/RiftCast"

    def design(self, name, gender, age, ethnicity, skin, hair_color, hair_style,
               height, build, voice, accent, wardrobe, distinguishing,
               template=None):
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

        # AUDITION ISOLATION (2026-07-30). An audition must be a FRESH ROLL.
        # Once a character is packed, its assets are installed
        # (joyecho_refs/<NAME>/ + joyecho_voices/<tag>/) and would otherwise be
        # re-cast on the next audition: RefPicker matches the name in prose and
        # injects the old face, and folder auto-cast seeds the old voice - so
        # the dropdowns would appear to do nothing. Both matchers are avoided
        # by construction:
        #   - prose refers to the subject as ID_A (RefPicker matches nothing;
        #     it strips quoted dialogue before scanning, so the name is safe
        #     to keep in the spoken line)
        #   - the speaker tag is a reserved audition tag that cannot match a
        #     voice folder
        # The REAL name still rides on dna_text/character_name for the Packer,
        # so the cartridge is written correctly.
        dna_audition = dna.replace(name, "ID_A", 1)
        tmpl = template or AUDITION_SCENE_DEFAULT
        prompt = (tmpl
                  .replace("{dialogue}", AUDITION_DIALOGUE_DEFAULT)
                  .replace("{dna}", dna_audition)
                  .replace("{name_title}", name.title())
                  .replace("{name}", "ID_A")
                  .replace("{accent}", accent)
                  .replace("{pro}", pro)
                  .replace("{pos_cap}", pos.capitalize())
                  .replace("{pos}", pos))
        script = {"speakers": [AUDITION_SPEAKER_TAG], "prompts": [prompt]}
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
            existed = os.path.isfile(out_path)
            pack(td, out_path)
            if existed:
                print(f"[RiftCast] NOTE: {name}.riftcast already existed and was "
                      f"REPLACED by this audition. (Auditions are isolated from "
                      f"installed cartridges, so this roll was a fresh one - but "
                      f"the previous {name} is now gone. Use a different name to "
                      f"keep both.)", flush=True)

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


class RiftCast_SourceSwitch:
    """Route ONE of two prompt sources into the render chain.

    'prompt file' passes the PromptSource/LPFF script through untouched -
    your normal batch rendering. 'character designer' passes the Designer's
    audition script. Both inputs stay wired; only the selected one flows.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (["prompt file", "character designer"], {
                    "default": "prompt file",
                    "tooltip": "Which script drives this queue: your LPFF/"
                               "JSON prompt file, or the Character Designer's "
                               "audition tape."}),
            },
            "optional": {
                "file_script": ("STRING", {"forceInput": True}),
                "designer_script": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("story_idea",)
    FUNCTION = "route"
    CATEGORY = "JoyAI-Echo/RiftCast"

    def route(self, source, file_script=None, designer_script=None):
        if source == "character designer":
            if not designer_script:
                raise ValueError("source is 'character designer' but no "
                                 "Character Designer is wired in")
            return (designer_script,)
        if not file_script:
            raise ValueError("source is 'prompt file' but no PromptSource "
                             "is wired in")
        return (file_script,)


NODE_CLASS_MAPPINGS["RiftCast_SourceSwitch"] = RiftCast_SourceSwitch
NODE_DISPLAY_NAME_MAPPINGS["RiftCast_SourceSwitch"] = "RiftCast Source Switch (file / designer)"


class JoyEcho_RenderClock:
    """One source of truth for time: fps + duration in, every fps/frames
    socket in the graph fed from here. Outputs are typed for their targets
    (INT for Generate/Packer, FLOAT for CreateVideo/LLMEnhance) because a
    single primitive type-locks and cannot feed both. num_frames snaps to
    the nearest valid 8n+1."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "fps": ("INT", {"default": 24, "min": 1, "max": 60, "tooltip":
                    "KEEP AT 24 for dialogue - the joint AV prior is "
                    "24fps-native; other values drift accents (25=British, "
                    "30=Australian) and override accent wording."}),
            "duration_seconds": ("FLOAT", {"default": 10.0, "min": 0.5,
                                           "max": 60.0, "step": 0.5, "tooltip":
                    "PER SHOT, not total. Every shot in the script renders "
                    "this long: a 5-shot script at 10s makes a ~50s master "
                    "(minus head-trims/transitions). Single-shot scripts and "
                    "auditions: this IS the full duration."}),
        }}

    RETURN_TYPES = ("INT", "FLOAT", "INT")
    RETURN_NAMES = ("video_fps", "fps_float", "num_frames")
    FUNCTION = "clock"
    CATEGORY = "JoyAI-Echo"

    def clock(self, fps, duration_seconds):
        raw = fps * duration_seconds
        n = max(1, round((raw - 1) / 8.0))
        frames = int(8 * n + 1)
        frames = min(frames, 1441)
        print(f"[JoyEcho] RenderClock: {fps} fps x {duration_seconds:.1f}s PER SHOT "
              f"-> {frames} frames/shot (8n+1 snapped).", flush=True)
        return (int(fps), float(fps), frames)


NODE_CLASS_MAPPINGS["JoyEcho_RenderClock"] = JoyEcho_RenderClock
NODE_DISPLAY_NAME_MAPPINGS["JoyEcho_RenderClock"] = "JoyEcho Render Clock (fps + duration -> frames)"

AUDITION_SPEAKER_TAG = "id_a"   # reserved: never a voice folder

AUDITION_SCENE_DEFAULT = (
    "Consumer mirrorless camera video, neutral color, modest dynamic "
    "range: a locked static medium close-up at eye level in a small "
    "casting room, head and shoulders filling most of the frame and "
    "the mouth fully visible, one continuous unbroken take that never "
    "cuts and never changes angle. The backdrop is matte gray seamless "
    "paper with a soft vertical curl at its edge, taped at the top "
    "corners, and the floor edge shows scuffed pale concrete. The "
    "light is one large soft key from the front, even and flattering. "
    "{dna} "
    "Looking into the lens, calm and personable, {name} is talking, "
    "saying in a {accent} accent, \"{dialogue}\" "
    "{name} shifts weight once, tucks a strand of "
    "hair back, and gives one small easy smile as the take ends. "
    "{pos_cap} lips move naturally in tight sync with every "
    "word. {name} is the only person in frame and "
    "{pos} voice is the only voice on the audio track. The only "
    "sounds are the room's quiet air, a faint camera-handling sound "
    "at the start, fabric shifting, and {pos} voice close and clean "
    "on the mic.")

AUDITION_DIALOGUE_DEFAULT = (
    "Hi, my name's {name_title}. This is my audition tape, so, here's a "
    "little about how I sound when I'm just talking. I'll read whatever "
    "you've got, whenever you're ready.")


class RiftCast_AuditionScript:
    """Editable casting-call template - the scene and the spoken lines the
    Character Designer uses for audition tapes. Separate from your
    production prompt files; wire audition_template into the Designer's
    template input. Placeholders (safe string replacement, stray braces
    are harmless): {dna} {name} {name_title} {accent} {pro} {pos}
    {pos_cap} {dialogue}.

    Keep these rules if you rewrite it: mouth-visible close-up, ONE
    unbroken take (the Packer cuts anchor + refs from this footage),
    keep the 'saying in a {accent} accent' binding, dialogue stays
    non-falsifiable, and END ON MOTION (a settled character dead-stares)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "scene_template": ("STRING", {"multiline": True,
                                          "default": AUDITION_SCENE_DEFAULT}),
            "dialogue": ("STRING", {"multiline": True,
                                    "default": AUDITION_DIALOGUE_DEFAULT}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audition_template",)
    FUNCTION = "template"
    CATEGORY = "JoyAI-Echo/RiftCast"

    def template(self, scene_template, dialogue):
        return (scene_template.replace("{dialogue}", dialogue),)


NODE_CLASS_MAPPINGS["RiftCast_AuditionScript"] = RiftCast_AuditionScript
NODE_DISPLAY_NAME_MAPPINGS["RiftCast_AuditionScript"] = "RiftCast Audition Script (casting call)"
