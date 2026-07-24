"""JoyEcho Prompt Source - ONE dropdown for both prompt pipelines.

Lists LPFF-style brief/JSON-block .txt files (from the inspire-pack prompts
tree) AND passthrough .json scripts (from <ComfyUI>/input/joyecho_prompts/) in
a single combo, and always emits the same outputs:

  story_idea (STRING, list) -> JoyEcho_LLMEnhance.story_idea
  character  (STRING, list) -> JoyEcho_RefPicker.character
  count      (INT)

.txt files are parsed LPFF-style (blocks split on ---, positive:/negative:/
name: fields); each block becomes one queue item, so multi-block files fan out
exactly like LoadPromptsFromFile. .json files load as a single passthrough item
(the raw JSON text). Blocks without a name: line emit "" (RefPicker falls
through to prompt-scan / fallback), never the filename.

Replaces the LPFF -> UnzipPrompt chain and the Script Picker: wire once,
switch sources by picking a different file.
"""

import json
import re
from pathlib import Path

import folder_paths

_JSON_SUBDIR = "joyecho_prompts"
_TXT_PREFIX = "TXT: "
_JSON_PREFIX = "JSON: "
_EMPTY = "(no prompt files found)"

_BLOCK_SPLIT = re.compile(r"\n\s*-+\s*\n")
_BLOCK_PATTERN = re.compile(
    r"^(?:(?:name:(?P<name>.*?)|positive:(?P<positive>.*?)|negative:(?P<negative>.*?))\n*)+$",
    re.DOTALL,
)


def _txt_root() -> Path | None:
    try:
        roots = folder_paths.get_folder_paths("inspire_prompts")
        return Path(roots[0]) if roots else None
    except Exception:
        return None


def _json_root() -> Path:
    d = Path(folder_paths.get_input_directory()) / _JSON_SUBDIR
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def _list_files() -> list[str]:
    out = []
    root = _txt_root()
    if root and root.is_dir():
        for p in sorted(root.rglob("*.txt")):
            out.append(_TXT_PREFIX + str(p.relative_to(root)))
    jroot = _json_root()
    for p in sorted(jroot.glob("*.json")):
        out.append(_JSON_PREFIX + p.name)
    return out or [_EMPTY]


def _resolve(choice: str) -> Path | None:
    if choice.startswith(_TXT_PREFIX):
        root = _txt_root()
        return (root / choice[len(_TXT_PREFIX):]) if root else None
    if choice.startswith(_JSON_PREFIX):
        return _json_root() / choice[len(_JSON_PREFIX):]
    return None


class JoyEcho_PromptSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_file": (_list_files(),),
            },
            "optional": {
                "load_cap": ("INT", {"default": 0, "min": 0,
                                     "tooltip": "TXT files only: 0 = all blocks, N = first N from start_index."}),
                "start_index": ("INT", {"default": 0, "min": 0,
                                        "tooltip": "TXT files only: first block index to load."}),
                "character_override": ("STRING", {"default": "",
                                                  "tooltip": "Force this character for EVERY emitted item "
                                                             "(useful for .json scripts, which carry no name field)."}),
                "manual_path": ("STRING", {"default": "",
                                           "tooltip": "Point at ANY .txt (LPFF blocks) or .json (passthrough script) "
                                                      "file by full path, ignoring the dropdown. Removes any dependency "
                                                      "on the inspire-pack prompts tree or input/joyecho_prompts/. "
                                                      "Leave empty to use source_file."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("story_idea", "character", "count")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "load"
    CATEGORY = "JoyAI-Echo"

    @classmethod
    def IS_CHANGED(cls, source_file, load_cap=0, start_index=0, character_override="", manual_path=""):
        p = Path(manual_path.strip()) if manual_path.strip() else _resolve(source_file)
        key = manual_path.strip() or source_file
        try:
            return f"{key}:{p.stat().st_mtime}:{load_cap}:{start_index}:{character_override}"
        except Exception:
            return key

    def load(self, source_file, load_cap=0, start_index=0, character_override="", manual_path=""):
        manual = manual_path.strip()
        if manual:
            p = Path(manual).expanduser()
            is_json = p.suffix.lower() == ".json"
        else:
            if source_file == _EMPTY:
                raise ValueError("PromptSource: no prompt files found in the inspire prompts tree "
                                 "or input/joyecho_prompts/. Set manual_path to a file instead.")
            p = _resolve(source_file)
            is_json = source_file.startswith(_JSON_PREFIX)
        if p is None or not p.exists():
            raise FileNotFoundError(f"PromptSource: {manual or source_file} -> {p} not found. "
                                    "Refresh the node list (R) after adding files, or check manual_path.")
        text = p.read_text(encoding="utf-8")
        override = character_override.strip().lower()

        if is_json:
            data = json.loads(text)  # fail early with a clear error
            arr = data.get("prompts") or data.get("shots")
            if not isinstance(arr, list) or not arr:
                raise ValueError(f"{p.name} must contain a non-empty 'prompts' (or 'shots') array.")
            print(f"[JoyEcho] PromptSource: {p.name} (json, {len(arr)} shots, 1 item).", flush=True)
            return ([text], [override], 1)

        # TXT: LPFF-style blocks
        items, names = [], []
        for blk in _BLOCK_SPLIT.split(text):
            m = _BLOCK_PATTERN.search(blk)
            if not m or m.group("positive") is None:
                continue
            items.append(m.group("positive").strip())
            nm = (m.group("name") or "").strip().lower()
            names.append(override or nm)
        total = len(items)
        items = items[start_index:]
        names = names[start_index:]
        if load_cap > 0:
            items = items[:load_cap]
            names = names[:load_cap]
        if not items:
            raise ValueError(f"PromptSource: {p.name} yielded no blocks "
                             f"(total {total}, start_index {start_index}, load_cap {load_cap}).")
        print(f"[JoyEcho] PromptSource: {p.name} (txt, {len(items)}/{total} blocks).", flush=True)
        return (items, names, len(items))


NODE_CLASS_MAPPINGS = {"JoyEcho_PromptSource": JoyEcho_PromptSource}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyEcho_PromptSource": "JoyEcho Prompt Source (txt briefs + json scripts)"}
