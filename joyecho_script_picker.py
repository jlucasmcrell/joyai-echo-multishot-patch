"""JoyEcho Script Picker — a dropdown node for selecting a saved prompt JSON.

The stock JoyEcho_TextEncode 'prompts' field only accepts pasted JSON / a typed
path, which is painful in the canvas. This node lists every *.json in
  <ComfyUI>/input/joyecho_prompts/
as a COMBO dropdown and outputs the file's contents, wired straight into
JoyEcho_TextEncode's 'prompts' input (which accepts inline {"prompts":[...]} JSON).

Add a .json to that folder, hit the ComfyUI refresh button (or press R) to
repopulate the dropdown, pick it, run. Editing the file re-triggers execution
automatically (IS_CHANGED tracks mtime) — no need to reselect.
"""

import json
from pathlib import Path

import folder_paths

_PROMPTS_SUBDIR = "joyecho_prompts"
_EMPTY = "(no .json in input/joyecho_prompts)"


def _scripts_dir() -> Path:
    d = Path(folder_paths.get_input_directory()) / _PROMPTS_SUBDIR
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def _list_scripts() -> list[str]:
    d = _scripts_dir()
    try:
        files = sorted(p.name for p in d.glob("*.json"))
    except OSError:
        files = []
    return files if files else [_EMPTY]


class JoyEcho_ScriptPicker:
    """Pick a prompt-script .json from input/joyecho_prompts via a dropdown."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"script": (_list_scripts(),)}}

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("prompts_json", "path",)
    FUNCTION = "load"
    CATEGORY = "JoyAI-Echo"

    @classmethod
    def IS_CHANGED(cls, script):
        # Re-run when the selected file changes on disk, so edits are picked up.
        p = _scripts_dir() / script
        try:
            return f"{script}:{p.stat().st_mtime}"
        except OSError:
            return script

    def load(self, script):
        if script == _EMPTY:
            raise ValueError(
                f"No .json scripts found. Put your prompt JSON in {_scripts_dir()} "
                "and press the refresh button (or R) to repopulate the dropdown."
            )
        p = _scripts_dir() / script
        if not p.exists():
            raise FileNotFoundError(
                f"Script not found: {p}. Refresh the node list (R) after adding files."
            )
        text = p.read_text(encoding="utf-8")
        # Fail early with a clear message rather than deep in the text encoder.
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{script} is not valid JSON: {e}")
        arr = data.get("prompts") if isinstance(data, dict) else None
        if arr is None and isinstance(data, dict):
            arr = data.get("shots")
        if not isinstance(arr, list) or not arr:
            raise ValueError(f"{script} must contain a non-empty 'prompts' (or 'shots') array.")
        print(f"[JoyEcho] ScriptPicker: {script} ({len(arr)} shots).", flush=True)
        return (text, str(p),)


NODE_CLASS_MAPPINGS = {"JoyEcho_ScriptPicker": JoyEcho_ScriptPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyEcho_ScriptPicker": "JoyEcho Script Picker (JSON dropdown)"}
