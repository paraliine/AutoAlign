import importlib.machinery
import os
import pathlib
import runpy
import sys
import types


def _dmpo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3] / "DMPO"


def _official_script_path() -> pathlib.Path:
    return _dmpo_root() / "fastchat" / "train" / "train_dmpo_efficient.py"


def _prepend_official_paths() -> None:
    dmpo_root = _dmpo_root()
    train_dir = dmpo_root / "fastchat" / "train"

    for path in (dmpo_root, train_dir):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)


def _install_cpu_smoke_deepspeed_shim() -> None:
    if os.environ.get("AUTOALIGN_CPU_SMOKE") != "1":
        return

    import torch

    if torch.cuda.is_available() or "deepspeed" in sys.modules:
        return

    deepspeed = types.ModuleType("deepspeed")
    deepspeed.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)

    class DeepSpeedEngine:  # pragma: no cover
        pass

    deepspeed.DeepSpeedEngine = DeepSpeedEngine
    sys.modules["deepspeed"] = deepspeed


def _install_trl_compat_shims() -> None:
    import trl.import_utils as trl_import_utils
    import trl.trainer.utils as trl_trainer_utils
    from transformers.integrations.integration_utils import is_wandb_available
    from transformers.utils import is_peft_available

    if not hasattr(trl_import_utils, "is_peft_available"):
        trl_import_utils.is_peft_available = is_peft_available
    if not hasattr(trl_import_utils, "is_wandb_available"):
        trl_import_utils.is_wandb_available = is_wandb_available
    if not hasattr(trl_trainer_utils, "trl_sanitze_kwargs_for_tagging"):
        trl_trainer_utils.trl_sanitze_kwargs_for_tagging = (
            lambda *, tag_names, kwargs: kwargs
        )


def main() -> None:
    official_script = _official_script_path()
    if not official_script.is_file():
        raise FileNotFoundError(f"DMPO official script not found: {official_script}")

    _prepend_official_paths()
    _install_cpu_smoke_deepspeed_shim()
    _install_trl_compat_shims()
    runpy.run_path(str(official_script), run_name="__main__")


if __name__ == "__main__":
    main()
