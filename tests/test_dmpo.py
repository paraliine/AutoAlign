from autoalign.train import dmpo


def test_official_dmpo_script_exists():
    assert dmpo._official_script_path().is_file()


def test_prepend_official_paths():
    original_sys_path = dmpo.sys.path[:]
    try:
        dmpo.sys.path = ["placeholder"]
        dmpo._prepend_official_paths()

        assert dmpo.sys.path[0] == str(dmpo._dmpo_root() / "fastchat" / "train")
        assert dmpo.sys.path[1] == str(dmpo._dmpo_root())
    finally:
        dmpo.sys.path = original_sys_path


def test_install_trl_compat_shims():
    dmpo._install_trl_compat_shims()

    import trl.import_utils as trl_import_utils
    import trl.trainer.utils as trl_trainer_utils

    assert hasattr(trl_import_utils, "is_peft_available")
    assert hasattr(trl_import_utils, "is_wandb_available")
    assert hasattr(trl_trainer_utils, "trl_sanitze_kwargs_for_tagging")
