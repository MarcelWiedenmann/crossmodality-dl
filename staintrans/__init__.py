from importlib import import_module


def get_model_type(model_type_name: str):
    mod = import_module(f"staintrans.{model_type_name}")
    return mod.CycleGan


def get_pred_subdir(opts):
    if opts.staintrans_name is not None:
        subdir = (
            f"{opts.staintrans_checkpoint_epoch:03}"
            if opts.staintrans_checkpoint_epoch is not None
            else "best"
        )
    else:
        subdir = ""
    return subdir
