def get_experiment_name(opts, cv_fold: int):
    return (
        "multiscale_"
        + _get_staintrans_suffix(opts)
        + _get_stainaugm_suffix(opts)
        + _get_stainnorm_suffix(opts)
        + _get_cv_fold_suffix(cv_fold)
    )


def get_segm_model_experiment_name(opts, cv_fold):
    return "multiscale_" + _get_stainaugm_suffix(opts) + _get_cv_fold_suffix(cv_fold)


def _get_staintrans_suffix(opts):
    return (
        f"staintrans_{opts.staintrans_name}_"
        if opts.staintrans_name is not None
        else ""
    )


def _get_stainaugm_suffix(opts):
    return (
        f"augm_{opts.stainaugm_color_space}_{opts.stainaugm_factor:.5f}_"
        if opts.stainaugm_color_space is not None
        else ""
    )


def _get_stainnorm_suffix(opts):
    return (
        f"norm_{opts.stainnorm_color_space}_{opts.stainnorm_factor:.5f}_"
        if opts.stainnorm_color_space is not None
        else ""
    )


def _get_cv_fold_suffix(cv_fold):
    return f"cv_fold_{cv_fold}"
