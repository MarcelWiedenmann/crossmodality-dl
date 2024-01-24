import glob
import math
import os
from importlib import import_module
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import xmltodict
from skimage.transform import resize

##### Common functionality #####


def import_function(fn_name):
    mod, fn = fn_name.rsplit(".", 1)
    mod = import_module(mod)
    return getattr(mod, fn)


def list_files(
    directory: Path,
    file_extension: Optional[Union[str, List[str]]] = None,
    file_pattern: Optional[str] = None,
    globstar: bool = False,
) -> List[Path]:
    if file_extension is not None and file_pattern is not None:
        raise ValueError(
            "file_extension and file_pattern cannot both be used at the same time."
        )
    if not directory.exists():
        raise ValueError(f"Directory {directory} does not exist.")
    if file_extension is not None:
        if isinstance(file_extension, str):
            file_extension = [file_extension]
        file_pattern = [f"*{fe}" for fe in file_extension]
    elif file_pattern is not None:
        file_pattern = [file_pattern]
    else:
        file_pattern = ["*"]
    files = []
    for fp in file_pattern:
        files.extend(Path(f) for f in glob.glob(str(directory) + os.path.sep + fp, recursive=globstar))
    return sorted(files)


def num_digits(n: int) -> int:
    return int(math.log10(n)) + 1


##### Image manipulation #####


def fuse_sparse_tiles(
    tiles: np.ndarray,
    tile_indices: List[Tuple[int, int]],
    full_image_shape: Tuple[int, int],
    overlap: Optional[Tuple[int, int]] = None,
    pad_y: Optional[Tuple[int, int]] = None,
    pad_x: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    *tile_size_c, tile_size_y, tile_size_x = tiles[0].shape
    overlap_y, overlap_x = overlap if overlap is not None else (0, 0)
    tile_size_y -= overlap_y
    tile_size_x -= overlap_x

    pad_y = pad_y if pad_y is not None else (0, 0)
    pad_x = pad_x if pad_x is not None else (0, 0)
    # Overlap is part of the total padding. Subtract here and remove each tile's overlap below.
    pad_y = max(0, pad_y[0] - overlap_y), max(0, pad_y[1] - overlap_y)
    pad_x = max(0, pad_x[0] - overlap_x), max(0, pad_x[1] - overlap_x)
    padded_image_shape = (
        pad_y[0] + full_image_shape[0] + pad_y[1],
        pad_x[0] + full_image_shape[1] + pad_x[1],
    )

    fused = np.zeros(tuple(tile_size_c) + padded_image_shape, dtype=tiles[0].dtype)
    for i, (tile_y, tile_x) in enumerate(tile_indices):
        fused[
            ...,
            tile_y * tile_size_y : (tile_y + 1) * tile_size_y,
            tile_x * tile_size_x : (tile_x + 1) * tile_size_x,
        ] = remove_padding(tiles[i], (overlap_y, overlap_x))

    return remove_padding(fused, (pad_y, pad_x))


def remove_padding(
    image: np.ndarray,
    padding_shape: Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]],
):
    if isinstance(padding_shape[0], int):
        pad_y_before = padding_shape[0]
        pad_y_after = -padding_shape[0] if padding_shape[0] != 0 else None
        pad_x_before = padding_shape[1]
        pad_x_after = -padding_shape[1] if padding_shape[1] != 0 else None
    else:
        pad_y_before = padding_shape[0][0]
        pad_y_after = -padding_shape[0][1] if padding_shape[0][1] != 0 else None
        pad_x_before = padding_shape[1][0]
        pad_x_after = -padding_shape[1][1] if padding_shape[1][1] != 0 else None
    return image[..., pad_y_before:pad_y_after, pad_x_before:pad_x_after]


##### Metrics #####


def confusion_matrix(
    target: np.ndarray, prediction: np.ndarray, mask: Optional[np.ndarray] = None
) -> Tuple[int, int, int, int]:
    """
    Returns TN, FP, FN, TP in this order.

    Optionally, allows to mask out regions in the prediction that should not be evaluated when computing the matrix.
    Predictions in masked out regions will be counted as true negatives.
    """
    if target.dtype != "bool":
        raise TypeError("Only binary classification is supported at the moment.")
    if prediction.dtype != "bool":
        raise TypeError("Predictions need to be discretized prior to evaluation.")
    if mask is not None and mask.dtype != "bool":
        raise TypeError("Mask must be boolean.")
    # Fast binary confusion matrix implementation. Will not work for multi-class comparisons.
    agreements = np.ravel(target) * 2 + np.ravel(prediction)
    if mask is not None:
        agreements = agreements * np.ravel(mask)
    return np.bincount(agreements, minlength=4)


def aggregate_confusion_matrices(
    matrices: List[Tuple[int, int, int, int]]
) -> Tuple[int, int, int, int]:
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    for tn, fp, fn, tp in matrices:
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp
    return total_tn, total_fp, total_fn, total_tp


def compute_metrics(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    total = float(tn + fp + fn + tp)
    sensitivity = tp / float(fn + tp)
    precision = tp / float(tp + fp) if tp + fp != 0 else float("nan")
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN (%)": tn / total,
        "FP (%)": fp / total,
        "FN (%)": fn / total,
        "TP (%)": tp / total,
        "Accuracy": (tp + tn) / total,
        "Sensitivity": sensitivity,
        "Specificity": tn / float(tn + fp),
        "Precision": precision,
        "F1": f1,
        "Jaccard": f1 / (2 - f1),
    }


##### Omero TIFFs #####


def read_image_size_yx(tiff_path_or_metadata: Union[Path, Dict]) -> Tuple[int, int]:
    metadata = (
        read_omero_metadata(tiff_path_or_metadata)
        if isinstance(tiff_path_or_metadata, Path)
        else tiff_path_or_metadata
    )
    pixels_metadata = metadata["Image"]["Pixels"]
    size_y = int(pixels_metadata["@SizeY"])
    size_x = int(pixels_metadata["@SizeX"])
    return size_y, size_x


def read_omero_metadata(tiff_path: Path) -> Dict[str, Any]:
    with tifffile.TiffFile(tiff_path) as tif:
        return xmltodict.parse(tif.ome_metadata)["OME"]


def read_tiff_channel(
    tiff_path: Path, channel_name_or_index: Union[str, int]
) -> np.ndarray:
    if isinstance(channel_name_or_index, str):
        channels_metadata = read_omero_metadata(tiff_path)["Image"]["Pixels"]["Channel"]
        channel_index = next(
            i
            for i, c in enumerate(channels_metadata)
            if c["@Name"] == channel_name_or_index
        )
    else:
        channel_index = channel_name_or_index
    return tifffile.imread(tiff_path, key=channel_index)


##### Plotting (images, labels) #####


class Plot:
    def __init__(
        self,
        title: Optional[str] = None,
        gamma_correction: bool = True,
        width: int = None,
        ncols: int = None,
        show_titles: bool = True,
    ):
        self._subplots = []
        self._title = title
        self._gamma_correction = gamma_correction
        self._figwidth = width
        self._ncols = ncols
        self._show_titles = show_titles

    def add_image(
        self,
        image_or_path: Union[np.ndarray, Path],
        image_name: str = "Image",
        window: Optional[Tuple] = None,
        vmin: Optional[Number] = None,
        vmax: Optional[Number] = None,
        nan_safe: bool = True,
    ) -> "Plot":
        self._subplots.append(
            (
                "image",
                _get_image(image_or_path, window=window, nan_safe=nan_safe),
                image_name,
                vmin,
                vmax,
            )
        )
        return self

    def add_label_image(
        self,
        label_image_or_path: Union[np.ndarray, Path],
        label_image_name: str = "Label image",
        window: Optional[Tuple] = None,
        vmin: Optional[Number] = None,
        vmax: Optional[Number] = None,
        nan_safe: bool = True,
    ) -> "Plot":
        self._subplots.append(
            (
                "label_image",
                _get_image(label_image_or_path, window=window, nan_safe=nan_safe),
                label_image_name,
                vmin,
                vmax,
            )
        )
        return self

    def add_label_mask(
        self,
        label_mask_or_path: Union[np.ndarray, Path],
        label_mask_name: str = "Label mask",
        window: Optional[Tuple] = None,
        nan_safe: bool = True,
    ) -> "Plot":
        self._subplots.append(
            (
                "label_mask",
                _get_image(label_mask_or_path, window=window, nan_safe=nan_safe),
                label_mask_name,
            )
        )
        return self

    def _build(self):
        subplots = self._subplots

        figwidth = self._figwidth if self._figwidth is not None else 30
        dpi = 100
        image_shape = subplots[0][1].shape
        aspect_ratio = image_shape[1] / image_shape[0]
        figheight = figwidth / aspect_ratio
        figsize = (figwidth, figheight)  # matplotlib axis order
        ncols = self._ncols if self._ncols is not None else len(subplots)
        fig, axs = plt.subplots(
            nrows=math.ceil(len(subplots) / float(ncols)),
            ncols=ncols,
            figsize=figsize,
            dpi=dpi,
        )
        colsize_pxs = (
            int(figheight / ncols * dpi),
            int(figwidth / ncols * dpi),
        )  # numpy axis order
        if len(subplots) == 1:
            axs = np.array([axs])
        axs = np.ravel(axs)
        if len(subplots) < len(axs):
            axs = axs[: len(subplots)]
        if self._title is not None and self._show_titles:
            fig.suptitle(self._title)
        for ax, (plot_type, image, image_name, *custom) in zip(axs, subplots):
            ax.set_axis_off()
            image_shape = image.shape
            image_dtype = image.dtype
            image = resize(image, colsize_pxs, order=0)
            if plot_type == "image":
                title = f"{image_name} (gamma corrected={self._gamma_correction}), shape={image_shape}, dtype={image_dtype}"
                image = image ** (1 / 2.2) if self._gamma_correction else image
                ax.imshow(image, vmin=custom[0], vmax=custom[1])
            elif plot_type == "label_image":
                title = f"{image_name} (gamma corrected={self._gamma_correction}), shape={image_shape}, dtype={image_dtype}"
                image = image ** (1 / 2.2) if self._gamma_correction else image
                ax.imshow(image, cmap="gray", vmin=custom[0], vmax=custom[1])
            elif plot_type == "label_mask":
                title = f"{image_name}, shape={image_shape}, dtype={image_dtype}"
                ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            else:
                raise ValueError("Unrecognized plot type: " + plot_type)
            if self._show_titles:
                ax.set_title(title)

    def show(self):
        self._build()
        plt.show()

    def save(self, save_path):
        self._build()
        plt.savefig(save_path, bbox_inches="tight")


def _get_image(
    image_or_image_path: Union[np.ndarray, Path],
    window: Optional[Tuple] = None,
    nan_safe: bool = True,
):
    if isinstance(image_or_image_path, np.ndarray):
        image = image_or_image_path
    else:
        image = np.load(
            image_or_image_path, mmap_mode="r" if window is not None else None
        )
    if image.ndim == 3:
        # Make sure channels dimension is last.
        if image.shape[0] < image.shape[2]:
            image = np.moveaxis(image, 0, 2)
    if window is not None:
        y, x, h, w = window
        image = image[y : y + h, x : x + w]
    if nan_safe and np.isnan(image).any():
        raise ValueError("Image contains NaNs.")
    return image


##### Tensorboard #####
# Manual extraction and plotting of training metrics. Useful if spinning up a TensorBoard server would otherwise be
# cumbersome.


def list_logged_quantities(
    logs_dir: Path, timestamp: str, format: str = "torch", version: str = "version_0"
):
    train_log_path, val_log_path = _get_log_paths(logs_dir, timestamp, format, version)

    def list_quantity(log_path):
        from tensorflow.python.summary.summary_iterator import (
            summary_iterator,
        )  # Keep local, otherwise we'd always have to load TF when using utils.

        quantities = set()
        for e in summary_iterator(str(log_path)):
            vs = e.summary.value
            if len(vs) != 0:
                quantities.add(vs[0].tag)

        return quantities

    return list_quantity(train_log_path).union(list_quantity(val_log_path))


def _get_log_paths(logs_dir: Path, timestamp: str, format: str, version: str):
    if format == "torch":
        log_dir = logs_dir / f"{timestamp}" / version
        log_path = list_files(log_dir, file_pattern="events.out.tfevents*")[0]
        train_log_path = log_path
        val_log_path = log_path
    else:
        train_log_dir = logs_dir / f"{timestamp}" / "train"
        val_log_dir = logs_dir / f"{timestamp}" / "validation"
        train_log_path = train_log_dir / os.listdir(train_log_dir)[0]
        val_log_path = val_log_dir / os.listdir(val_log_dir)[0]
    return train_log_path, val_log_path


def get_training_metrics(
    logs_dir: Path,
    timestamp: str,
    metrics: List[str] = None,
    format: str = "torch",
    version: str = "version_0",
) -> Tuple[Dict[str, Dict[int, float]]]:
    if metrics is None:
        if format == "torch":
            metrics = ["loss_epoch", "accuracy_epoch", "val_loss", "val_accuracy"]
        else:
            metrics = ["epoch_loss", "epoch_sparse_categorical_accuracy"]

    if format == "torch":
        train_metrics = [m for m in metrics if not m.startswith("val_")]
        val_metrics = [m for m in metrics if m.startswith("val_")]
    else:
        train_metrics = list(metrics)
        val_metrics = list(metrics)

    train_log_path, val_log_path = _get_log_paths(logs_dir, timestamp, format, version)

    train_metrics = _get_metrics(train_log_path, train_metrics)
    val_metrics = _get_metrics(val_log_path, val_metrics)
    return train_metrics, val_metrics


def _get_metrics(log_path: Path, metric_tags: List[str]) -> Dict[str, Dict[int, float]]:
    from tensorflow.python.summary.summary_iterator import (
        summary_iterator,
    )  # Keep local, otherwise we'd always have to load TF when using utils.

    metrics = {}
    for e in summary_iterator(str(log_path)):
        vs = e.summary.value
        if len(vs) != 0:
            v = vs[0]
            metric_tag = v.tag
            for m in metric_tags:
                if metric_tag == m:
                    if m not in metrics:
                        metrics[m] = {}
                    metrics[m][e.step] = v.simple_value
    return metrics


def plot_training_metrics(
    timestamp: str,
    train_metrics: Dict[str, Dict[int, float]],
    val_metrics: Dict[str, Dict[int, float]],
    smoothen: float = 0,
    separate_scales: bool = False,
    log_scale: bool = False,
) -> None:
    first_train_metric = list(train_metrics.values())[0]
    train_epochs = np.arange(1, len(first_train_metric) + 1)
    val_epochs = []
    val_steps = list(val_metrics.values())[0].keys() if len(val_metrics) > 0 else []
    for epoch, step in enumerate(first_train_metric.keys(), 1):
        if step in val_steps:
            val_epochs.append(epoch)
    val_epochs = np.array(val_epochs)

    num_plots = len(train_metrics)
    fig, axs = plt.subplots(ncols=num_plots, figsize=[num_plots * 15, 5])
    if num_plots == 1:
        axs = [axs]
    fig.suptitle("Training run " + timestamp)

    train_color = "tab:red"
    val_color = "tab:blue"
    alpha_unsmoothed = 0.3

    for i, (name, metric) in enumerate(train_metrics.items()):
        ax = axs[i]
        _plot_metric(
            ax,
            name,
            name,
            train_epochs,
            metric.values(),
            "o-",
            train_color,
            train_color,
            smoothen,
            alpha_unsmoothed,
            log_scale,
        )

    for i, (name, metric) in enumerate(val_metrics.items()):
        ax = axs[i]
        if separate_scales:
            ax = ax.twinx()
            axis_name = name
            axis_color = val_color
        else:
            axis_name = list(train_metrics.keys())[i] + " & " + name
            axis_color = "black"
        _plot_metric(
            ax,
            name,
            axis_name,
            val_epochs,
            metric.values(),
            "o-",
            val_color,
            axis_color,
            smoothen,
            alpha_unsmoothed,
            log_scale,
        )
        if not separate_scales:
            if "loss" in name:
                ax.legend(loc="upper right")
            else:
                ax.legend(loc="lower right")

    plt.show()


def _plot_metric(
    ax,
    curve_name,
    axis_name,
    epochs,
    values,
    marker_line,
    curve_color,
    axis_color,
    smoothen,
    alpha_unsmoothed,
    log_scale,
):
    values = list(values)
    ax.set_xlabel("Epoch")
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(axis_name, color=axis_color)
    ax.tick_params(axis="y", labelcolor=axis_color)
    do_smoothen = smoothen != 0
    ax.plot(
        epochs,
        values,
        marker_line,
        alpha=alpha_unsmoothed if do_smoothen else 1,
        color=curve_color,
        label=curve_name,
    )
    if do_smoothen:
        ax.plot(
            epochs,
            _smoothen_curve(values, smoothen),
            marker_line,
            color=curve_color,
            label=f"{curve_name} (smoothened)",
        )


def _smoothen_curve(curve: List[float], weight: float) -> List[float]:
    """
    Smoothing also used by TensorBoard internally.
    """
    smoothed_curve = list()
    last_point = curve[0]
    for point in curve:
        smoothed_point = last_point * weight + (1 - weight) * point
        smoothed_curve.append(smoothed_point)
        last_point = smoothed_point
    return smoothed_curve
