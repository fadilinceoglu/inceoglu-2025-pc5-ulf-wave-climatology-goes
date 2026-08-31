"""Atomic, non-interactive generators for Figures 1--4 of the paper."""

from __future__ import annotations

# Matplotlib's backend must be selected before importing pyplot.
# ruff: noqa: E402
import os
import shutil
import tempfile
from pathlib import Path
from typing import Final, Optional

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .config import (
    COMPONENTS,
    FIGURE_1_FREQUENCY_HISTOGRAM_BINS,
    FIGURE_1_MLT_HISTOGRAM_BINS,
    FIGURE_DPI,
    FREQUENCY_BIN_COUNT,
    PC5_FREQUENCY_MAX_HZ,
    PC5_FREQUENCY_MIN_HZ,
)
from .io import atomic_destination
from .statistics import (
    annual_occurrence_rates,
    annual_omni_parameters,
    atomic_write_csv,
    condition_binned_statistics,
    expand_events_with_daily_omni,
    fit_power_law,
    load_event_catalogs,
    load_or_build_observation_counts,
    mlt_sector_binned_medians,
    pearson_r,
    power_law,
    prepare_condition_omni,
    read_omni_hourly,
    solar_cycle_correlations,
    split_solar_wind_conditions,
    validate_omni_study_grid,
)

_COMPONENT_ORDER: Final = COMPONENTS
_COMPONENT_LABELS: Final = {
    "radial": r"B$_r$",
    "azimuthal": r"B$_{\phi}$",
    "parallel": r"B$_{||}$",
}
_OMEGA_LABELS: Final = {
    "radial": "r",
    "azimuthal": r"\phi",
    "parallel": "||",
}


def _read_validated_omni(omni_path: Path) -> pd.DataFrame:
    """Read OMNI once and enforce the complete study-hour contract."""

    path = Path(omni_path)
    omni = read_omni_hourly(path)
    validate_omni_study_grid(omni, source=path)
    return omni


class _StretchedNorm(Normalize):
    """Apply the Figure-1 color stretch above 90% of the range."""

    def __init__(
        self,
        vmin: float,
        vmax: float,
        *,
        stretch_point: float,
        stretch_factor: float = 3,
    ) -> None:
        super().__init__(vmin=vmin, vmax=vmax)
        self.stretch_point = stretch_point
        self.stretch_factor = stretch_factor

    def __call__(self, value: object, clip: Optional[bool] = None) -> np.ndarray:
        values = np.asanyarray(value)
        normalized = np.asanyarray(super().__call__(values, clip)).copy()
        mask = values > self.stretch_point
        normalized[mask] = (
            self.stretch_point
            + (values[mask] - self.stretch_point) * self.stretch_factor
            - float(self.vmin)
        ) / (float(self.vmax) - float(self.vmin))
        return np.clip(normalized, 0, 1)


def _atomic_save_figure(figure: Figure, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_format = output_path.suffix.lstrip(".").lower() or "png"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=f".{image_format}",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(
            temporary_path,
            format=image_format,
            bbox_inches="tight",
            dpi=FIGURE_DPI,
        )
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def plot_figure_1(checkpoint_dir: Path, output_path: Path) -> Path:
    """Generate Figure 1 from the three component event catalogs."""

    catalogs = load_event_catalogs(Path(checkpoint_dir))
    norm = _StretchedNorm(vmin=0, vmax=20, stretch_point=18, stretch_factor=3)
    mlt_ticks = np.asarray([0, 3, 6, 9, 12, 15, 18, 21, 24])
    x_edges = np.linspace(0, 24, FIGURE_1_MLT_HISTOGRAM_BINS + 1)
    y_edges = np.linspace(
        PC5_FREQUENCY_MIN_HZ * 1_000,
        PC5_FREQUENCY_MAX_HZ * 1_000,
        FIGURE_1_FREQUENCY_HISTOGRAM_BINS + 1,
    )

    figure = plt.figure(figsize=(15, 8))
    try:
        grid = GridSpec(
            2,
            4,
            figure=figure,
            width_ratios=[1, 1, 1, 0.05],
            hspace=0.4,
            wspace=0.3,
        )

        top_scatter = None
        bottom_meshes = []
        for column, component in enumerate(_COMPONENT_ORDER):
            data = catalogs[component].sort_values("power")
            top = figure.add_subplot(grid[0, column])
            scatter = top.scatter(
                data["t1"],
                data["freq"] * 1_000,
                c=10 * np.log10(data["power"].astype(float)),
                cmap="jet",
                marker="s",
                norm=norm,
            )
            if top_scatter is None:
                top_scatter = scatter
            top.set_title(
                f"({chr(ord('a') + column)}) {_COMPONENT_LABELS[component]}",
                fontsize=15,
                loc="left",
            )
            top.set_xticks(mlt_ticks)
            top.set_ylim(1.639, 6.7)
            top.set_xlim(0, 24)
            top.set_xlabel("MLT (hr)", fontsize=15)
            if column == 0:
                top.set_ylabel("Frequency (mHz)", fontsize=15)
            else:
                top.tick_params(labelleft=False)
            top.grid(True, linestyle="--")
            top.tick_params(labelsize=12)

            bottom = figure.add_subplot(grid[1, column])
            counts, _, _ = np.histogram2d(
                catalogs[component]["t1"],
                catalogs[component]["freq"] * 1_000,
                bins=[x_edges, y_edges],
            )
            mesh = bottom.pcolormesh(
                x_edges,
                y_edges,
                np.log10(counts.T + 1),
                cmap="plasma",
            )
            bottom_meshes.append(mesh)
            bottom.set_title(
                f"({chr(ord('d') + column)}) {_COMPONENT_LABELS[component]}",
                fontsize=15,
                loc="left",
            )
            bottom.set_xticks(mlt_ticks)
            bottom.set_ylim(1.639, 6.7)
            bottom.set_xlim(0, 24)
            bottom.set_xlabel("MLT (hr)", fontsize=15)
            if column == 0:
                bottom.set_ylabel("Frequency (mHz)", fontsize=15)
            else:
                bottom.tick_params(labelleft=False)
            bottom.grid(True, linestyle="--")
            bottom.tick_params(labelsize=15)

        if top_scatter is None:
            raise ValueError("No event data were available for Figure 1")
        top_color_axis = figure.add_subplot(grid[0, 3])
        top_colorbar = figure.colorbar(top_scatter, cax=top_color_axis)
        top_colorbar.ax.tick_params(labelsize=15)
        top_colorbar.set_label(r"10$\times$log$_{10}$(amp) (nT)", fontsize=15)
        ticks = top_colorbar.get_ticks()
        labels = [f"{tick:.0f}" if tick < 20 else r"$\geq$20" for tick in ticks]
        top_colorbar.set_ticks(ticks)
        top_colorbar.set_ticklabels(labels)

        # Each histogram autoscales independently, while the displayed colorbar
        # belongs to B_parallel.
        bottom_color_axis = figure.add_subplot(grid[1, 3])
        bottom_colorbar = figure.colorbar(bottom_meshes[-1], cax=bottom_color_axis)
        bottom_colorbar.ax.tick_params(labelsize=15)
        bottom_colorbar.set_label(r"log$_{10}$(counts + 1)", fontsize=15)
        return _atomic_save_figure(figure, Path(output_path))
    finally:
        plt.close(figure)


def _plot_power_law_panel(
    axis: plt.Axes,
    binned: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    component: str,
    color: str,
    panel_label: str,
    correlation_y: float,
) -> None:
    fit = fit_power_law(binned[x_column], binned[y_column])
    axis.plot(binned[x_column], binned[y_column], color=color, linestyle="", marker="d")
    axis.plot(
        fit.fitted_x,
        fit.fitted_y,
        color=color,
        linestyle="--",
        label=(
            rf"$y = {fit.coefficient:.2f} "
            rf"\omega_{{{_OMEGA_LABELS[component]}}}^{{{fit.exponent:.2f}}}$"
        ),
    )
    valid = np.isfinite(binned[x_column].astype(float)) & np.isfinite(
        binned[y_column].astype(float)
    )
    observed_x = binned.loc[valid, x_column].astype(float).to_numpy()
    observed_y = binned.loc[valid, y_column].astype(float).to_numpy()
    predicted = power_law(observed_x, fit.exponent, fit.coefficient)
    correlation = pearson_r(predicted, observed_y)
    axis.legend(frameon=False, fancybox=False, shadow=False, loc="upper right")
    axis.text(
        0.01,
        0.9,
        f"({panel_label}) {_COMPONENT_LABELS[component]}",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=axis.transAxes,
        fontsize=12,
        color="black",
    )
    axis.text(
        0.98,
        correlation_y,
        f"Pearson's R={correlation:.2f}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=axis.transAxes,
        fontsize=12,
        color="black",
    )


def plot_figure_2(checkpoint_dir: Path, output_path: Path) -> Path:
    """Generate Figure 2: MLT-sector median amplitude-frequency fits."""

    catalogs = load_event_catalogs(Path(checkpoint_dir))
    sectors = ("dawn", "day", "dusk", "night")
    sector_titles = ("Dawn", "Day", "Dusk", "Night")
    colors = {"radial": "tab:red", "azimuthal": "tab:blue", "parallel": "tab:orange"}
    binned = {
        component: mlt_sector_binned_medians(catalogs[component], bins=FREQUENCY_BIN_COUNT)
        for component in _COMPONENT_ORDER
    }

    figure, axes = plt.subplots(3, 4, figsize=(15, 10))
    try:
        figure.subplots_adjust(hspace=0.1, wspace=0.1)
        for row, component in enumerate(_COMPONENT_ORDER):
            for column, sector in enumerate(sectors):
                axis = axes[row, column]
                if row == 0:
                    axis.set_title(sector_titles[column])
                panel = chr(ord("a") + row * 4 + column)
                _plot_power_law_panel(
                    axis,
                    binned[component][sector],
                    x_column="frq_centroid",
                    y_column="median_power",
                    component=component,
                    color=colors[component],
                    panel_label=panel,
                    correlation_y=0.75,
                )
                axis.set_xlim(1, 7)
                axis.set_xticks([1, 2, 3, 4, 5, 6, 7])
                axis.set_ylim(1.0, 3.0)
                axis.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
                axis.grid(color="k", linestyle="dotted", linewidth=1, alpha=0.1)
                axis.yaxis.set_ticks_position("both")
                if row < 2:
                    axis.tick_params(labelbottom=False)
                else:
                    axis.set_xlabel("cent. freq. (mHz)", fontsize=12)
                if column == 0:
                    axis.set_ylabel("median amp. (nT)", fontsize=12)
                else:
                    axis.tick_params(labelleft=False)
        return _atomic_save_figure(figure, Path(output_path))
    finally:
        plt.close(figure)


def plot_figure_3(checkpoint_dir: Path, omni_path: Path, output_path: Path) -> Path:
    """Generate Figure 3 with a daily many-to-many event/OMNI expansion."""

    catalogs = load_event_catalogs(Path(checkpoint_dir))
    omni = _read_validated_omni(Path(omni_path))
    condition_omni = prepare_condition_omni(omni)
    condition_names = ("strong", "moderate", "weak")
    condition_titles = ("strong sw cond.", "moderate sw cond.", "weak sw cond.")
    colors = {"radial": "tab:red", "azimuthal": "tab:blue", "parallel": "tab:orange"}

    binned: dict[str, dict[str, pd.DataFrame]] = {}
    for component in _COMPONENT_ORDER:
        expanded = expand_events_with_daily_omni(catalogs[component], condition_omni)
        conditions = split_solar_wind_conditions(expanded)
        binned[component] = {
            name: condition_binned_statistics(conditions[name], bins=FREQUENCY_BIN_COUNT)
            for name in condition_names
        }

    figure, axes = plt.subplots(3, 3, figsize=(10, 10))
    try:
        figure.subplots_adjust(hspace=0.1, wspace=0.1)
        for row, component in enumerate(_COMPONENT_ORDER):
            for column, condition in enumerate(condition_names):
                axis = axes[row, column]
                if row == 0:
                    axis.set_title(condition_titles[column])
                panel = chr(ord("a") + row * 3 + column)
                _plot_power_law_panel(
                    axis,
                    binned[component][condition],
                    x_column="freq",
                    y_column="amp",
                    component=component,
                    color=colors[component],
                    panel_label=panel,
                    correlation_y=0.55,
                )
                axis.set_xlim(1, 7)
                axis.set_xticks([1, 2, 3, 4, 5, 6, 7])
                # Figure 3 uses the same vertical limits in all nine panels.
                axis.set_ylim(1, 4)
                axis.set_yticks([1, 2, 3, 4])
                axis.grid(color="k", linestyle="dotted", linewidth=1, alpha=0.1)
                axis.yaxis.set_ticks_position("both")
                if row < 2:
                    axis.tick_params(labelbottom=False)
                else:
                    axis.set_xlabel("cent. freq. (mHz)", fontsize=12)
                if column == 0:
                    axis.set_ylabel("med. amp. (nT)", fontsize=12)
                else:
                    axis.tick_params(labelleft=False)
        return _atomic_save_figure(figure, Path(output_path))
    finally:
        plt.close(figure)


def plot_figure_4(
    checkpoint_dir: Path,
    omni_path: Path,
    observation_counts_path: Path,
    output_path: Path,
    correlations_path: Path,
) -> tuple[Path, Path]:
    """Generate the 1/5-year Figure 4 and its Pearson-r table."""

    checkpoint_dir = Path(checkpoint_dir)
    catalogs = load_event_catalogs(checkpoint_dir)
    observation_counts = load_or_build_observation_counts(
        checkpoint_dir, Path(observation_counts_path)
    )
    occurrence = annual_occurrence_rates(catalogs, observation_counts)
    first_year = int(occurrence["radial"]["date"].iloc[0])
    omni = _read_validated_omni(Path(omni_path))
    annual_omni = annual_omni_parameters(omni, first_year=first_year)
    correlations = solar_cycle_correlations(occurrence, annual_omni)

    mlt_ticks = [1995, 2000, 2005, 2010, 2015, 2020, 2025]
    component_colors = {
        "radial": "tab:red",
        "azimuthal": "tab:blue",
        "parallel": "tab:green",
    }
    component_labels = {
        "radial": r"B$_r$",
        "azimuthal": r"B$_{\phi}$",
        "parallel": r"B$_{||}$",
    }
    secondary = (
        ("dyn_pres", "dyn_pres_hp", "dyn. press. (nPa)", (1, 3), (-0.3, 0.3)),
        ("B_z", "B_z_hp", r"$|B_z|$ (nT)", (1, 3), (-0.75, 0.75)),
        ("sw", "sw_hp", r"V$_{sw}$ (km/s)", (350, 550), (-70, 70)),
    )

    figure, axes = plt.subplots(2, 3, figsize=(18, 6))
    try:
        figure.subplots_adjust(hspace=0.15, wspace=0.5)
        for row in range(2):
            for column, (raw_name, hp_name, secondary_label, raw_limits, hp_limits) in enumerate(
                secondary
            ):
                axis = axes[row, column]
                rate_column = "oc_rate" if row == 0 else "oc_rate_hp"
                for component in _COMPONENT_ORDER:
                    axis.plot(
                        occurrence[component]["date"],
                        occurrence[component][rate_column],
                        color=component_colors[component],
                        label=component_labels[component],
                    )
                if row == 0 and column == 0:
                    axis.legend(loc="upper right", frameon=False)
                panel = chr(ord("a") + column + row * 3)
                axis.text(
                    0,
                    0.87,
                    f"({panel})",
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    transform=axis.transAxes,
                    fontsize=12,
                )
                axis.set_xticks(mlt_ticks)
                axis.set_xlim(1995, 2025)
                axis.set_ylabel("occ. rate", fontsize=12)
                axis.grid(True, linestyle="--")
                axis.tick_params(axis="both", which="major", labelsize=12)
                if row == 0:
                    axis.set_ylim(0, 6)
                    axis.tick_params(labelbottom=False)
                else:
                    axis.set_ylim(-2, 2)
                    axis.set_xlabel("time (year)", fontsize=12)

                twin = axis.twinx()
                parameter = raw_name if row == 0 else hp_name
                twin.plot(annual_omni["date"], annual_omni[parameter], color="k")
                twin.set_ylabel(secondary_label, color="k", fontsize=12)
                twin.set_ylim(*(raw_limits if row == 0 else hp_limits))
                twin.tick_params(axis="both", which="major", labelsize=12)

        # Render both products before replacing either destination.
        staging_dir = Path(tempfile.mkdtemp(prefix="pc5-figure4-"))
        try:
            staged_table = staging_dir / Path(correlations_path).name
            staged_figure = staging_dir / Path(output_path).name
            atomic_write_csv(correlations, staged_table)
            _atomic_save_figure(figure, staged_figure)

            for source, destination in (
                (staged_table, Path(correlations_path)),
                (staged_figure, Path(output_path)),
            ):
                with atomic_destination(destination, suffix=".promote.part") as temporary:
                    shutil.copyfile(source, temporary)
            return Path(output_path), Path(correlations_path)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
    finally:
        plt.close(figure)
