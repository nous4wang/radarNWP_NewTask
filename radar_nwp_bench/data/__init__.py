"""Data layer: case indexing, packing, alignment, and the dataset."""
from .alignment import NWPAlignment, NWPWindow, select_nwp_window
from .coordinates import (
    GridSpec,
    bilinear_upsample_center_crop,
    parse_grid_file,
    resample_nwp_to_radar_torch,
)
from .dataset import RadarNWPDataset, SampleSpec, build_dataloader
from .index import (
    NWP_VARS,
    RADAR_VARS,
    CaseIndex,
    add_minutes,
    discover_cases,
    floor_hour,
    minutes_between,
    scan_all,
    scan_case,
)
from .pack import CasePacker, load_packed
from .transforms import (
    clean_label_rain,
    clean_nwp_field,
    clean_radar_field,
    denormalize_rain,
    normalize_rain,
    radar_log_transform,
)

__all__ = [
    "NWPAlignment",
    "NWPWindow",
    "select_nwp_window",
    "GridSpec",
    "bilinear_upsample_center_crop",
    "parse_grid_file",
    "resample_nwp_to_radar_torch",
    "RadarNWPDataset",
    "SampleSpec",
    "build_dataloader",
    "NWP_VARS",
    "RADAR_VARS",
    "CaseIndex",
    "add_minutes",
    "discover_cases",
    "floor_hour",
    "minutes_between",
    "scan_all",
    "scan_case",
    "CasePacker",
    "load_packed",
    "clean_label_rain",
    "clean_nwp_field",
    "clean_radar_field",
    "denormalize_rain",
    "normalize_rain",
    "radar_log_transform",
]
