"""radar-nwp-bench: unified benchmark for radar + NWP precipitation nowcasting.

Public surface:
    from radar_nwp_bench import build_model, RadarNWPDataset, build_dataloader
"""
from .data import RadarNWPDataset, build_dataloader, CasePacker
from .models import build_model, BasePrecipitationModel, list_models

__all__ = [
    "RadarNWPDataset",
    "build_dataloader",
    "CasePacker",
    "build_model",
    "BasePrecipitationModel",
    "list_models",
]
__version__ = "0.1.0"
