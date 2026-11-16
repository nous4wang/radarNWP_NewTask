"""Pack each case into a single container file.

The raw archive holds ~62 800 individual ``.npy`` files across 62 cases. Every
training sample therefore pays thousands of filesystem round-trips. Packing
writes one ``.npz`` per case holding the whole temporal cube, so a sample is one
contiguous read.

Design notes:
    * The container stores **cleaned float32** arrays, not raw int16, so the
      sentinel decoding happens once at pack time rather than on every epoch.
    * RADAR and LABEL are stored on the RADAR grid (301x301). NWP is stored on
      its own 167x167 grid and resampled on the fly, because resampling is
      cheap relative to disk I/O and keeps the container 3x smaller.
    * Timestamps are stored as ``MMDDHHMM`` integers so the loader can rebuild
      the alignment without re-reading filenames.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coordinates import NWP_SHAPE, RADAR_SHAPE
from .index import CaseIndex, NWP_VARS, RADAR_VARS, scan_case
from .transforms import clean_label_rain, clean_nwp_field, clean_radar_field


@dataclass
class PackStats:
    case_id: str
    n_radar: int
    n_label: int
    n_nwp: int
    radar_vars: list[str]
    nwp_vars: list[str]
    bytes_written: int
    seconds: float


class CasePacker:
    """Convert on-disk cases into single-file containers."""

    def __init__(
        self,
        radar_vars: tuple[str, ...] = RADAR_VARS,
        nwp_vars: tuple[str, ...] = NWP_VARS,
        radar_shape: tuple[int, int] = RADAR_SHAPE,
        nwp_shape: tuple[int, int] = NWP_SHAPE,
        compress: bool = True,
    ) -> None:
        self.radar_vars = tuple(radar_vars)
        self.nwp_vars = tuple(nwp_vars)
        self.radar_shape = radar_shape
        self.nwp_shape = nwp_shape
        self.compress = compress

    # -- public API ----------------------------------------------------------
    def pack_case(self, case_dir: str | Path, out_dir: str | Path) -> PackStats:
        """Pack one case into ``out_dir/<case_id>.npz``."""
        import time

        t0 = time.perf_counter()
        case_dir = Path(case_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        index = scan_case(case_dir)
        radar_times = index.radar_times
        label_times = index.label_times
        nwp_times = index.nwp_times

        radar = np.zeros(
            (len(radar_times), len(self.radar_vars), *self.radar_shape), dtype=np.float32
        )
        for ti, stamp in enumerate(radar_times):
            for vi, var in enumerate(self.radar_vars):
                path = index.radar_file(var, stamp)
                if path is None:
                    continue
                radar[ti, vi] = clean_radar_field(np.load(path), var)

        label = np.zeros((len(label_times), 1, *self.radar_shape), dtype=np.float32)
        for ti, stamp in enumerate(label_times):
            path = index.label_file(stamp)
            if path is None:
                continue
            label[ti, 0] = clean_label_rain(np.load(path))

        nwp = np.zeros(
            (len(nwp_times), len(self.nwp_vars), *self.nwp_shape), dtype=np.float32
        )
        for ti, stamp in enumerate(nwp_times):
            for vi, var in enumerate(self.nwp_vars):
                path = index.nwp_file(var, stamp)
                if path is None:
                    continue
                nwp[ti, vi] = clean_nwp_field(np.load(path))

        out_path = out_dir / f"{case_dir.name}.npz"
        payload = {
            "radar": radar,
            "label": label,
            "nwp": nwp,
            "radar_times": np.asarray(radar_times, dtype=np.int64),
            "label_times": np.asarray(label_times, dtype=np.int64),
            "nwp_times": np.asarray(nwp_times, dtype=np.int64),
            "radar_vars": np.asarray(self.radar_vars),
            "nwp_vars": np.asarray(self.nwp_vars),
            "meta": json.dumps(
                {
                    "case_id": case_dir.name,
                    "radar_shape": list(self.radar_shape),
                    "nwp_shape": list(self.nwp_shape),
                    "source": str(case_dir),
                }
            ),
        }
        saver = np.savez_compressed if self.compress else np.savez
        saver(out_path, **payload)

        return PackStats(
            case_id=case_dir.name,
            n_radar=len(radar_times),
            n_label=len(label_times),
            n_nwp=len(nwp_times),
            radar_vars=list(self.radar_vars),
            nwp_vars=list(self.nwp_vars),
            bytes_written=out_path.stat().st_size,
            seconds=time.perf_counter() - t0,
        )

    def pack_case_mmap(
        self, case_dir: str | Path, out_dir: str | Path
    ) -> PackStats:
        """Pack one case as a directory of ``.npy`` files.

        A ``.npz`` is a zip archive, so every access has to inflate the whole
        member: 1.4 s and 280 MB for a typical case. That is fatal when the
        loader is shuffled across many cases, because the cache cannot hold
        them all.

        Plain ``.npy`` files can be memory-mapped, so the operating system
        pages them in on demand and *all* worker processes share one copy of
        the page cache. The layout is ``out_dir/<case_id>/{radar,label,nwp}.npy``
        plus a small ``meta.json``.

        Args:
            case_dir: the raw case directory.
            out_dir: the root of the packed tree.

        Returns:
            Statistics for the packed case.
        """
        import time

        t0 = time.perf_counter()
        case_dir = Path(case_dir)
        out_dir = Path(out_dir)
        index = scan_case(case_dir)

        radar_times = index.radar_times
        label_times = index.label_times
        nwp_times = index.nwp_times

        radar = np.zeros(
            (len(radar_times), len(self.radar_vars), *self.radar_shape), dtype=np.float32
        )
        for ti, stamp in enumerate(radar_times):
            for vi, var in enumerate(self.radar_vars):
                path = index.radar_file(var, stamp)
                if path is None:
                    continue
                radar[ti, vi] = clean_radar_field(np.load(path), var)

        label = np.zeros((len(label_times), 1, *self.radar_shape), dtype=np.float32)
        for ti, stamp in enumerate(label_times):
            path = index.label_file(stamp)
            if path is None:
                continue
            label[ti, 0] = clean_label_rain(np.load(path))

        nwp = np.zeros(
            (len(nwp_times), len(self.nwp_vars), *self.nwp_shape), dtype=np.float32
        )
        for ti, stamp in enumerate(nwp_times):
            for vi, var in enumerate(self.nwp_vars):
                path = index.nwp_file(var, stamp)
                if path is None:
                    continue
                nwp[ti, vi] = clean_nwp_field(np.load(path))

        case_out = out_dir / case_dir.name
        case_out.mkdir(parents=True, exist_ok=True)
        np.save(case_out / "radar.npy", radar)
        np.save(case_out / "label.npy", label)
        np.save(case_out / "nwp.npy", nwp)
        (case_out / "meta.json").write_text(
            json.dumps(
                {
                    "case_id": case_dir.name,
                    "radar_shape": list(self.radar_shape),
                    "nwp_shape": list(self.nwp_shape),
                    "radar_times": radar_times,
                    "label_times": label_times,
                    "nwp_times": nwp_times,
                    "radar_vars": list(self.radar_vars),
                    "nwp_vars": list(self.nwp_vars),
                    "source": str(case_dir),
                }
            ),
            encoding="utf-8",
        )

        bytes_written = sum(p.stat().st_size for p in case_out.iterdir())
        return PackStats(
            case_id=case_dir.name,
            n_radar=len(radar_times),
            n_label=len(label_times),
            n_nwp=len(nwp_times),
            radar_vars=list(self.radar_vars),
            nwp_vars=list(self.nwp_vars),
            bytes_written=bytes_written,
            seconds=time.perf_counter() - t0,
        )

    def pack_all(
        self,
        data_root: str | Path,
        out_dir: str | Path,
        workers: int = 4,
        progress: bool = True,
    ) -> list[PackStats]:
        """Pack every case under ``data_root``."""
        from .index import discover_cases

        cases = discover_cases(data_root)
        stats: list[PackStats] = []
        if workers <= 1:
            for case_dir in cases:
                st = self.pack_case(case_dir, out_dir)
                stats.append(st)
                if progress:
                    print(f"  {st.case_id}: radar={st.n_radar} label={st.n_label} "
                          f"nwp={st.n_nwp} {st.bytes_written/1e6:.1f}MB {st.seconds:.1f}s")
            return stats

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.pack_case, case_dir, out_dir): case_dir
                for case_dir in cases
            }
            for fut in as_completed(futures):
                st = fut.result()
                stats.append(st)
                if progress:
                    print(f"  {st.case_id}: radar={st.n_radar} label={st.n_label} "
                          f"nwp={st.n_nwp} {st.bytes_written/1e6:.1f}MB {st.seconds:.1f}s")
        stats.sort(key=lambda s: s.case_id)
        return stats

    def pack_all_mmap(
        self,
        data_root: str | Path,
        out_dir: str | Path,
        workers: int = 4,
        progress: bool = True,
    ) -> list[PackStats]:
        """Pack every case under ``data_root`` in the memory-mappable layout.

        See :meth:`pack_case_mmap` for why this exists.
        """
        from .index import discover_cases

        cases = discover_cases(data_root)
        stats: list[PackStats] = []
        if workers <= 1:
            for case_dir in cases:
                st = self.pack_case_mmap(case_dir, out_dir)
                stats.append(st)
                if progress:
                    print(f"  {st.case_id}: radar={st.n_radar} label={st.n_label} "
                          f"nwp={st.n_nwp} {st.bytes_written/1e6:.1f}MB {st.seconds:.1f}s")
            return stats

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.pack_case_mmap, case_dir, out_dir): case_dir
                for case_dir in cases
            }
            for fut in as_completed(futures):
                st = fut.result()
                stats.append(st)
                if progress:
                    print(f"  {st.case_id}: radar={st.n_radar} label={st.n_label} "
                          f"nwp={st.n_nwp} {st.bytes_written/1e6:.1f}MB {st.seconds:.1f}s")
        stats.sort(key=lambda s: s.case_id)
        return stats


def load_packed(path: str | Path) -> dict:
    """Read a packed case container into a plain dict of arrays."""
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    if "meta" in out:
        out["meta"] = json.loads(str(out["meta"]))
    return out


def load_packed_mmap(path: str | Path) -> dict:
    """Read a memory-mappable packed case directory.

    The arrays are opened with ``mmap_mode="r"``, so nothing is read until
    it is touched and the operating system shares the page cache between all
    worker processes. A ``.npz`` cannot be memory-mapped -- it is a zip
    archive, so every access inflates the whole member (1.4 s and 280 MB for
    a typical case).

    Args:
        path: the case directory holding ``radar.npy``, ``label.npy``,
            ``nwp.npy`` and ``meta.json``.

    Returns:
        A dict of arrays, memory-mapped where possible.
    """
    path = Path(path)
    out: dict = {}
    for name in ("radar", "label", "nwp"):
        array = path / f"{name}.npy"
        if array.is_file():
            out[name] = np.load(array, mmap_mode="r")
    meta_path = path / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out["meta"] = meta
        out["radar_times"] = np.asarray(meta.get("radar_times", []), dtype=np.int64)
        out["label_times"] = np.asarray(meta.get("label_times", []), dtype=np.int64)
        out["nwp_times"] = np.asarray(meta.get("nwp_times", []), dtype=np.int64)
        out["radar_vars"] = np.asarray(meta.get("radar_vars", []))
        out["nwp_vars"] = np.asarray(meta.get("nwp_vars", []))
    return out
