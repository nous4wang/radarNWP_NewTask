"""Tests for the packing and dataset layers.

The real archive holds ~62 800 individual ``.npy`` files, so every case is
packed into a single container.  These tests exercise that path end to end on
a synthetic case, plus the sample indexing and the dataloader.

Everything here works on tiny arrays in a temporary directory: the archive
itself is never touched.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from radar_nwp_bench.data.alignment import NWPAlignment
from radar_nwp_bench.data.dataset import RadarNWPDataset, build_dataloader
from radar_nwp_bench.data.index import add_minutes, encode_stamp
from radar_nwp_bench.data.pack import CasePacker, load_packed


#: 30 six-minute radar frames starting at 05-17 15:00.
RADAR_TIMES: list[int] = [
    add_minutes(encode_stamp(517, 1500), 6 * k) for k in range(30)
]
#: Four whole hours of NWP.
NWP_TIMES: list[int] = [encode_stamp(517, 1500 + 100 * k) for k in range(4)]

RADAR_SHAPE = (4, 4)
NWP_SHAPE = (3, 3)


def _write_raw_case(root: Path, case_id: str = "CP_01_05171518_00001") -> Path:
    """Write a minimal raw case tree and return its path."""
    case = root / case_id
    for var in ("CAP20", "CR"):
        var_dir = case / "RADAR" / var
        var_dir.mkdir(parents=True, exist_ok=True)
        for stamp in ("0517-1500", "0517-1506", "0517-1512"):
            np.save(
                var_dir / f"CP_RADA_00001_{stamp}_SA_CR.npy",
                np.zeros(RADAR_SHAPE, dtype=np.int16),
            )
    label_dir = case / "LABEL" / "RA"
    label_dir.mkdir(parents=True, exist_ok=True)
    for stamp in ("0517-1500", "0517-1506", "0517-1512"):
        np.save(label_dir / f"CP_Label_RA_00001_{stamp}.npy", np.zeros(RADAR_SHAPE, dtype=np.int16))
    var_dir = case / "NWP" / "CAPE"
    var_dir.mkdir(parents=True, exist_ok=True)
    for stamp in ("0517-1500", "0517-1600"):
        np.save(var_dir / f"CP_RRA_00001_{stamp}_CAPE.npy", np.zeros(NWP_SHAPE, dtype=np.float16))
    return case


def _write_packed_case(root: Path, case_id: str = "CP_01_05171518_00001") -> Path:
    """Write a packed container with the real grid shapes."""
    n = len(RADAR_TIMES)
    path = root / f"{case_id}.npz"
    np.savez_compressed(
        path,
        radar=np.zeros((n, 1) + (301, 301), dtype=np.float32),
        label=np.zeros((n, 1) + (301, 301), dtype=np.float32),
        nwp=np.zeros((n, 1) + (167, 167), dtype=np.float32),
        radar_times=np.asarray(RADAR_TIMES, dtype=np.int64),
        label_times=np.asarray(RADAR_TIMES, dtype=np.int64),
        nwp_times=np.asarray(NWP_TIMES, dtype=np.int64),
        radar_vars=np.asarray(["CAP20"]),
        nwp_vars=np.asarray(["CAPE"]),
        meta=json.dumps({"case_id": case_id}),
    )
    return path


# ---------------------------------------------------------------------------
# CasePacker / load_packed
# ---------------------------------------------------------------------------
def test_packer_writes_a_container_for_a_case():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case = _write_raw_case(root)
        out = root / "packed"

        packer = CasePacker(
            radar_vars=("CAP20", "CR"),
            nwp_vars=("CAPE",),
            radar_shape=RADAR_SHAPE,
            nwp_shape=NWP_SHAPE,
            compress=False,
        )
        stats = packer.pack_case(case, out)

        assert stats.case_id == "CP_01_05171518_00001"
        assert stats.n_radar == 3
        assert stats.n_label == 3
        assert stats.n_nwp == 2
        assert stats.bytes_written > 0
        assert (out / "CP_01_05171518_00001.npz").is_file()


def test_packed_container_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case = _write_raw_case(root)
        out = root / "packed"

        packer = CasePacker(
            radar_vars=("CAP20", "CR"),
            nwp_vars=("CAPE",),
            radar_shape=RADAR_SHAPE,
            nwp_shape=NWP_SHAPE,
            compress=False,
        )
        packer.pack_case(case, out)
        data = load_packed(out / "CP_01_05171518_00001.npz")

        assert data["radar"].shape == (3, 2) + RADAR_SHAPE
        assert data["label"].shape == (3, 1) + RADAR_SHAPE
        assert data["nwp"].shape == (2, 1) + NWP_SHAPE
        assert list(data["radar_vars"]) == ["CAP20", "CR"]
        assert list(data["nwp_vars"]) == ["CAPE"]
        assert data["meta"]["case_id"] == "CP_01_05171518_00001"


def test_packed_container_keeps_the_timestamps():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case = _write_raw_case(root)
        out = root / "packed"

        packer = CasePacker(
            radar_vars=("CAP20", "CR"),
            nwp_vars=("CAPE",),
            radar_shape=RADAR_SHAPE,
            nwp_shape=NWP_SHAPE,
            compress=False,
        )
        packer.pack_case(case, out)
        data = load_packed(out / "CP_01_05171518_00001.npz")

        assert data["radar_times"].tolist() == [
            encode_stamp(517, 1500),
            encode_stamp(517, 1506),
            encode_stamp(517, 1512),
        ]
        assert data["nwp_times"].tolist() == [
            encode_stamp(517, 1500),
            encode_stamp(517, 1600),
        ]


# ---------------------------------------------------------------------------
# RadarNWPDataset
# ---------------------------------------------------------------------------
def _dataset(root: Path, **kwargs) -> RadarNWPDataset:
    defaults = dict(
        packed=True,
        radar_vars=("CAP20",),
        nwp_vars=("CAPE",),
        patch_size=32,
        random_crop=False,
    )
    defaults.update(kwargs)
    return RadarNWPDataset(root, **defaults)


def test_dataset_builds_samples_from_a_packed_case():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_packed_case(root)
        dataset = _dataset(root)

        assert len(dataset) > 0
        for spec in dataset.samples:
            assert len(spec.radar_idx) == dataset.in_len
            assert len(spec.label_idx) == dataset.out_len
            assert len(spec.nwp_idx) == 2


def test_dataset_samples_carry_the_alignment_lead_times():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_packed_case(root)
        dataset = _dataset(root, alignment=NWPAlignment.FORECAST)

        for spec in dataset.samples:
            assert len(spec.nwp_lead_minutes) == 2
            # FORECAST must hand the model a slice valid after the issue time.
            assert spec.has_forecast_nwp is True
            assert max(spec.nwp_lead_minutes) > 0


def test_dataset_item_has_the_documented_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_packed_case(root)
        dataset = _dataset(root)

        item = dataset[0]
        assert item["x_radar"].shape == (11, 1, 32, 32)
        assert item["x_nwp"].shape == (2, 1, 32, 32)
        assert item["y_rain"].shape == (10, 1, 32, 32)
        assert item["issue_time"].dtype == torch.long
        assert item["nwp_lead"].shape == (2,)


def test_dataset_item_is_finite():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_packed_case(root)
        dataset = _dataset(root)

        item = dataset[0]
        assert torch.isfinite(item["x_radar"]).all()
        assert torch.isfinite(item["x_nwp"]).all()
        assert torch.isfinite(item["y_rain"]).all()


def test_dataset_can_normalize_the_target():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_packed_case(root)
        dataset = _dataset(root, normalize_target=True)

        item = dataset[0]
        assert (item["y_rain"] >= 0.0).all()


def test_dataset_raises_when_no_case_is_found():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _dataset(Path(tmp))
        except FileNotFoundError:
            return
        raise AssertionError("an empty root should raise FileNotFoundError")


# ---------------------------------------------------------------------------
# build_dataloader
# ---------------------------------------------------------------------------
def test_build_dataloader_batches_the_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_packed_case(root)
        dataset = _dataset(root)

        loader = build_dataloader(
            dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=False
        )
        batch = next(iter(loader))

        assert batch["x_radar"].shape == (2, 11, 1, 32, 32)
        assert batch["x_nwp"].shape == (2, 2, 1, 32, 32)
        assert batch["y_rain"].shape == (2, 10, 1, 32, 32)


def test_build_dataloader_with_a_seed_is_reproducible():
    """Two loaders built with the same seed shuffle in the same order."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_packed_case(root)
        dataset = _dataset(root)

        def order() -> list[int]:
            loader = build_dataloader(
                dataset,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                drop_last=False,
                seed=1234,
            )
            return [b["issue_time"].item() for b in loader]

        assert order() == order()


# ---------------------------------------------------------------------------
# sliding patches over the full 301x301 grid
# ---------------------------------------------------------------------------
def _sliding_dataset(patch_size, stride):
    from radar_nwp_bench.data import RadarNWPDataset

    class _Fake(RadarNWPDataset):
        def __init__(self):
            self.patch_size = patch_size
            self.slide_stride = stride
            self.random_crop = False
            self._patch_origins = ()
            self.set_grid_shape(301, 301)

    return _Fake()


def test_sliding_window_covers_the_whole_grid():
    for size, stride in ((32, 32), (64, 64), (128, 128), (64, 32)):
        ds = _sliding_dataset(size, stride)
        origins = ds.patch_origins
        ys = sorted({o[0] for o in origins})
        xs = sorted({o[1] for o in origins})
        assert ys[0] == 0 and xs[0] == 0, (size, stride)
        assert ys[-1] == 301 - size, (size, stride)
        assert xs[-1] == 301 - size, (size, stride)


def test_sliding_window_never_leaves_the_grid():
    for size, stride in ((32, 32), (64, 16), (128, 64)):
        ds = _sliding_dataset(size, stride)
        for y, x in ds.patch_origins:
            assert 0 <= y and y + size <= 301, (size, y)
            assert 0 <= x and x + size <= 301, (size, x)


def test_sliding_window_position_count():
    assert len(_sliding_dataset(32, 32).patch_origins) == 10 * 10
    assert len(_sliding_dataset(64, 64).patch_origins) == 5 * 5
    assert len(_sliding_dataset(128, 128).patch_origins) == 3 * 3


def test_sliding_window_last_position_is_appended():
    """A stride that does not divide the span still covers the far edge."""
    ds = _sliding_dataset(64, 100)
    origins = ds.patch_origins
    ys = sorted({o[0] for o in origins})
    assert ys[-1] == 301 - 64


def test_patch_size_at_or_above_the_grid_keeps_a_single_position():
    ds = _sliding_dataset(301, 32)
    assert ds.patch_origins == ((0, 0),)


# ---------------------------------------------------------------------------
# the sliding tiling is not exact, and that is deliberate
# ---------------------------------------------------------------------------
def test_sliding_tiling_covers_the_grid_but_overlaps_at_the_far_edge():
    """301 is not a multiple of the patch size, so the origins are spread.

    The origins used to step by a fixed stride and append one last origin
    at the far edge, which left a narrow, densely overlapping strip there
    and nothing in between. They are now spread evenly across the grid, so
    the overlap is the same everywhere.

    What still matters for the verification is that the grid is covered
    without a gap: a gap would be a region that no tile ever scores, and
    the rain that fell there would never be counted as a miss.
    """
    from radar_nwp_bench.data import RadarNWPDataset

    class _Fake(RadarNWPDataset):
        def __init__(self, size, stride):
            self.patch_size = size
            self.slide_stride = stride
            self.random_crop = False
            self._patch_origins = ()
            self.set_grid_shape(301, 301)

    for size, stride in ((32, 32), (64, 64), (128, 128)):
        ds = _Fake(size, stride)
        origins = sorted({o[0] for o in ds.patch_origins})
        gaps = [origins[i + 1] - origins[i] for i in range(len(origins) - 1)]

        # the far edge is always reached, and the grid is fully covered
        assert origins[-1] == 301 - size
        covered = set()
        for y in origins:
            covered.update(range(y, y + size))
        assert len(covered) == 301, f"{size}/{stride} leaves a gap"

        # the origins are spread evenly: no gap is much larger than another
        assert max(gaps) - min(gaps) <= 1, (
            f"{size}/{stride}: the origins are not evenly spread "
            f"(gaps {gaps})"
        )
        # and consecutive tiles overlap, so no pixel falls between them
        assert min(gaps) < size, f"{size}/{stride} leaves a gap between tiles"


# ---------------------------------------------------------------------------
# the loader must not be the bottleneck
# ---------------------------------------------------------------------------
def test_case_cache_is_actually_populated():
    """The cache used to be read but never written.

    ``_load_case`` looked the path up in ``self._cache`` and, on a miss,
    loaded the case -- but never stored it. The eviction branch was dead
    code, so every single item decompressed the whole ``.npz`` again
    (about 1.4 s), which made the loader the bottleneck and left the GPU
    idle. The cache must hold the case after the first access.
    """
    from radar_nwp_bench.data import RadarNWPDataset

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case = root / "CP_00_05170000_00001.npz"
        np.savez(
            case,
            radar=np.zeros((4, 10, 301, 301), dtype=np.float32),
            label=np.zeros((4, 1, 301, 301), dtype=np.float32),
            nwp=np.zeros((4, 27, 167, 167), dtype=np.float32),
            radar_times=np.arange(4, dtype=np.int64),
            label_times=np.arange(4, dtype=np.int64),
            nwp_times=np.arange(4, dtype=np.int64),
            radar_vars=np.array(["CAP20"]),
            nwp_vars=np.array(["PWAT"]),
            meta="{}",
        )
        ds = RadarNWPDataset(root, packed=True, patch_size=64, in_len=2,
                             out_len=2, radar_vars=("CAP20",),
                             nwp_vars=("PWAT",), cache_bytes=64 << 20)
        assert ds._cache == {}, "the cache must start empty"

        # Touch the case through the normal path.
        ds._load_case(case)
        assert case in ds._cache, "the loaded case was never cached"
        assert ds._cache_bytes > 0, "the cache did not account for the bytes"

        # A second access must reuse the entry rather than reload it.
        before = ds._cache_bytes
        ds._load_case(case)
        assert ds._cache_bytes == before
        assert len(ds._cache_order) == 1


def test_case_cache_evicts_to_the_byte_budget():
    """The cache is bounded by bytes, not by entry count.

    Packed cases vary from 180 MB to 450 MB, so a fixed number of entries
    would either waste memory or thrash. The budget must be respected after
    every load.
    """
    from radar_nwp_bench.data import RadarNWPDataset

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(4):
            np.savez(
                root / f"CP_{i:02d}_05170000_00001.npz",
                radar=np.zeros((4, 10, 301, 301), dtype=np.float32),
                label=np.zeros((4, 1, 301, 301), dtype=np.float32),
                nwp=np.zeros((4, 27, 167, 167), dtype=np.float32),
                radar_times=np.arange(4, dtype=np.int64),
                label_times=np.arange(4, dtype=np.int64),
                nwp_times=np.arange(4, dtype=np.int64),
                radar_vars=np.array(["CAP20"]),
                nwp_vars=np.array(["PWAT"]),
                meta="{}",
            )
        budget = 6 << 20   # smaller than a single case
        ds = RadarNWPDataset(root, packed=True, patch_size=64, in_len=2,
                             out_len=2, radar_vars=("CAP20",),
                             nwp_vars=("PWAT",), cache_bytes=budget)
        for path in sorted(root.glob("*.npz")):
            ds._load_case(path)
            assert ds._cache_bytes <= budget, (
                f"cache holds {ds._cache_bytes} bytes, over the {budget} budget"
            )


def test_patch_resampling_matches_the_full_registration():
    """The per-patch NWP resampling must equal the full-grid one.

    The fast path crops the source window before up-sampling, so it has to
    agree with up-sampling the whole field and centre-cropping. It must be
    exact, not merely close: the two routes feed the same model.
    """
    from radar_nwp_bench.data.coordinates import bilinear_upsample_center_crop

    rng = np.random.default_rng(0)
    field = rng.random((4, 167, 167)).astype(np.float32)
    full = bilinear_upsample_center_crop(field)

    import radar_nwp_bench.data.dataset as ds_mod

    ds = object.__new__(ds_mod.RadarNWPDataset)
    for size, y0, x0 in ((64, 0, 0), (64, 100, 100), (64, 237, 237),
                          (32, 33, 211), (128, 173, 173)):
        got = ds._resample_nwp_patch(field, [0, 1, 2, 3], y0, x0, size)
        want = full[:, y0:y0 + size, x0:x0 + size]
        assert np.array_equal(got, want), (size, y0, x0)


# ---------------------------------------------------------------------------
# the memory-mappable packed layout
# ---------------------------------------------------------------------------
def test_mmap_layout_matches_the_npz_layout():
    """The two packed layouts must produce identical samples.

    The mmap layout exists only for speed: a ``.npz`` is a zip archive, so
    every access has to inflate the whole member (1.4 s and 280 MB for a
    typical case), while ``.npy`` files can be paged in on demand. The
    numbers must not change.
    """
    from radar_nwp_bench.data import RadarNWPDataset

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        npz_dir = root / "npz"
        mmap_dir = root / "mmap"
        npz_dir.mkdir()
        mmap_dir.mkdir()

        radar = np.arange(4 * 2 * 9 * 9, dtype=np.float32).reshape(4, 2, 9, 9)
        label = np.arange(4 * 1 * 9 * 9, dtype=np.float32).reshape(4, 1, 9, 9)
        nwp = np.arange(4 * 1 * 5 * 5, dtype=np.float32).reshape(4, 1, 5, 5)
        times = np.arange(4, dtype=np.int64)
        np.savez(
            npz_dir / "CP_00_05170000_00001.npz",
            radar=radar, label=label, nwp=nwp,
            radar_times=times, label_times=times, nwp_times=times,
            radar_vars=np.array(["CAP20", "CR"]), nwp_vars=np.array(["PWAT"]),
            meta="{}",
        )
        case_dir = mmap_dir / "CP_00_05170000_00001"
        case_dir.mkdir()
        np.save(case_dir / "radar.npy", radar)
        np.save(case_dir / "label.npy", label)
        np.save(case_dir / "nwp.npy", nwp)
        (case_dir / "meta.json").write_text(
            json.dumps({
                "case_id": "CP_00_05170000_00001",
                "radar_times": times.tolist(),
                "label_times": times.tolist(),
                "nwp_times": times.tolist(),
                "radar_vars": ["CAP20", "CR"],
                "nwp_vars": ["PWAT"],
            }),
            encoding="utf-8",
        )

        kw = dict(packed=True, in_len=2, out_len=2, patch_size=9,
                  radar_vars=("CAP20", "CR"), nwp_vars=("PWAT",))
        a = RadarNWPDataset(npz_dir, **kw)
        b = RadarNWPDataset(mmap_dir, **kw)
        assert len(a.samples) == len(b.samples)
        for i in range(len(a.samples)):
            ia, ib = a[i], b[i]
            for key in ("x_radar", "x_nwp", "y_rain"):
                assert torch.equal(ia[key], ib[key]), key


# ---------------------------------------------------------------------------
# the grid extent is a property of the data, not of the crop
# ---------------------------------------------------------------------------
def test_grid_shape_is_the_full_archive_extent():
    """A centre crop still comes out of a 301x301 field.

    The extent is a fact about the archive. Deriving it from the patch
    origins would report 182x182 for a centre crop, and the verification
    would then stitch the tiles into the wrong field.
    """
    from radar_nwp_bench.data import RadarNWPDataset

    class _Fake(RadarNWPDataset):
        def __init__(self, stride):
            self.patch_size = 64
            self.slide_stride = stride
            self.random_crop = False
            self._patch_origins = ()
            self._grid_shape = ()
            self.set_grid_shape(301, 301)

    assert _Fake(None).grid_shape == (301, 301)
    assert _Fake(64).grid_shape == (301, 301)
