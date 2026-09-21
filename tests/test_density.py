import pytest

from conquest2a.density import density, bandden, chden

DATA_DIR = "tests/data/banddensity"


# density class


def test_density_single_file() -> None:
    d = density((1, 0, 0), 0.0, paths=[f"{DATA_DIR}/Band000098den_totS1.cube"])
    assert d.densities[0][0].shape[0] == 72
    assert d.densities[0][0].shape[1] == 72
    assert d.densities[0][0].shape[2] == 100


def test_density_add_two_files() -> None:
    d = density(
        (1, 0, 0),
        0.0,
        paths=[
            f"{DATA_DIR}/Band000098den_totS1.cube",
            f"{DATA_DIR}/Band000098den_totS2.cube",
        ],
        operations="+",
    )
    assert d.densities[0][0].shape[0] == 72
    assert d.densities[0][0].shape[1] == 72
    assert d.densities[0][0].shape[2] == 100


def test_density_sub_two_files() -> None:
    d = density(
        (1, 0, 0),
        0.0,
        paths=[
            f"{DATA_DIR}/Band000098den_totS1.cube",
            f"{DATA_DIR}/Band000098den_totS2.cube",
        ],
        operations="-",
    )

    assert d.densities[0][0].shape[0] == 72
    assert d.densities[0][0].shape[1] == 72
    assert d.densities[0][0].shape[2] == 100


def test_density_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        density((1, 0, 0), 0.0, paths=[f"{DATA_DIR}/does_not_exist.cube"])


def test_density_rejects_zero_miller_indices() -> None:
    with pytest.raises(ValueError):
        density((0, 0, 0), 0.0, paths=[f"{DATA_DIR}/Band000098den_totS1.cube"])


def test_density_rejects_wrong_number_of_operations() -> None:
    with pytest.raises(ValueError):
        density(
            (1, 0, 0),
            0.0,
            paths=[
                f"{DATA_DIR}/Band000098den_totS1.cube",
                f"{DATA_DIR}/Band000098den_totS2.cube",
            ],
            operations="+-",
        )


# bandden class


def test_bandden_locates_files() -> None:
    bd = bandden(DATA_DIR, (1, 0, 0), 0.0, operations="+", bands=[98])
    assert len(bd.all_dens_files) > 0
    assert 98 in bd.bands


def test_bandden_filters_by_band_summed_over_kpoints() -> None:
    bd = bandden(DATA_DIR, (1, 0, 0), 0.0, operations="+", bands=[98])
    assert len(bd.filtered_dens_files) == 2
    for f in bd.filtered_dens_files:
        assert "Band000098" in f
        assert "tot" in f


def test_bandden_filters_by_kpoint() -> None:
    bd = bandden(DATA_DIR, (1, 0, 0), 0.0, operations="+", bands=[98], kpts=[1])
    assert len(bd.filtered_dens_files) == 2
    for f in bd.filtered_dens_files:
        assert "kp001" in f


def test_bandden_filters_by_spin() -> None:
    bd = bandden(DATA_DIR, (1, 0, 0), 0.0, operations="", bands=[98], spin=1)
    assert len(bd.filtered_dens_files) == 1
    assert "S1.cube" in bd.filtered_dens_files[0]


def test_bandden_filters_by_kpoint_and_spin() -> None:
    bd = bandden(DATA_DIR, (1, 0, 0), 0.0, operations="", bands=[98], kpts=[1], spin=2)
    assert len(bd.filtered_dens_files) == 1
    assert "kp001" in bd.filtered_dens_files[0]
    assert "S2.cube" in bd.filtered_dens_files[0]


def test_bandden_second_band_summed_only() -> None:
    bd = bandden(DATA_DIR, (1, 0, 0), 0.0, operations="+", bands=[99])
    assert len(bd.filtered_dens_files) == 2
    for f in bd.filtered_dens_files:
        assert "Band000099" in f
        assert "tot" in f


def test_bandden_unknown_band_raises() -> None:
    with pytest.raises(ValueError):
        bandden(DATA_DIR, (1, 0, 0), 0.0, operations="", bands=[999])


def test_bandden_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError):
        bandden("tests/data/banddensity_missing", (1, 0, 0), 0.0, operations="", bands=[1])


DATA_DIR2 = "tests/data/chargedensity"

# chden class


def test_chden_locates_unpolarised_file_by_default() -> None:
    c = chden(DATA_DIR2, (1, 0, 0), 0.0, operations="")
    names = [f.split("/")[-1] for f in c.filtered_dens_files]
    assert names == ["chden.cube"]
    assert c.density.data.ndim == 3


def test_chden_locates_all_matching_files() -> None:
    c = chden(DATA_DIR2, (1, 0, 0), 0.0, operations="")
    # chden.cube, chden_up.cube, chden_dn.cube all match the default "chden" stub
    names = {f.split("/")[-1] for f in c.all_dens_files}
    assert names == {"chden.cube", "chden_up.cube", "chden_dn.cube"}


def test_chden_spin_polarised_selects_up_and_dn() -> None:
    c = chden(DATA_DIR2, (1, 0, 0), 0.0, operations="+", spin=1)
    names = {f.split("/")[-1] for f in c.filtered_dens_files}
    assert names == {"chden_up.cube", "chden_dn.cube"}
    assert c.density.data.ndim == 3


def test_chden_spin_value_does_not_discriminate_up_vs_dn() -> None:
    c_spin1 = chden(DATA_DIR2, (1, 0, 0), 0.0, operations="+", spin=1)
    c_spin2 = chden(DATA_DIR2, (1, 0, 0), 0.0, operations="+", spin=2)
    assert sorted(c_spin1.filtered_dens_files) == sorted(c_spin2.filtered_dens_files)


def test_chden_custom_charge_stub() -> None:
    c = chden(DATA_DIR2, (1, 0, 0), 0.0, operations="+", charge_stub="custom", spin=1)
    names = {f.split("/")[-1] for f in c.filtered_dens_files}
    assert names == {"custom_up.cube", "custom_dn.cube"}


def test_chden_add_up_and_dn() -> None:
    c = chden(DATA_DIR2, (1, 0, 0), 0.0, operations="+", spin=1)
    up = next(d for d, _ in c.density.densities if True)
    assert c.density.data.shape == up.shape


def test_chden_rejects_missing_directory() -> None:
    with pytest.raises(FileNotFoundError):
        chden("tests/data/chargedensity_missing", (1, 0, 0), 0.0, operations="")


def test_chden_rejects_zero_miller_indices() -> None:
    with pytest.raises(ValueError):
        chden(DATA_DIR2, (0, 0, 0), 0.0, operations="")


def test_chden_raises_when_no_matching_stub_found() -> None:
    with pytest.raises(ValueError, match="At least one"):
        chden(DATA_DIR2, (1, 0, 0), 0.0, operations="", charge_stub="doesnotexist")


def test_chden_locate_is_idempotent() -> None:
    c = chden(DATA_DIR2, (1, 0, 0), 0.0, operations="")
    first = list(c.all_dens_files)
    c.locate_chden_files()
    assert c.all_dens_files == first
