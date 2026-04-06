from autofdtd.planning import (
    FeatureStatus,
    feature_entry,
    feature_matrix,
    feature_status,
    planned_namespace_map,
)


def test_feature_matrix_contains_expected_families() -> None:
    entries = feature_matrix()
    assert any(entry.feature == "Simulation" for entry in entries)
    assert any(entry.feature == "PML" for entry in entries)
    assert any(entry.feature == "ModeSource" for entry in entries)
    assert any(entry.status is FeatureStatus.REJECT_CLEARLY for entry in entries)


def test_namespace_plan_covers_runtime_boundaries() -> None:
    modules = {entry.module for entry in planned_namespace_map()}
    assert "autofdtd.ir" in modules
    assert "autofdtd.compiler" in modules
    assert "autofdtd.runtime" in modules
    assert "autofdtd.kernels" in modules


def test_feature_lookup_helpers_return_expected_status() -> None:
    entry = feature_entry("PML")

    assert entry is not None
    assert entry.family == "boundaries"
    assert feature_status("MicrowaveTerminalSource") is FeatureStatus.REJECT_CLEARLY
