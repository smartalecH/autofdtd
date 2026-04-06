from __future__ import annotations

import math

import numpy as np
import pytest

from autofdtd.api import BroadbandPulse, ContinuousWave, CustomSourceTime, GaussianPulse
from autofdtd.ir import (
    BroadbandPulseIR,
    ContinuousWaveIR,
    CustomSourceTimeIR,
    GaussianPulseIR,
    source_time_to_ir,
)
from autofdtd.sources import source_time_model_from_value


def test_gaussian_pulse_validates_parameters() -> None:
    with pytest.raises(ValueError, match="freq0 must be a positive finite value"):
        GaussianPulse(freq0=0.0, fwidth=1.0)

    with pytest.raises(ValueError, match="fwidth must be a positive finite value"):
        GaussianPulse(freq0=1.0, fwidth=-1.0)

    with pytest.raises(ValueError, match="amplitude must be a non-negative finite value"):
        GaussianPulse(freq0=1.0, fwidth=1.0, amplitude=-1.0)

    with pytest.raises(ValueError, match="greater than or equal to 2.5"):
        GaussianPulse(freq0=1.0, fwidth=1.0, offset=2.0)


def test_gaussian_pulse_waveform_peaks_near_offset_time() -> None:
    pulse = GaussianPulse(freq0=2.0e14, fwidth=4.0e13, amplitude=1.5, phase=0.2)
    peak_amp = pulse.amp_time(pulse.peak_time + pulse._peak_time_shift)[0]
    early_amp = pulse.amp_time(0.0)[0]

    assert abs(peak_amp) > abs(early_amp)
    assert pulse.frequency_range() == (4.0e13, 3.6e14)
    assert pulse.end_time() > pulse.offset_time


def test_gaussian_pulse_without_dc_removal_matches_closed_form() -> None:
    pulse = GaussianPulse(
        freq0=3.0,
        fwidth=0.5,
        amplitude=2.0,
        phase=math.pi / 3.0,
        remove_dc_component=False,
    )
    times = np.array([pulse.offset_time])
    expected = (
        1j
        * np.exp(1j * pulse.phase)
        * np.exp(-1j * 2.0 * math.pi * pulse.freq0 * times)
        * pulse.amplitude
    )
    np.testing.assert_allclose(pulse.amp_time(times), expected)


def test_continuous_wave_ramps_and_persists() -> None:
    cw = ContinuousWave(freq0=1.5e14, fwidth=2.0e13, amplitude=2.5, phase=0.1)
    before = cw.amp_time(cw.offset_time - 8.0 * cw.twidth)[0]
    midpoint = cw.amp_time(cw.offset_time)[0]
    late = cw.amp_time(cw.offset_time + 8.0 * cw.twidth)[0]

    assert abs(before) < abs(midpoint) < abs(late)
    assert abs(late) == pytest.approx(cw.amplitude, rel=5.0e-4)
    assert cw.end_time() is None


def test_broadband_pulse_derives_equivalent_gaussian_subset() -> None:
    pulse = BroadbandPulse(freq_range=(1.8e14, 2.6e14), minimum_amplitude=0.25, amplitude=1.2)
    edge_response = np.abs(pulse.amp_freq(np.array([pulse.freq_range[0], pulse.freq_range[1]])))

    assert pulse.freq0 == pytest.approx(2.2e14)
    assert pulse.bandwidth == pytest.approx(8.0e13)
    assert edge_response[0] == pytest.approx(pulse.amplitude * pulse.minimum_amplitude)
    assert edge_response[1] == pytest.approx(pulse.amplitude * pulse.minimum_amplitude)
    assert pulse.end_time() > pulse.offset_time


def test_custom_source_time_interpolates_and_clamps_outside_range() -> None:
    source_time = CustomSourceTime.from_values(
        freq0=2.0,
        fwidth=0.5,
        values=np.array([0.0 + 0.0j, 1.0 + 0.5j, 0.0 + 0.0j]),
        dt=0.25,
        amplitude=2.0,
        phase=0.0,
    )

    midpoint = source_time.amp_time(source_time.offset_time + 0.125)[0]
    peak = source_time.amp_time(source_time.offset_time + 0.25)[0]
    trailing = source_time.amp_time(source_time.offset_time + 1.0)[0]

    expected_midpoint = 2.0 * np.exp(-1j * 2.0 * math.pi * 2.0 * (source_time.offset_time + 0.125)) * (
        0.5 + 0.25j
    )
    expected_peak = 2.0 * np.exp(-1j * 2.0 * math.pi * 2.0 * (source_time.offset_time + 0.25)) * (
        1.0 + 0.5j
    )

    np.testing.assert_allclose(midpoint, expected_midpoint)
    np.testing.assert_allclose(peak, expected_peak)
    np.testing.assert_allclose(trailing, 0.0 + 0.0j)
    assert source_time.end_time() == pytest.approx(source_time.offset_time + 0.25)


def test_custom_source_time_validates_sample_layout() -> None:
    with pytest.raises(ValueError, match="more than one time sample"):
        CustomSourceTime(freq0=1.0, fwidth=0.5, time_samples=(0.0,), envelope_values=((0.0, 0.0),))

    with pytest.raises(ValueError, match="strictly increasing"):
        CustomSourceTime(
            freq0=1.0,
            fwidth=0.5,
            time_samples=(0.0, 0.1, 0.1),
            envelope_values=((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)),
        )

    with pytest.raises(ValueError, match="match time_samples length"):
        CustomSourceTime(
            freq0=1.0,
            fwidth=0.5,
            time_samples=(0.0, 0.1, 0.2),
            envelope_values=((0.0, 0.0), (1.0, 0.0)),
        )


def test_source_time_model_from_value_and_ir_lowering() -> None:
    pulse = source_time_model_from_value(
        {
            "type": "GaussianPulse",
            "freq0": 2.5e14,
            "fwidth": 5.0e13,
            "amplitude": 0.75,
            "phase": 0.4,
        }
    )
    cw = source_time_model_from_value(
        {
            "type": "ContinuousWave",
            "freq0": 2.5e14,
            "fwidth": 5.0e13,
            "offset": 6.0,
        }
    )
    broadband = source_time_model_from_value(
        {
            "type": "BroadbandPulse",
            "freq_range": (2.0e14, 3.0e14),
            "minimum_amplitude": 0.2,
        }
    )
    custom = source_time_model_from_value(
        {
            "type": "CustomSourceTime",
            "freq0": 2.0e14,
            "fwidth": 5.0e13,
            "time_samples": (0.0, 1.0e-15, 2.0e-15),
            "envelope_values": ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)),
        }
    )

    pulse_ir = source_time_to_ir(pulse)
    cw_ir = source_time_to_ir(cw)
    broadband_ir = source_time_to_ir(broadband)
    custom_ir = source_time_to_ir(custom)

    assert isinstance(pulse_ir, GaussianPulseIR)
    assert pulse_ir.offset_time == pytest.approx(pulse.offset_time)
    assert pulse_ir.frequency_range == pytest.approx(pulse.frequency_range())
    assert isinstance(cw_ir, ContinuousWaveIR)
    assert cw_ir.end_time is None
    assert cw_ir.twidth == pytest.approx(cw.twidth)
    assert isinstance(broadband_ir, BroadbandPulseIR)
    assert broadband_ir.freq_range == pytest.approx(broadband.freq_range)
    assert broadband_ir.freq0 == pytest.approx(broadband.freq0)
    assert isinstance(custom_ir, CustomSourceTimeIR)
    assert custom_ir.sample_count == 3
    assert custom_ir.time_samples == pytest.approx(custom.time_samples)
