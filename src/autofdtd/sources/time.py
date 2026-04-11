"""Source-time profile models for Phase 1 source families."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import Field, field_validator

from autofdtd.core.models import TaggedModel

DEFAULT_SIGMA = 4.0
END_TIME_FACTOR_GAUSSIAN = 10.0


def _finite_float(value: float, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _positive_finite(value: float, *, field_name: str) -> float:
    normalized = _finite_float(value, field_name=field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite value")
    return normalized


def _non_negative_finite(value: float, *, field_name: str) -> float:
    normalized = _finite_float(value, field_name=field_name)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be a non-negative finite value")
    return normalized


def _normalize_times(time: float | Sequence[float]) -> np.ndarray:
    values = np.asarray(time, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError("time samples must be a scalar or one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("time samples must be finite")
    return values


def _normalize_freqs(freq: float | Sequence[float]) -> np.ndarray:
    values = np.asarray(freq, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError("frequency samples must be a scalar or one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("frequency samples must be finite")
    return values


class SourceTime(TaggedModel):
    """Base class for time-domain source envelopes."""

    type: str = "SourceTime"
    amplitude: float = 1.0
    phase: float = 0.0

    @field_validator("amplitude")
    @classmethod
    def _validate_amplitude(cls, value: float) -> float:
        return _non_negative_finite(value, field_name="amplitude")

    @field_validator("phase")
    @classmethod
    def _validate_phase(cls, value: float) -> float:
        return _finite_float(value, field_name="phase")

    def amp_complex(self) -> complex:
        """Return the complex-valued scalar amplitude."""

        return self.amplitude * complex(math.cos(self.phase), math.sin(self.phase))

    def amp_time(self, time: float | Sequence[float]) -> np.ndarray:
        """Evaluate the complex source amplitude at one or more times."""

        raise NotImplementedError

    def end_time(self) -> float | None:
        """Return the time after which the source is effectively inactive."""

        raise NotImplementedError


class Pulse(SourceTime):
    """Carrier-modulated source-time profile with finite ramp bandwidth."""

    type: str = "Pulse"
    freq0: float
    fwidth: float
    offset: float = Field(default=5.0, ge=2.5)

    @field_validator("freq0")
    @classmethod
    def _validate_freq0(cls, value: float) -> float:
        return _positive_finite(value, field_name="freq0")

    @field_validator("fwidth")
    @classmethod
    def _validate_fwidth(cls, value: float) -> float:
        return _positive_finite(value, field_name="fwidth")

    @property
    def twidth(self) -> float:
        """Envelope width in seconds."""

        return 1.0 / (2.0 * math.pi * self.fwidth)

    @property
    def offset_time(self) -> float:
        """Envelope delay in seconds."""

        return self.offset * self.twidth

    def frequency_range(self, num_fwidth: float = DEFAULT_SIGMA) -> tuple[float, float]:
        """Return the carrier frequency range covered by this profile."""

        sigma = _positive_finite(num_fwidth, field_name="num_fwidth")
        return (max(0.0, self.freq0 - sigma * self.fwidth), self.freq0 + sigma * self.fwidth)


class GaussianPulse(Pulse):
    """Gaussian-envelope source-time profile."""

    type: Literal["GaussianPulse"] = "GaussianPulse"
    remove_dc_component: bool = True

    @property
    def peak_time(self) -> float:
        return self.offset * self.twidth

    @property
    def peak_frequency(self) -> float:
        if not self.remove_dc_component:
            return self.freq0
        return 0.5 * (self.freq0 + math.sqrt(self.freq0**2 + 4.0 * self.fwidth**2))

    @property
    def _peak_time_shift(self) -> float:
        if self.remove_dc_component and self.fwidth > self.freq0:
            return self.twidth * math.sqrt(1.0 - self.freq0**2 / self.fwidth**2)
        return 0.0

    @property
    def offset_time(self) -> float:
        return self.peak_time + self._peak_time_shift

    def amp_time(self, time: float | Sequence[float]) -> np.ndarray:
        times = _normalize_times(time)
        omega0 = 2.0 * math.pi * self.freq0
        time_shifted = times - self.offset_time
        base = (
            np.exp(1j * self.phase)
            * np.exp(-1j * omega0 * times)
            * np.exp(-(time_shifted**2) / (2.0 * self.twidth**2))
            * self.amplitude
        )
        if self.remove_dc_component:
            corrected = base * (1j * omega0 + time_shifted / (self.twidth**2))
            return corrected / (2.0 * math.pi * self.peak_frequency)
        return 1j * base

    def end_time(self) -> float:
        end_time = self.offset_time + END_TIME_FACTOR_GAUSSIAN * self.twidth
        if self.remove_dc_component and self.fwidth > self.freq0:
            end_time += 2.0 * self._peak_time_shift
        return end_time

    def amp_freq(self, freq: float | Sequence[float]) -> np.ndarray:
        """Analytical Fourier transform of the GaussianPulse time profile.

        For a Gaussian-modulated cosine pulse:
        J(t) = exp(-(t-t0)^2 / (2*sigma^2)) * exp(i*omega0*t)
        FT(f) = sigma * sqrt(2*pi) * exp(-2*(pi*sigma*(f-f0))^2) * exp(i*2*pi*f*t0)

        Reference: meep sources.cpp gaussian_src_time::fourier_transform
        """
        freqs = _normalize_freqs(freq)
        omega0 = 2.0 * math.pi * self.freq0
        omega = 2.0 * math.pi * freqs
        delta = (omega - omega0) * self.twidth
        # sigma * sqrt(2*pi) * exp(-0.5 * delta^2) * exp(i * omega * peak_time)
        ft = (
            self.twidth
            * math.sqrt(2.0 * math.pi)
            * np.exp(-0.5 * delta * delta)
            * np.exp(1j * omega * self.offset_time)
            * self.amplitude
        )
        return ft


class ContinuousWave(Pulse):
    """Logistic-ramp continuous-wave source-time profile."""

    type: Literal["ContinuousWave"] = "ContinuousWave"

    def amp_time(self, time: float | Sequence[float]) -> np.ndarray:
        times = _normalize_times(time)
        omega0 = 2.0 * math.pi * self.freq0
        time_shifted = times - self.offset_time
        ramp = 1.0 / (1.0 + np.exp(-time_shifted / self.twidth))
        return (
            np.exp(1j * self.phase)
            * np.exp(-1j * omega0 * times)
            * ramp
            * self.amplitude
        )

    def end_time(self) -> None:
        return None

    def amp_freq(self, freq: float | Sequence[float]) -> np.ndarray:
        """Analytical Fourier transform of the ContinuousWave time profile.

        For a CW with logistic ramp envelope:
        FT(f) = sum over timesteps of ramp(t) * exp(-i*omega*t)
        Using the identity integral of logistic-sine convolution gives:
        FT(f) ~ pi * delta(omega - omega0) + i * omega * pi / (omega - omega0) * ...
        For practical purposes, we return a delta-like response at freq0.

        Reference: meep sources.cpp
        """
        freqs = _normalize_freqs(freq)
        omega0 = 2.0 * math.pi * self.freq0
        omega = 2.0 * math.pi * freqs
        # CW produces a response peaked at the carrier frequency
        # The logistic ramp produces a broad spectrum centered at freq0
        delta = (omega - omega0) * self.twidth
        spectrum = np.exp(-0.5 * delta * delta) * self.amplitude
        return spectrum


def _complex_pairs_to_array(values: Sequence[Sequence[float]]) -> np.ndarray:
    pairs = np.asarray(values, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("complex sample values must be a sequence of (real, imag) pairs")
    if not np.all(np.isfinite(pairs)):
        raise ValueError("complex sample values must be finite")
    return pairs[:, 0] + 1j * pairs[:, 1]


def _complex_array_to_pairs(values: Sequence[complex]) -> tuple[tuple[float, float], ...]:
    array = np.asarray(values, dtype=complex)
    if array.ndim != 1:
        raise ValueError("complex sample values must be one-dimensional")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError("complex sample values must be finite")
    return tuple((float(value.real), float(value.imag)) for value in array)


class CustomSourceTime(Pulse):
    """Interpolated custom envelope modulated by a harmonic carrier."""

    type: Literal["CustomSourceTime"] = "CustomSourceTime"
    offset: float = 0.0
    time_samples: tuple[float, ...]
    envelope_values: tuple[tuple[float, float], ...]

    @field_validator("time_samples")
    @classmethod
    def _validate_time_samples(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) <= 1:
            raise ValueError("CustomSourceTime requires more than one time sample")
        normalized = tuple(_finite_float(item, field_name="time_samples") for item in value)
        if any(later <= earlier for earlier, later in zip(normalized, normalized[1:], strict=False)):
            raise ValueError("CustomSourceTime time samples must be strictly increasing")
        return normalized

    @field_validator("envelope_values")
    @classmethod
    def _validate_envelope_values(
        cls,
        value: tuple[tuple[float, float], ...],
        info,
    ) -> tuple[tuple[float, float], ...]:
        _complex_pairs_to_array(value)
        time_samples = info.data.get("time_samples")
        if time_samples is not None and len(value) != len(time_samples):
            raise ValueError("CustomSourceTime envelope_values must match time_samples length")
        return value

    @classmethod
    def from_values(
        cls,
        *,
        freq0: float,
        fwidth: float,
        values: Sequence[complex],
        dt: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
        offset: float = 0.0,
    ) -> "CustomSourceTime":
        sample_dt = _positive_finite(dt, field_name="dt")
        sample_values = np.asarray(values, dtype=complex)
        if sample_values.ndim != 1:
            raise ValueError("values must be a one-dimensional sequence")
        times = tuple(float(index * sample_dt) for index in range(sample_values.size))
        return cls(
            freq0=freq0,
            fwidth=fwidth,
            amplitude=amplitude,
            phase=phase,
            offset=offset,
            time_samples=times,
            envelope_values=_complex_array_to_pairs(sample_values),
        )

    @property
    def envelope_complex(self) -> np.ndarray:
        return _complex_pairs_to_array(self.envelope_values)

    def amp_time(self, time: float | Sequence[float]) -> np.ndarray:
        times = _normalize_times(time)
        shifted = times - self.offset_time
        data_times = np.asarray(self.time_samples, dtype=float)
        data_values = self.envelope_complex
        envelope_real = np.interp(
            shifted,
            data_times,
            data_values.real,
            left=data_values.real[0],
            right=data_values.real[-1],
        )
        envelope_imag = np.interp(
            shifted,
            data_times,
            data_values.imag,
            left=data_values.imag[0],
            right=data_values.imag[-1],
        )
        envelope = envelope_real + 1j * envelope_imag
        omega0 = 2.0 * math.pi * self.freq0
        modulation = np.exp(1j * self.phase) * np.exp(-1j * omega0 * times)
        return self.amplitude * modulation * envelope

    def end_time(self) -> float | None:
        magnitudes = np.abs(self.envelope_complex)
        non_zero = np.flatnonzero(~np.isclose(magnitudes, 0.0))
        if non_zero.size == 0:
            return None
        return self.offset_time + self.time_samples[int(non_zero[-1])]


class BroadbandPulse(SourceTime):
    """Broadband Gaussian subset defined by a target frequency interval."""

    type: Literal["BroadbandPulse"] = "BroadbandPulse"
    freq_range: tuple[float, float]
    minimum_amplitude: float = Field(default=0.3, gt=0.05, lt=0.5)
    offset: float = 0.0

    @field_validator("freq_range")
    @classmethod
    def _validate_freq_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError("freq_range must contain exactly two elements")
        fmin = _positive_finite(value[0], field_name="freq_range[0]")
        fmax = _positive_finite(value[1], field_name="freq_range[1]")
        if fmax <= fmin:
            raise ValueError("freq_range[1] must be greater than freq_range[0]")
        return (fmin, fmax)

    @property
    def freq0(self) -> float:
        return 0.5 * (self.freq_range[0] + self.freq_range[1])

    @property
    def bandwidth(self) -> float:
        return self.freq_range[1] - self.freq_range[0]

    @property
    def fwidth(self) -> float:
        half_range = 0.5 * self.bandwidth
        return half_range / math.sqrt(2.0 * math.log(1.0 / self.minimum_amplitude))

    @property
    def twidth(self) -> float:
        return 1.0 / (2.0 * math.pi * self.fwidth)

    @property
    def automatic_delay(self) -> float:
        return 5.0 * self.twidth

    @property
    def offset_time(self) -> float:
        range_twidth = 1.0 / (2.0 * math.pi * self.bandwidth)
        return self.automatic_delay + self.offset * range_twidth

    @property
    def peak_time(self) -> float:
        return self.offset_time

    def amp_time(self, time: float | Sequence[float]) -> np.ndarray:
        times = _normalize_times(time)
        omega0 = 2.0 * math.pi * self.freq0
        shifted = times - self.offset_time
        return (
            1j
            * np.exp(1j * self.phase)
            * np.exp(-1j * omega0 * times)
            * np.exp(-(shifted**2) / (2.0 * self.twidth**2))
            * self.amplitude
        )

    def amp_freq(self, freq: float | Sequence[float]) -> np.ndarray:
        freqs = np.asarray(freq, dtype=float)
        if freqs.ndim == 0:
            freqs = freqs.reshape(1)
        if freqs.ndim != 1 or not np.all(np.isfinite(freqs)):
            raise ValueError("frequency samples must be a scalar or one-dimensional finite sequence")
        spread = np.exp(-((freqs - self.freq0) ** 2) / (2.0 * self.fwidth**2))
        return self.amplitude * np.exp(1j * self.phase) * spread

    def frequency_range_sigma(self, sigma: float = DEFAULT_SIGMA) -> tuple[float, float]:
        sigma_value = _positive_finite(sigma, field_name="sigma")
        return (
            max(0.0, self.freq0 - sigma_value * self.fwidth),
            self.freq0 + sigma_value * self.fwidth,
        )

    def frequency_range(self, num_fwidth: float = DEFAULT_SIGMA) -> tuple[float, float]:
        return self.frequency_range_sigma(sigma=num_fwidth)

    def end_time(self) -> float:
        return self.offset_time + END_TIME_FACTOR_GAUSSIAN * self.twidth


SourceTimeModel = GaussianPulse | ContinuousWave | BroadbandPulse | CustomSourceTime


def source_time_model_from_value(value: object) -> SourceTimeModel:
    """Coerce a mapping payload into a supported Phase 1 source-time profile."""

    if isinstance(value, (GaussianPulse, ContinuousWave, BroadbandPulse, CustomSourceTime)):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping or source-time model, got {type(value)!r}")

    source_time_type = str(value.get("type", ""))
    if source_time_type == "GaussianPulse":
        return GaussianPulse.model_validate(value)
    if source_time_type == "ContinuousWave":
        return ContinuousWave.model_validate(value)
    if source_time_type == "BroadbandPulse":
        return BroadbandPulse.model_validate(value)
    if source_time_type == "CustomSourceTime":
        return CustomSourceTime.model_validate(value)
    raise TypeError(f"unsupported Phase 1 source-time type {source_time_type!r}")
