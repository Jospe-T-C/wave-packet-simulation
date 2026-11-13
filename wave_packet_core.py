"""Core numerical routines for the Wave Packet Simulation app.

Natural units are used throughout: hbar = m = 1.
"""
import numpy as np

HBAR = 1.0
MASS = 1.0


def build_grid(N=1024, L=200.0):
    """Return position grid x, FFT-compatible wave-number grid k, and dx."""
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    return x, k, dx


def gaussian_wave_packet(x, x0, k0, sigma0):
    """Normalized Gaussian wave packet with spatial standard deviation sigma0."""
    psi = (1.0 / (2.0 * np.pi * sigma0**2)) ** 0.25
    psi = psi * np.exp(-(x - x0) ** 2 / (4.0 * sigma0**2) + 1j * k0 * x)
    return psi / np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))


def potential_free(x):
    return np.zeros_like(x, dtype=float)


def potential_harmonic(x, omega):
    return 0.5 * MASS * omega**2 * x**2


def potential_barrier(x, V0, barrier_width):
    return np.where(np.abs(x) <= barrier_width / 2.0, V0, 0.0)


def _normalise(psi, dx):
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi / norm if norm > 0 else psi


def split_step_evolve(psi0, V, k, dt, n_steps):
    """Evolve psi using the second-order split-step Fourier method."""
    psi = np.asarray(psi0, dtype=complex).copy()
    V = np.asarray(V, dtype=float)
    # The grid spacing is not needed explicitly for unitary FFT propagation.
    # Potential and kinetic operators for hbar=m=1.
    expV = np.exp(-0.5j * V * dt / HBAR)
    expT = np.exp(-1j * HBAR * k**2 * dt / (2.0 * MASS))

    history = np.empty((n_steps + 1, len(psi)), dtype=complex)
    history[0] = psi

    for n in range(1, n_steps + 1):
        psi = expV * psi
        psi_k = np.fft.fft(psi)
        psi_k *= expT
        psi = np.fft.ifft(psi_k)
        psi = expV * psi
        history[n] = psi

    return history


def normalisation(psi, dx):
    return float(np.sum(np.abs(psi) ** 2) * dx)


def expectation_x(psi, x, dx):
    p = np.abs(psi) ** 2
    norm = np.sum(p) * dx
    return float(np.sum(x * p) * dx / norm) if norm > 0 else 0.0


def sigma_of_t(psi, x, dx):
    p = np.abs(psi) ** 2
    norm = np.sum(p) * dx
    if norm <= 0:
        return 0.0
    mean = np.sum(x * p) * dx / norm
    mean2 = np.sum(x**2 * p) * dx / norm
    return float(np.sqrt(max(mean2 - mean**2, 0.0)))


def sigma_analytical_free(sigma0, times):
    """Free-particle Gaussian width for hbar=m=1."""
    times = np.asarray(times, dtype=float)
    return sigma0 * np.sqrt(1.0 + (times / (2.0 * sigma0**2)) ** 2)


def x_mean_analytical_free(x0, k0, times):
    """Free-particle packet centre for hbar=m=1."""
    return x0 + k0 * np.asarray(times, dtype=float)
