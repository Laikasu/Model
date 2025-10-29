"""
Implement functions to simulate and model an interferometric microscopy system on spherical particles.
It contains
1: scattering
2a: propagation towards objective
2b: projection from objective onto camera
"""

import numpy as np

import miepython as mie

from numpy.typing import NDArray
from scipy.special import jv
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from importlib import resources
import os
from collections.abc import Iterable

from functools import lru_cache, wraps
import inspect

from typing import Literal, Sequence


def lru_cache_args(*relevant_functions, maxsize=None):
    """
    Decorator that applies lru_cache, but only uses kwargs
    from the specified functions to form the cache key.
    This prevents expensive reruns for arguments that don't affect the result.
    """
    def decorator(func):
        relevant_kwargs = set()
        for f in relevant_functions + (func,):
            sig = inspect.signature(f)
            relevant_kwargs.update({
                name
                for name, param in sig.parameters.items()
                if param.default is not inspect._empty
                and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD})
        # Inner cached function — key is based on relevant args
        @lru_cache(maxsize=maxsize)
        def cached_call(args_key, relevant_kwargs_tuple):
            return func(*args_key, **dict(relevant_kwargs_tuple))

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Only include kwargs that were actually passed and are relevant
            relevant = tuple(sorted(
                (k, kwargs[k]) for k in relevant_kwargs if k in kwargs
            ))
            return cached_call(args, relevant)

        # Preserve cache control methods
        setattr(wrapper, "cache_clear", cached_call.cache_clear)
        setattr(wrapper, "cache_info", cached_call.cache_info)
        return wrapper
    return decorator


# Johnsonn and Christy data for gold
# https://refractiveindex.info/?shelf=main&book=Au&page=Johnson
with resources.files("model").joinpath('Johnson.csv').open("r") as file:
    gold = np.genfromtxt(file, delimiter=',', skip_header=1).T
_gold_wavelen = gold[0]
_n_gold = gold[1] - 1j*gold[2]
n_gold = interp1d(_gold_wavelen*1000, _n_gold, kind='cubic')

import scipy.constants as const

def drude_gold(wavelen):
    """
    Get the refractive index of gold according only to the Drude model.
    """
    # Johnson and Christy
    f = 1
    tau = 9*10**-15
    damping = const.hbar/tau/const.e
    res_p = 9.06
    

    freq_eV = const.h * const.c / (wavelen * 1e-9) / const.e

    drude = -f * res_p**2 / (freq_eV**2 + 1j*freq_eV*damping)
    epsilon = 1 + drude
    return np.sqrt(epsilon)

# constants
n_ps = 1.5537
n_air = 1
n_water = 1.333
n_glass = 1.499
n_oil = 1.518
n_glyc = 1.461

# default design parameters
aberrations = False
n_oil: float = n_oil
n_glass: float = n_glass
n_medium: float = n_water

t_oil: float = 100 # micron
t_glass: float = 170 # micron

diameter: float = 30 # nm
z_p: float = 0 # micron
defocus: float = 0
wavelen: float = 532 # nm
NA: float = 1.4


multipolar: bool = True


roi_size: float = 2 # micron
pxsize: float = 3.45 # micron
magnification: int = 60

scat_mat: str = "gold"

x0: float = 0
y0: float = 0
gold_model: str = 'experiment'
r_resolution: int = 50

# Angles
# Particle polarization direction
anisotropic = False
azimuth: int = 0 # degrees
inclination: int = 0 # degrees

# Excitation beam
polarized: bool = False
polarization_azimuth = 0 # degrees
beam_angle = 0 # degrees
beam_azimuth = 0 # degrees

    

class Camera():
    """Helper class to handle coordinate conversions"""
    def __init__(self, x0=x0, y0=y0, roi_size=roi_size, pxsize=pxsize, magnification=magnification,**kwargs):
        x0 = x0*10**-6
        y0 = y0*10**-6
        self.roi_size = roi_size*10**-6
        self.pxsize = pxsize*10**-6
        self.magnification = magnification
        self.pxsize_obj = self.pxsize/self.magnification
        pixels = int(self.roi_size//self.pxsize_obj)
        self.pixels = pixels + 1 if pixels%2 == 0 else pixels
        xs = np.linspace(-self.roi_size/2, self.roi_size/2, pixels)
        ys = np.linspace(-self.roi_size/2, self.roi_size/2, pixels)
        self.x, self.y = np.meshgrid(xs, ys)
        self.r = np.sqrt((self.x-x0)**2 + (self.y-y0)**2)
        self.phi = np.arctan2(self.y-y0, self.x-x0)


def opd(angle_oil, z_p=z_p, defocus=defocus, aberrations=aberrations,
        n_oil=n_oil, n_oil0=n_oil, n_glass=n_glass, n_glass0=n_glass, n_medium=n_medium,
        t_glass=t_glass, t_glass0=t_glass, t_oil0=t_oil, **kwargs):
    """
    Optical path difference between the design path of the objective (in focus, z_p = 0, RI's match design, thicknesses match design etc)
    and the actual  path where all parameters can differ.
    """

    z_p = z_p*10**-6
    defocus = defocus*10**-6
    t_glass = t_glass*10**-6
    t_glass0 = t_glass0*10**-6
    t_oil0  = t_oil0*10**-6
    

    if aberrations:
        # Full opd

        # effective RI in the z direction
        n_eff = lambda RI: np.sqrt(RI**2 - n_oil**2 *np.sin(angle_oil)**2)
        # extra phase incurred by excitation beam
        excitation_opd = n_medium*z_p

        # Phase differences in different media
        medium_opd = z_p*n_eff(n_medium)
        glass_opd = t_glass*n_eff(n_glass) - t_glass0*n_eff(n_glass0)
        
        # Practical focusing condition provides real t_oil (Gibson, 1991)
        t_oil = z_p - defocus + n_oil*(-z_p/n_medium - t_glass/n_glass + t_glass0/n_glass0 + t_oil0/n_oil0)
        n_eff_oil = n_oil*np.cos(angle_oil) # simplified

        oil_opd = t_oil*n_eff_oil - t_oil0*n_eff(n_oil0)

        return  excitation_opd + medium_opd + glass_opd + oil_opd
    else:
        # simplified for constant RI and thickness

        # extra phase incurred by excitation beam
        excitation_opd = n_medium*z_p

        Dt_oil = z_p*(1 - n_oil/n_medium) - defocus
        n_eff_oil = n_oil*np.cos(angle_oil) # simplified
        
        return n_eff_oil*Dt_oil + excitation_opd

def opd_ref(z_p=z_p, defocus=defocus, aberrations=aberrations,
        n_oil=n_oil, n_oil0=n_oil, n_glass=n_glass, n_glass0=n_glass, n_medium=n_medium,
        t_glass=t_glass, t_glass0=t_glass, t_oil0=t_oil, **kwargs):
    """
    Optical path difference of the reference beam from the glass-medium layer to the aperture
    It travels through glass and oil at orthogonal angle.
    """
    defocus = defocus*10**-6
    if aberrations:

        # Phase differences in different media
        glass_opd = t_glass*n_glass - t_glass0*n_glass0
        
        # Practical focusing condition provides real t_oil (Gibson, 1991)
        t_oil = z_p - defocus + n_oil*(-z_p/n_medium - t_glass/n_glass + t_glass0/n_glass0 + t_oil0/n_oil0)

        oil_opd = t_oil*n_oil - t_oil0*n_oil0

        return glass_opd + oil_opd
    else:
        # simplified for constant RI and thickness

        Dt_oil = z_p*(1 - n_oil/n_medium) - defocus
        
        return n_oil*Dt_oil


def snells_law(n1, angle1, n2):
    """
    Return angle2 given n1*angle1 = n2*angle2
    """
    return np.arcsin(n1 / n2 * np.sin(angle1))
    

def t_p(n1, angle1, n2, angle2):
    return 2*n1*np.cos(angle1)/(n2*np.cos(angle1) + n1*np.cos(angle2))

def t_s(n1, angle1, n2, angle2):
    return 2*n1*np.cos(angle1)/(n1*np.cos(angle1) + n2*np.cos(angle2))


def B(n, angle_oil, rs, wavelen=wavelen, n_oil=n_oil, **kwargs):
    wavelen = wavelen*10**-9

    k = -2*np.pi/wavelen
    return np.sqrt(np.cos(angle_oil))*np.sin(angle_oil)*jv(n, k*rs*n_oil*np.sin(angle_oil))*np.exp(1j*k*opd(angle_oil, **kwargs))


epsrel=1e-3
def Integral_0(rs, scatter_field, n_medium=n_medium, n_glass=n_glass, n_oil=n_oil, NA=NA, **kwargs):
    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))

    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_medium, angle_medium, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        t_s_1 = t_s(n_medium, angle_medium, n_glass, angle_glass)
        t_s_2 = t_s(n_glass, angle_glass, n_oil, angle_oil)
        n_eff_medium = np.sqrt(n_medium**2 - n_oil**2*np.sin(angle_oil)**2)

        return (B(0, angle_oil, rs, **kwargs)*
            (t_s_1*t_s_2 + t_p_1*t_p_2/n_medium*n_eff_medium))

    return quad_vec(integrand, 0, capture_angle_medium, epsrel=epsrel)[0]

def Integral_1(rs, scatter_field, n_medium=n_medium, n_glass=n_glass, n_oil=n_oil, NA=NA, **kwargs):
    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))

    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_glass, angle_glass, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)

        return (B(1, angle_oil, rs, **kwargs)*
            t_p_1*t_p_2 * n_oil/n_glass * np.sin(angle_oil))

    return quad_vec(integrand, 0, capture_angle_medium, epsrel=epsrel)[0]

def Integral_2(rs, scatter_field, n_medium=n_medium, n_glass=n_glass, n_oil=n_oil, NA=NA, **kwargs):
    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))

    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_glass, angle_glass, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        t_s_1 = t_s(n_medium, angle_medium, n_glass, angle_glass)
        t_s_2 = t_s(n_glass, angle_glass, n_oil, angle_oil)
        n_eff_medium = np.sqrt(n_medium**2 - n_oil**2*np.sin(angle_oil)**2)

        return (B(2, angle_oil, rs, **kwargs)*
            (t_s_1*t_s_2 - t_p_1*t_p_2/n_medium*n_eff_medium))

    return quad_vec(integrand, 0, capture_angle_medium, epsrel=epsrel)[0]



@lru_cache_args(Integral_0, Integral_1, Integral_2, B, Camera.__init__, opd)
def calculate_propagation(r_resolution=r_resolution, wavelen=wavelen, **kwargs):
    """
    Calculate the propagation from particle to camera.
    The mathematics is radial and the radial data is interpolated to project onto a grid
    Returns a matrix which converts an xyz polarized scatter field into an S and P component at the detector.
    [Es, Ep] = M [Ex, Ey, Ez]
    """
    camera = Camera(**kwargs)
    rs = np.linspace(0, np.max(camera.r), r_resolution)

    I_0 = interp1d(rs, Integral_0(rs, np.ndarray(1), **kwargs), kind='cubic')(camera.r)
    I_1 = interp1d(rs, Integral_1(rs, np.ndarray(1), **kwargs), kind='cubic')(camera.r)
    I_2 = interp1d(rs, Integral_2(rs, np.ndarray(1), **kwargs), kind='cubic')(camera.r)
    
    e_sx = I_0 + I_2*np.cos(2*camera.phi)
    e_sy = I_2*np.sin(2*camera.phi)
    e_sz = -2j*I_1*np.cos(camera.phi)
    e_px = I_2*np.sin(2*camera.phi)
    e_py = I_0 - I_2*np.cos(2*camera.phi)
    e_pz = -2j*I_1*np.sin(camera.phi)

    wavelen = wavelen*10**-9 # m
    k = -2*np.pi/wavelen
    return -1j*k*np.stack([[e_sx, e_sy, e_sz], [e_px, e_py, e_pz]]).transpose((2, 3, 0, 1))

def calculate_intensities(
        scatter_field,
        signal: str | Iterable[str] ='signal',
        wavelen=wavelen,
        anisotropic=anisotropic,
        azimuth=azimuth,
        inclination=inclination,
        beam_azimuth=beam_azimuth,
        beam_angle=beam_angle,
        polarization_azimuth=polarization_azimuth,
        polarized=polarized,
        **kwargs
        ) -> NDArray[np.complex128]:
    """
    Propagate and project the scatter field onto the detector

    Parameters
    ----------
    scatter_field : array_like
        A polarized field to propagate and project
    p : DesignParams
        A custom class containing experimental data
    signal : str or list of str
        Specifies the desired output.
        The string has to be one of 'interference', 'scattering', 'reference', 'signal', 'total',
        'reference_field', 'detector_field'
    """
    # Verify input
    signal = [signal] if isinstance(signal, str) else signal
    for sig in signal:
        if sig not in {'interference', 'scattering', 'reference', 'signal', 'total', 'reference_field', 'detector_field'}:
            raise ValueError(f"Invalid signal: {sig}. Must be one of 'interference', 'scattering', 'reference', 'signal', 'total', 'reference_field', 'detector_field'.")
    

    # Relative signal strength change due to layer boundaries
    r_gm = (n_glass - n_medium)/(n_glass + n_medium)
    E_reference = r_gm

    # takes 2x3 matrix M takes polarization p and Mp gives E in p and s components.
    detector_field_components = calculate_propagation(wavelen=wavelen, **kwargs)

    # Average over all angles if unpolarized
    if not polarized:
        polarization_azimuth = np.linspace(0, 180, 100)

    xyz_polarization = np.array([np.cos(np.radians(polarization_azimuth)),
        np.sin(np.radians(polarization_azimuth)),
        np.zeros_like(polarization_azimuth)], dtype=np.complex128)
                         
    
    xyz_to_sp = np.array([[1, 0, 0],
                         [0, 1, 0]])
    reference_field = (xyz_to_sp@xyz_polarization).T*E_reference

    
    
    if not anisotropic:
        polarization = xyz_polarization
        if polarized:
            detector_field = detector_field_components@polarization
        else:
            detector_field = np.einsum('ijab,bk->ijka', detector_field_components, polarization)
    else:
        polarizability_direction = np.array([
            np.cos(np.radians(inclination))*np.cos(np.radians(azimuth)),
            np.cos(np.radians(inclination))*np.sin(np.radians(azimuth)),
            np.sin(np.radians(inclination))])
        if polarized:
            polarization = np.dot(polarizability_direction, xyz_polarization)*polarizability_direction
            detector_field = detector_field_components@polarizability_direction
        else:
            polarization = np.expand_dims(np.dot(polarizability_direction, xyz_polarization), 1)*polarizability_direction
            detector_field = np.einsum('ijab,kb->ijka', detector_field_components, polarization)
    
    detector_field *= scatter_field
    
    # effect of inclination on opd
    wavelen = wavelen*10**-9 # m
    k = -2*np.pi/wavelen

    camera = Camera(**kwargs)
    slant_opd = (camera.x*np.cos(np.radians(beam_azimuth)) + camera.y*np.sin(np.radians(beam_azimuth)))*np.sin(np.radians(beam_angle))
    slant_opd = slant_opd.reshape(slant_opd.shape + (1,) * reference_field.ndim)

    reference_field = reference_field*np.exp(1j*k*slant_opd)
    # correct detector field for phase common with reference

    detector_field /= np.exp(1j*k*opd_ref(**kwargs))
    
    interference_intensity = 2*np.sum(np.real((detector_field*np.conj(reference_field))), axis=-1)
    scatter_intensity = np.sum(np.abs(detector_field)**2, axis=-1)

    

    # Average over all polarization angles if unpolarized
    if not polarized:
        interference_intensity = np.mean(interference_intensity, axis=-1)
        scatter_intensity = np.mean(scatter_intensity, axis=-1)
    

    returns = {'interference':interference_intensity,
        'scattering':scatter_intensity,
        'signal':scatter_intensity+interference_intensity,
        'reference_field':reference_field,
        'detector_field':detector_field}
    
    return np.squeeze(np.stack([np.squeeze(returns[sig]) for sig in signal]))



def calculate_scatter_field_dipole(
        n_glass=n_glass,
        n_medium=n_medium,
        n_oil=n_oil,
        diameter=diameter,
        NA=NA,
        wavelen=wavelen,
        scat_mat=scat_mat,
        efficiency: float = 1.,
        **kwargs):

    a = diameter/2*10**-9
    

    # Check input
    if scat_mat not in {'gold', 'polystyrene'}:
        raise ValueError(f'Scattering material {scat_mat} not implemented. Possible values: gold, polystyrene')

    if scat_mat == 'gold':
        n_scat = n_gold(wavelen)
    else:
        n_scat = n_ps
    
    wavelen = wavelen*10**-9

    scatter_field = 1 + 0j

    # Magnitude and phase of scatter field
    k = -2*np.pi*n_medium/wavelen
    e_scat = n_scat**2
    e_medium = n_medium**2
    polarizability = 4*np.pi*a**3*(e_scat-e_medium)/(e_scat + 2*e_medium)

    # if (x_mie > 0.1):
    #     print("Exceeded bounds of Rayleigh approximation")
    scatter_cross_section = k**4/6/np.pi *polarizability**2
    
    scatter_field *= np.sqrt(scatter_cross_section)*efficiency*1j
    return scatter_field

def calculate_scatter_field_mie(
        n_glass=n_glass,
        n_medium=n_medium,
        n_oil=n_oil,
        diameter=diameter,
        NA=NA,
        wavelen=wavelen,
        scat_mat=scat_mat,
        efficiency: float = 1.,
        **kwargs) -> np.complexfloating | NDArray[np.complexfloating]:
    
    a = diameter/2*10**-9
    # Check input
    if scat_mat not in {'gold', 'polystyrene'}:
        raise ValueError(f'Scattering material {scat_mat} not implemented. Possible values: gold, polystyrene')

    if scat_mat == 'gold':
        n_scat = n_gold(wavelen)
    else:
        n_scat = n_ps
    wavelen = wavelen*10**-9

    scatter_field = 1 + 0j

    # Magnitude and phase of scatter field
    # capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
    # collection_efficiency = capture_angle_medium/np.pi
    k = 2*np.pi*n_medium/wavelen
    x_mie = 2*np.pi*a*n_medium / wavelen
    m = n_scat/n_medium

    angle = np.pi
    mu = np.cos(angle)

    # S1 and S2 give the scatter amplitudes perpendicular (S2) and parallel (S1) to the incoming light
    S1, S2 = mie.S1_S2(m, x_mie, mu,norm='wiscombe')

    qext, qsca, qback, g = mie.efficiencies_mx(m, x_mie)
    F = (S1**2+S2**2)/2
    
    scatter_cross_section = qback*np.pi*a**2
    scatter_amplitude = np.sqrt(scatter_cross_section)

    # In the scattered, the real part is the amplitude and the angle gives the scatter phase.
    scatter_phase = np.angle(np.sqrt(F))
    scatter_field = scatter_field*scatter_amplitude*np.exp(1j*scatter_phase)*efficiency
    return scatter_field

@lru_cache_args(calculate_scatter_field_mie)
def calculate_scatter_field(multipolar=True, **kwargs):
    if multipolar:
        return calculate_scatter_field_mie(**kwargs)
    
    return calculate_scatter_field_dipole(**kwargs)


# User convenience functions for elegant numpy usage

# Scattering only for performance
def _simulate_scattering(**kwargs):
    return calculate_scatter_field(**kwargs)

simulate_scattering = np.vectorize(_simulate_scattering)


# Scattering + Propagation + Projection at the center of the camera
def _simulate_center(r_resolution=2, **kwargs):
    scatter_field = calculate_scatter_field(**kwargs)
    # 1 pixel camera
    return calculate_intensities(scatter_field, r_resolution=2, roi_size=pxsize/magnification, **kwargs)

_simulate_center_vec = np.vectorize(_simulate_center, excluded={'signal'},otypes=[np.ndarray])

SignalLiteral = Literal[
    'interference',
    'scattering',
    'signal',
    'reference_field',
    'detector_field',
]

def simulate_center(
        signal: SignalLiteral  | Sequence[SignalLiteral]='signal',
        **kwargs):
    result = _simulate_center_vec(signal=signal, **kwargs)
    return np.squeeze(np.stack(result)).astype(np.float64)

def _simulate_camera(**kwargs):
    scatter_field = calculate_scatter_field(**kwargs)
    return calculate_intensities(scatter_field, **kwargs)

_simulate_camera_vec = np.vectorize(_simulate_camera, excluded={'signal'},otypes=[np.ndarray])


def simulate_camera(
        signal: SignalLiteral  | Sequence[SignalLiteral]='signal',
        **kwargs):
    result = _simulate_camera_vec(signal=signal, **kwargs)
    return np.squeeze(np.stack(result)).astype(float)


def polarization(vector: np.ndarray) -> np.ndarray:
    v = vector.real
    return v/np.linalg.norm(v,axis=-1,keepdims=True)*np.sign(v[...,0:1])

def magnitude(vector: np.ndarray) -> np.complex128:
    # dot product
    return np.einsum('...i,...i', polarization(vector), np.conj(vector))

if __name__ == "__main__":
    simulate_camera()