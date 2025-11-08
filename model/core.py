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

from typing import Literal, TypeAlias
from collections.abc import Sequence


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
n_gold = interp1d(_gold_wavelen*10**-6, _n_gold, kind='cubic')

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
defaults = {
    "aberrations": False,
    "n_medium": n_water,
    "n_glass": n_glass,
    "n_glass0": n_glass,
    "n_oil0": n_oil,
    "n_oil": n_oil,
    "t_oil0": 100e-6,       # micron
    "t_oil": 100e-6,       # micron
    "t_glass0": 170e-6,    # micron
    "t_glass": 170e-6,     # micron
    "diameter": 30e-9,     # nm
    "z_p": 0,              # micron
    "defocus": 0,          # um
    "wavelen": 532e-9,     # nm
    "NA": 1.4,
    "multipolar": True,
    "roi_size": 2e-6,      # micron
    "pxsize": 3.45e-6,     # micron
    "magnification": 60,
    "scat_mat": "gold",
    "x0": 0,               # micron
    "y0": 0,               # micron
    "r_resolution": 50,
    "efficiency": 1.,   # Determined through experiment
    # Angles / Polarization
    "anisotropic": False,
    "azimuth": 0,           # radians
    "inclination": 0,       # radians
    "polarized": False,
    "polarization_azimuth": 0,  # radians
    "beam_angle": 0,             # radians
    "beam_azimuth": 0            # radians
}


    

class Camera():
    """Helper class to handle coordinate conversions"""
    def __init__(self, **kwargs):
        roi_size = kwargs['roi_size']
        pxsize = kwargs['pxsize']
        magnification = kwargs['magnification']
        x0 = kwargs['x0']
        y0 = kwargs['y0']

        self.roi_size = roi_size
        self.pxsize = pxsize
        self.magnification = magnification
        self.pxsize_obj = self.pxsize/self.magnification
        pixels = int(self.roi_size//self.pxsize_obj)
        self.pixels = pixels + 1 if pixels%2 == 0 else pixels
        xs = np.linspace(-self.roi_size/2, self.roi_size/2, pixels)
        ys = np.linspace(-self.roi_size/2, self.roi_size/2, pixels)
        self.x, self.y = np.meshgrid(xs, ys)
        self.r = np.sqrt((self.x-x0)**2 + (self.y-y0)**2)
        self.phi = np.arctan2(self.y-y0, self.x-x0)


def opd(angle_oil, **kwargs):
    """
    Optical path difference between the design path of the objective (in focus, z_p = 0, RI's match design, thicknesses match design etc)
    and the actual  path where all parameters can differ.
    """
    z_p = kwargs['z_p']
    defocus = kwargs['defocus']
    aberrations = kwargs['aberrations']
    n_oil = kwargs['n_oil']
    n_oil0 = kwargs['n_oil0']
    n_glass = kwargs['n_glass']
    n_glass0 = kwargs['n_glass0']
    n_medium = kwargs['n_medium']
    t_glass = kwargs['t_glass']
    t_glass0 = kwargs['t_glass0']
    t_oil0 = kwargs['t_oil0']

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

def opd_ref(**kwargs):
    """
    Optical path difference of the reference beam from the glass-medium layer to the aperture
    It travels through glass and oil at orthogonal angle.
    """
    z_p = kwargs['z_p']
    defocus = kwargs['defocus']
    aberrations = kwargs['aberrations']
    n_oil = kwargs['n_oil']
    n_oil0 = kwargs['n_oil0']
    n_glass = kwargs['n_glass']
    n_glass0 = kwargs['n_glass0']
    n_medium = kwargs['n_medium']
    t_glass = kwargs['t_glass']
    t_glass0 = kwargs['t_glass0']
    t_oil0 = kwargs['t_oil0']

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


def B(n, angle_oil, rs, **kwargs):
    wavelen = kwargs['wavelen']
    n_oil = kwargs['n_oil']

    k_0 = -2*np.pi/wavelen
    return np.sqrt(np.cos(angle_oil))*np.sin(angle_oil)*jv(n, k_0*rs*n_oil*np.sin(angle_oil))*np.exp(1j*k_0*opd(angle_oil, **kwargs))


epsrel=1e-3
def Integral_0(rs, **kwargs):
    n_medium = kwargs['n_medium']
    n_glass = kwargs['n_glass']
    n_oil = kwargs['n_oil']
    NA = kwargs['NA']

    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))

    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_medium, angle_medium, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        t_s_1 = t_s(n_medium, angle_medium, n_glass, angle_glass)
        t_s_2 = t_s(n_glass, angle_glass, n_oil, angle_oil)
        n_eff_medium = np.sqrt(n_medium**2 - n_oil**2*np.sin(angle_oil)**2)
        E_s, E_p = calculate_scatter_field(**kwargs)(angle_medium)

        return (B(0, angle_oil, rs, **kwargs)*
            (E_s*t_s_1*t_s_2 + E_p*t_p_1*t_p_2 * n_eff_medium/n_medium))

    return quad_vec(integrand, 0, capture_angle_medium, epsrel=epsrel)[0]

def Integral_1(rs, **kwargs):
    n_medium = kwargs['n_medium']
    n_glass = kwargs['n_glass']
    n_oil = kwargs['n_oil']
    NA = kwargs['NA']

    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))

    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_glass, angle_glass, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        E_s, E_p = calculate_scatter_field(**kwargs)(angle_medium)

        return (B(1, angle_oil, rs, **kwargs)*
            E_p*t_p_1*t_p_2 * n_oil/n_glass * np.sin(angle_oil))

    return quad_vec(integrand, 0, capture_angle_medium, epsrel=epsrel)[0]

def Integral_2(rs, **kwargs):
    n_medium = kwargs['n_medium']
    n_glass = kwargs['n_glass']
    n_oil = kwargs['n_oil']
    NA = kwargs['NA']

    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))

    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_glass, angle_glass, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        t_s_1 = t_s(n_medium, angle_medium, n_glass, angle_glass)
        t_s_2 = t_s(n_glass, angle_glass, n_oil, angle_oil)
        n_eff_medium = np.sqrt(n_medium**2 - n_oil**2*np.sin(angle_oil)**2)
        E_s, E_p = calculate_scatter_field(**kwargs)(angle_medium)

        return (B(2, angle_oil, rs, **kwargs)*
            (E_s*t_s_1*t_s_2 - E_p*t_p_1*t_p_2 * n_eff_medium/n_medium))

    return quad_vec(integrand, 0, capture_angle_medium, epsrel=epsrel)[0]



#@lru_cache_args(Integral_0, Integral_1, Integral_2, B, Camera.__init__, opd)
def calculate_propagation(**kwargs):
    """
    Calculate the propagation from particle to camera.
    The mathematics is radial and the radial data is interpolated to project onto a grid
    Returns a matrix which converts an xyz polarized scatter field into an S and P component at the detector.
    [Es, Ep] = M [Ex, Ey, Ez]
    """
    wavelen = kwargs['wavelen']
    r_resolution = kwargs['r_resolution']

    camera = Camera(**kwargs)
    rs = np.linspace(0, np.max(camera.r), r_resolution)

    I_0 = interp1d(rs, Integral_0(rs, **kwargs))(camera.r)
    I_1 = interp1d(rs, Integral_1(rs, **kwargs))(camera.r)
    I_2 = interp1d(rs, Integral_2(rs, **kwargs))(camera.r)
    
    # Components for x polarization
    e_xx = I_0 + I_2*np.cos(2*camera.phi)
    e_yx = I_2*np.sin(2*camera.phi)
    e_zx = -2j*I_1*np.cos(camera.phi)
    # Components for y polarization
    e_xy = I_2*np.sin(2*camera.phi)
    e_yy = I_0 - I_2*np.cos(2*camera.phi)
    e_zy = -2j*I_1*np.sin(camera.phi)

    k_0 = -2*np.pi/wavelen
    return -1j*k_0*np.stack([[e_xx, e_yx, e_zx], [e_xy, e_yy, e_zy]]).transpose((2, 3, 0, 1))


def calculate_fields(**kwargs) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Propagate and project the scatter field onto the detector

    Parameters
    ----------
    scatter_field : array_like
        A polarized field to propagate and project
    p : DesignParams
        A custom class containing experimental data
    """
    wavelen = kwargs['wavelen']
    anisotropic = kwargs['anisotropic']
    azimuth = kwargs['azimuth']
    inclination = kwargs['inclination']
    beam_azimuth = kwargs['beam_azimuth']
    beam_angle = kwargs['beam_angle']
    polarization_azimuth = kwargs['polarization_azimuth']
    polarized = kwargs['polarized']
    n_medium = kwargs['n_medium']
    efficiency = kwargs['efficiency']
    

    # Relative signal strength change due to layer boundaries
    r_gm = (n_glass - n_medium)/(n_glass + n_medium)
    E_reference = r_gm

    # takes 2x3 matrix M takes polarization p and Mp gives E in p and s components.
    detector_field_components = calculate_propagation(**kwargs)

    # Average over angles if unpolarized
    if not polarized:
        polarization_azimuth = np.linspace(0, 2*np.pi, 20)

    xyz_polarization = np.array([np.cos(polarization_azimuth),
        np.sin(polarization_azimuth),
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
            np.cos(inclination)*np.cos(azimuth),
            np.cos(inclination)*np.sin(azimuth),
            np.sin(inclination)])
        if polarized:
            polarization = np.dot(polarizability_direction, xyz_polarization)*polarizability_direction
            detector_field = detector_field_components@polarizability_direction
        else:
            polarization = np.expand_dims(np.dot(polarizability_direction, xyz_polarization), 1)*polarizability_direction
            detector_field = np.einsum('ijab,kb->ijka', detector_field_components, polarization)
    

    # Apply collection efficiency modification
    detector_field *= efficiency
    
    # effect of inclination on opd
    k = -2*np.pi/wavelen

    camera = Camera(**kwargs)
    slant_opd = (camera.x*np.cos(beam_azimuth) + camera.y*np.sin(beam_azimuth))*np.sin(beam_angle)
    slant_opd = slant_opd.reshape(slant_opd.shape + (1,) * reference_field.ndim)

    reference_field = reference_field*np.exp(1j*k*slant_opd)
    detector_field /= np.exp(1j*k*opd_ref(**kwargs))
    # correct detector field for phase common with reference

    return detector_field, reference_field
    
def calculate_intensities(**kwargs) -> NDArray[np.floating]:
    """
    Propagate and project the scatter field onto the detector

    Parameters
    ----------
    polarized: bool
    """
    polarized = kwargs['polarized']

    detector_field, reference_field = calculate_fields(**kwargs)
    
    interference_contrast = 2*np.sum(np.real((detector_field*np.conj(reference_field))), axis=-1)
    scatter_contrast = np.sum(np.abs(detector_field)**2, axis=-1)

    

    # Average over all polarization angles if unpolarized
    if not polarized:
        interference_contrast = np.mean(interference_contrast, axis=-1)
        scatter_contrast = np.mean(scatter_contrast, axis=-1)

    
    return np.stack([interference_contrast, scatter_contrast])



def calculate_scatter_field_dipole(**kwargs):
    n_medium = kwargs['n_medium']
    diameter = kwargs['diameter']
    wavelen = kwargs['wavelen']
    scat_mat = kwargs['scat_mat']

    a = diameter/2
    

    # Check input
    if scat_mat not in {'gold', 'polystyrene'}:
        raise ValueError(f'Scattering material {scat_mat} not implemented. Possible values: gold, polystyrene')

    if scat_mat == 'gold':
        n_scat = n_gold(wavelen)
    else:
        n_scat = n_ps

    # Magnitude and phase of scatter field
    k = 2*np.pi*n_medium/wavelen
    e_scat = n_scat**2
    e_medium = n_medium**2
    polarizability = 4*np.pi*a**3*(e_scat-e_medium)/(e_scat + 2*e_medium)

    #scatter_field = k**2*polarizability/2/np.sqrt(np.pi)

    def scattering_amplitude(angle):
        # [1 1]/sqrt(2) gives unpolarized average, [1 cos(angle)] gives contributions of thoses components
        polarization_components = np.squeeze([1, np.cos(angle)])/np.sqrt(2)
        return k**2*polarizability/4/np.pi*polarization_components*1j
    # if (x > 0.1):
    #     print("Exceeded bounds of Rayleigh approximation")
    
    # 1j delay is not captured in polarizability. The physical origin is the scattered wave being spherical and the incoming being planar.
    # For the radius of curvature to match it needs an i phase change
    return scattering_amplitude

def calculate_scatter_field_mie(**kwargs):
    """
    Supports array inputs
    """
    n_medium = kwargs['n_medium']
    diameter = kwargs['diameter']
    wavelen = kwargs['wavelen']
    scat_mat = kwargs['scat_mat']
    polarization_angle = kwargs['polarization_azimuth']
    azimuth = kwargs['azimuth']
    a = diameter/2
    # Check input
    if scat_mat not in {'gold', 'polystyrene'}:
        raise ValueError(f'Scattering material {scat_mat} not implemented. Possible values: gold, polystyrene')

    if scat_mat == 'gold':
        n_scat = n_gold(wavelen)
    else:
        n_scat = n_ps

    # Magnitude and phase of scatter field
    # capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
    # collection_efficiency = capture_angle_medium/np.pi
    k = 2*np.pi*n_medium/wavelen
    x_mie = k*a
    m = n_scat/n_medium

    # Benchmark

    # # S1 and S2 give the scatter amplitudes parallel (S1) and perpendicular (S2) to the scattering plane (k_in and k_out)
    # # for backscattering S1 = -S2, I take S1
    # S1_0, S2_0 = mie.S1_S2(m, x_mie, 1, norm='wiscombe')
    # S1_pi, S2_pi = mie.S1_S2(m, x_mie, -1, norm='wiscombe')

    # backscatter_field = S1_pi/k

    # q_ext = 4*np.real(S1_0)/x_mie**2

    # # def integrand(angle):
    # #     mu = np.cos(angle)
    # #     S1, S2 = mie.S1_S2(m, x_mie, mu, norm='wiscombe')
    # #     F = (np.abs(S1)**2+np.abs(S2)**2)/2
    # #     return F*np.sin(angle)
    
    # def integrand(angle):
    #     mu = np.cos(angle)
    #     S1, S2 = mie.S1_S2(m, x_mie, mu, norm='wiscombe')
    #     S = np.squeeze([S1, S2])/np.sqrt(2)
    #     E = np.sqrt(np.sum(S**2))
    #     return np.abs(E)**2*np.sin(angle)
    
    # q_sca = 2*quad(integrand, 0, np.pi)[0]/x_mie**2

    # q_back = 4*np.abs(backscatter_field)**2/a**2


    # print(q_ext, q_sca, q_back)
    # # Other function for benchmark
    # qext, qsca, qback, g = np.vectorize(mie.efficiencies_mx)(m, x_mie)
    # print(qext, qsca, qback)
    # exit()


    def scattering_amplitude(angle):
        mu = np.cos(angle)
        S1, S2 = mie.S1_S2(m, x_mie, mu, norm='wiscombe')
        S = np.squeeze([S1, S2])/np.sqrt(2) #unpolarized average
        return S/k

    # In the scattered field, the absolute is the amplitude and the angle gives the scatter phase.
    # The square of the scatter field gives the backscatter cross section.
    return scattering_amplitude

#@lru_cache_args(calculate_scatter_field_mie)
def calculate_scatter_field(multipolar=True, **kwargs):
    if multipolar:
        return calculate_scatter_field_mie(**kwargs)
    
    return calculate_scatter_field_dipole(**kwargs)


microns = {'z_p', 'defocus', 't_oil0', 't_glass0', 't_oil', 't_glass', 'roi_size', 'pxsize', 'x0', 'y0'}
nanometers = {'wavelen', 'diameter'}
degrees = {'azimuth', 'inclination', 'beam_angle', 'beam_azimuth', 'polarization_azimuth'}

def create_params(**kwargs) -> dict:
    # Unit conversions
    for um in microns:
        if um in kwargs.keys():
            kwargs[um] = kwargs[um]*10**-6
    
    for nm in nanometers:
        if nm in kwargs.keys():
            kwargs[nm] = kwargs[nm]*10**-9
    
    for deg in degrees:
        if deg in kwargs.keys():
            kwargs[deg] = np.radians(kwargs[deg])
    

    return {**defaults, **kwargs}


# User convenience functions for elegant numpy usage

# backscattering
def simulate_backscattering(**kwargs):
    """Simulate Mie/dipole scattering"""
    params = create_params(**kwargs)
    return calculate_scatter_field(**params)(np.pi)

def vectorize_array(func, **kwargs):
    """Vectorization with array output"""
    # Vectorize loses type
    ret = np.vectorize(func, otypes=[np.ndarray])(**kwargs)
    
    stack = np.stack(ret)
    if stack.dtype == object:
        stack = np.array(stack.tolist())
    
    # Force the type
    #stack = np.array(stack.tolist(), dtype=dtype)
    # 1D
    if ret.shape == stack.shape:
        return stack
    return np.moveaxis(stack,ret.ndim, 0)

def simulate_center(**kwargs) -> NDArray[np.float64]:
    """Simulate only the center pixel"""
    params = create_params(**kwargs)
    # Single pixel
    params['r_resolution'] = 2
    roi_size = params['pxsize']/params['magnification']
    params['roi_size'] = roi_size

    return np.squeeze(vectorize_array(calculate_intensities, **params))


def simulate_field(**kwargs) -> NDArray[np.complex128]:
    """Simulate the field at the center pixel"""
    params = create_params(**kwargs)
    # Single pixel
    params['r_resolution'] = 2
    roi_size = params['pxsize']/params['magnification']
    params['roi_size'] = roi_size

    return np.squeeze(vectorize_array(calculate_fields, **params))

def simulate_camera(**kwargs) -> NDArray[np.complex128]:
    """Simulate an entire sensor"""
    # Unit conversions
    params = create_params(**kwargs)
    return vectorize_array(calculate_intensities, **params)


def polarization(vector: np.ndarray) -> np.ndarray:
    v = np.abs(vector)
    return v/np.linalg.norm(v,axis=-1,keepdims=True)*np.sign(v[...,0:1])

def magnitude(vector: np.ndarray) -> np.complex128:
    # dot product
    return np.einsum('...i,...i', polarization(vector), np.conj(vector))