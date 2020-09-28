import numpy as np
import scipy.constants as const
from scipy.optimize import minimize
from physunits import *

pe = 1. # photo electrons

hbar = const.hbar
c = const.c
pi = np.pi

def absorption_SNR(probe_pulse, sample, detector, technical_noise=True):
    """ Calculates the signal-to-noise ratio for the optical depth (OD) as a
    function of the intensity. As input, it uses the target OD, the duration 
    of the resonant probe pulse in seconds, a CCD dictionary containing the 
    sensor specs, the saturation parameter s0=I/Isat, and a boolean to include 
    technical noise contributions in the calculation. """
        
    if technical_noise:
        N_read, N_dark = detector.read_noise, detector.dark_current * probe_pulse.duration
    else:
        N_read, N_dark = 0.0*pe, 0.0*pe
    print(fR"Computing SNR for a target OD of {sample.target_OD:.2f}")

    # Dimensionless intensity
    s0 = probe_pulse.intensity/sample.Isat
    
    # How many photoelectrons per Isat per pixel?
    N_sat = detector.quantum_efficiency * (sample.Isat * detector.eff_pixel_area * probe_pulse.duration / probe_pulse.single_photon_energy)
    print(fR'The saturation intensity corresponds to {N_sat:.1f} p.e.')
    
    # How many photoelectrons per probe pulse per pixel?
    N_probe = detector.quantum_efficiency * probe_pulse.photon_flux * detector.eff_pixel_area
    print(fR'The detector sees a probe ranging {(N_probe/N_sat).min():.2f} to {(N_probe/N_sat).max():.2f} Isat')

    # Fun fact
    print(fR"Your detector will saturate at {detector.max_well_depth:.1f} p.e., or {detector.max_well_depth/N_sat:.2f} Isat")
    
    # For the number of detected absorbed photoelectrons, solve the
    # transcendental equation ln(x/s) - s*(1-x) + y = 0
    # where x = Na/Nsat, s = Np/Nsat, and y = OD. This avoids
    # any s << 1 assumptions.
    def OD_opt_funct(x, s, y):
        x = np.abs(x) # Always positive
        return np.log(x) - np.log(s) - (s)*(1-x) + y
    N_atoms = np.array([np.abs(minimize(OD_opt_funct, x0=10, args=(s, sample.target_OD)).x[0] * N_sat) for s in N_probe/N_sat])

    # Compare with naive Beer's law (unsaturated)
    N_beer = N_probe * np.exp(-sample.target_OD)  # Beer's law
    print(fR"The integrated absorbed number is {N_atoms.sum():.1f} p.e., or {N_atoms.sum()/N_beer.sum():.2f} the number from Beer's law")

    # Shot noise contributions
    sigma_Na = np.sqrt(N_atoms + N_read**2 + N_dark**2)
    sigma_Np = np.sqrt(N_probe + N_read**2 + N_dark**2)

    # Analytic uncorrelated error propagation
    sqpartial_Na = (1 + N_atoms / N_sat) ** 2 / (N_atoms ** 2)
    sqpartial_Np = (1 + N_probe / N_sat) ** 2 / (N_probe ** 2)
    
    # Uncertainty in OD
    sigma_OD = np.sqrt(sqpartial_Na * sigma_Na ** 2 + sqpartial_Np * sigma_Np ** 2)
    
    # Estimated SNR
    signal_to_noise = sample.target_OD / sigma_OD
    print(fR"The peak SNR is {signal_to_noise.max():.2f}, at ~ {s0[signal_to_noise.argmax()]:.2f} Isat")

    # "Chop" SNR from sensor saturation point onward
    saturation_mask = np.zeros_like(s0)
    saturation_mask = s0 < detector.max_well_depth / N_sat
    return signal_to_noise * saturation_mask

class light_pulse():

    def __init__(self, wavelength, intensity, duration, **kwargs):
        self.wavelength = wavelength
        self.intensity = intensity
        self.duration = duration
        self.__dict__.update(kwargs)
        
        # Derived quantities
        self.single_photon_energy = hbar * 2 * pi * c / self.wavelength
        self.irradiance = self.intensity * self.duration
        self.photon_flux = self.irradiance / self.single_photon_energy

class two_level_scatterer():
    
    def __init__(self, resonant_wavelength, Isat, **kwargs):
        self.resonant_wavelength = resonant_wavelength
        self.Isat = Isat
        self.__dict__.update(kwargs)
        
        # Derived quantities
        self.sigma_0 = (3 * self.resonant_wavelength ** 2) / (2 * pi)

        
class camera():
    
    def __init__(self, pixel_size, quantum_efficiency, max_well_depth, read_noise, dark_current, *args, **kwargs):
        self.pixel_size = pixel_size
        self.quantum_efficiency = quantum_efficiency
        self.max_well_depth = max_well_depth
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.__dict__.update(kwargs)
        
        # Derived quantities
        self.pixel_area = self.pixel_size ** 2
        if 'magnification' in kwargs.keys():
            self.eff_pixel_size = self.pixel_size / self.magnification
        else:
            self.eff_pixel_size = self.pixel_size
            
        self.eff_pixel_area = self.eff_pixel_size ** 2 
