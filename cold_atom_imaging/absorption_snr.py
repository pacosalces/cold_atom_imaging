import numpy as np
np.warnings.filterwarnings('ignore')
from matplotlib import rc
rc('font', family='serif')
rc('font', size=18)
rc('text', usetex=True)
import matplotlib.pyplot as plt

# SI units everywhere
Hz, kHz, MHz, GHz = 1., 1e3, 1e6, 1e9
s, ms, us, ns = 1., 1e-3, 1e-6, 1e-9
m, cm, mm, um, nm = 1., 1e-2, 1e-3, 1e-6, 1e-9
W, mW, uW = 1., 1e-3, 1e-6
pe = 1. # photo electrons

from scipy.constants import codata
hbar = codata.value('Planck constant over 2 pi')
c = codata.value('speed of light in vacuum')
pi = codata.pi

DEBUG = True                                                                                        

def OD_opt_funct(x, s, y):
    x = np.abs(x) # Always positive
    return np.log(x) - np.log(s) - (s)*(1-x) + y

def SNR_calculator(target_OD, pulse_duration, CCD, s0, technical_noise=True):
    """ Calculates the signal-to-noise ratio for the optical depth (OD) as a
    function of the intensity in units of Isat. As input, it uses the target
    OD, the duration of the resonant probe pulse in seconds, a CCD dictionary
    containing the sensor specs, the saturation parameter s0=I/Isat, and a 
    boolean to include technical noise contributions in the calculation. """
    
    QE = CCD['Quantum Efficiency']
    eff_pix_size = CCD['Pixel Size']
    eff_pixel_area = (eff_pix_size / magnification) ** 2
    single_photon_energy = hbar * 2 * pi * frequency
    
    # Nx denotes numbers of photoelectrons.
    # From here on, assume photo(electro)n shot noise for any measured Nx
    N_sat = (QE * eff_pixel_area * pulse_duration * Isat) / (single_photon_energy)
    N_probe = N_sat * s0
    if technical_noise:
        N_read = CCD['Read Noise']
        N_dark = CCD['Dark Current'] * pulse_duration
    else:
        N_read, N_dark = 0.0, 0.0
    if DEBUG:
        print(fR"Computing SNR for a target OD of {target_OD:.2f}")
        print(fR'The saturation intensity corresponds to {N_sat:.1f} p.e.')
        print(fR"Note that this detector saturates at {CCD['Max Well Depth']:.1f} p.e., or {CCD['Max Well Depth']/N_sat:.2f} Isat")
    
    # For the number of detected absorbed photoelectrons, solve the
    # transcendental equation ln(x/s) - s*(1-x) + y = 0
    # where x = Na/Nsat, s = Np/Nsat, and y = OD. This avoids
    # any s << 1 assumptions.
    from scipy.optimize import minimize
    
    N_atoms = []
    for s in N_probe / N_sat:
        N_atoms_solver = minimize(OD_opt_funct, x0=10, args=(s, target_OD))
        N_atoms.append(np.abs(N_atoms_solver.x[0]) * N_sat)
    
    N_atoms = np.array(N_atoms)
    if DEBUG:
        # Compare with naive Beer's law (unsaturated)
        N_beer = N_probe * np.exp(-target_OD)  # Beer's law
        print(
            fR"The integrated absorbed number is {N_atoms.sum():.1f} p.e., "
            + fR"or {N_atoms.sum()/N_beer.sum():.2f} the number from Beer's law"
        )

    # Shot noise contributions
    sigma_Na = np.sqrt(N_atoms + N_read**2 + N_dark**2)
    sigma_Np = np.sqrt(N_probe + N_read**2 + N_dark**2)

    # Analytic uncorrelated error propagation
    sqpartial_Na = (1 + N_atoms / N_sat) ** 2 / (N_atoms ** 2)
    sqpartial_Np = (1 + N_probe / N_sat) ** 2 / (N_probe ** 2)
    
    sigma_OD = np.sqrt(sqpartial_Na * sigma_Na ** 2 + sqpartial_Np * sigma_Np ** 2)
    
    # Estimated SNR
    signal_to_noise = target_OD / sigma_OD
    if DEBUG:
        print(
            fR"The peak SNR is {signal_to_noise.max():.2f}, at ~ {s0[signal_to_noise.argmax()]:.2f} Isat"
        )

    # "Chop" SNR from sensor saturation point onward
    saturation_mask = np.zeros_like(s0)
    saturation_mask = s0 < CCD['Max Well Depth']/N_sat
    return signal_to_noise * saturation_mask

if __name__ in '__main__':

    # Imaging characteristics
    Isat = 1.67 * mW / (cm ** 2)
    magnification = 6.0
    wavelength = 780.24 * nm
    wavenumber = 2 * pi / wavelength
    frequency = c / wavelength
    sigma_0 = (3 * wavelength ** 2) / (2 * np.pi)

    # Sensor specs
    FLEA3 = {
        'Quantum Efficiency': 0.3,
        'Max Well Depth': 23035.06 * pe,
        'Pixel Size': 5.6 * um,
        'Read Noise': 38.74 * pe,
        'Dark Current': 307.92 * pe / s,
    }
                                                 
    ZYLA_5p5 = {
        'Quantum Efficiency': 0.35,
        'Max Well Depth': 30000 * pe,
        'Pixel Size': 6.5 * um,
        'Read Noise': 2.7 * pe,
        'Dark Current': 0.01 * pe / s,
    }
                                                 
    PRIME95B = {
        'Quantum Efficiency': 0.5,
        'Max Well Depth': 80000 * pe,
        'Pixel Size': 11 * um,
        'Read Noise': 1.8 * pe,
        'Dark Current': 0.1 * pe / s,
    }

    # Avoid using zero intensity to avoid numerical overflow
    intensities = np.logspace(-2, 2, 2**10)
    target_linear_density = 10/um
    pulse_length = 20*us
    CCD = FLEA3

    # Optical depth at the object plane
    OD = target_linear_density * sigma_0 / CCD['Pixel Size']
    signaltonoise = SNR_calculator(OD, pulse_length, CCD, intensities)
    fig = plt.figure(num=1)
    plt.plot(
        intensities[signaltonoise > 0], signaltonoise[signaltonoise > 0], c='crimson'
    )
    plt.xlabel(r'$I/I_{sat}$', fontsize=16)
    plt.ylabel(r'OD Signal To Noise Ratio', fontsize=14)
    plt.xscale('log')
    plt.xlim(np.amin(intensities), np.amax(intensities))
    plt.tight_layout()
    plt.show()
