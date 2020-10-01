from physunits import um, s
pe = 1.0

FLEA3 = {
    'sensor':'CCD-ICX618'
    'model':'FL3-FW-03S1C/M-C'
    'quantum_efficiency':0.3, 
    'max_well_depth': 23035.06*pe,
    'pixel_size': 5.6*um,
    'read_noise': 38.74*pe,
    'dark_current': 307.92*pe/s,
    'notes':'QE@780nm'
    }

ZYLA_5p5 = {
    'quantum_efficiency': 0.35,
    'max_well_depth': 30000*pe,
    'pixel_size': 6.5*um,
    'read_noise': 2.7*pe,
    'dark_current': 0.01*pe/s,
}

PRIME95B = {
    'quantum_efficiency': 0.5,
    'max_well_depth': 80000*pe,
    'pixel_size': 11*um,
    'read_noise': 1.8*pe,
    'dark_current': 0.1*pe/s,
}