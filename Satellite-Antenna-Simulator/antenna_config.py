"""
Satellite Phased-Array Antenna Simulator

Configuration is based on a 16 x 16 planar phased-array antenna. 

"""

import numpy as np

# Antenna operating frequency and derived parameters

FREQUENCY_HZ = 20e9       # Representative Ka-band frequency
SPEED_OF_LIGHT = 299792458

WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY_HZ
WAVE_NUMBER = 2 * np.pi / WAVELENGTH

# Array configuration

ELEMENTS_X = 16
ELEMENTS_Y = 16

TOTAL_ELEMENTS = ELEMENTS_X * ELEMENTS_Y

# Element spacing in wavelengths
SPACING_X = 0.5
SPACING_Y = 0.5

SPACING_X_METRES = SPACING_X * WAVELENGTH
SPACING_Y_METRES = SPACING_Y * WAVELENGTH

# Beam steering angles in degrees and radians

STEERING_THETA_DEG = 30.0
STEERING_PHI_DEG = 0.0

STEERING_THETA = np.radians(STEERING_THETA_DEG)
STEERING_PHI = np.radians(STEERING_PHI_DEG)

# Element radiation model (cosine-power model)
# Simplified cosine-power element pattern, not a measured antenna-element pattern.

ELEMENT_PATTERN_EXPONENT = 1.0


# Simulation resolution for radiation pattern calculations

THETA_RESOLUTION = 181
PHI_RESOLUTION = 361

#Display floor for radiation pattern in dB (for plotting)
DB_FLOOR = -40.0

#Print the configuration parameters in a readable format
def display_configuration():

    print("=" * 55)
    print("SATELLITE PHASED-ARRAY ANTENNA CONFIGURATION")
    print("=" * 55)

    print(f"Operating frequency: {FREQUENCY_HZ / 1e9:.2f} GHz")
    print(f"Wavelength: {WAVELENGTH * 1000:.3f} mm")

    print(
        f"Array configuration: "
        f"{ELEMENTS_X} x {ELEMENTS_Y}"
    )

    print(f"Total antenna elements: {TOTAL_ELEMENTS}")

    print(
        f"Element spacing X: "
        f"{SPACING_X_METRES * 1000:.3f} mm"
    )

    print(
        f"Element spacing Y: "
        f"{SPACING_Y_METRES * 1000:.3f} mm"
    )

    print(
        f"Beam steering: "
        f"Theta={STEERING_THETA_DEG}°, "
        f"Phi={STEERING_PHI_DEG}°"
    )

    print("=" * 55)


if __name__ == "__main__":
    display_configuration()
