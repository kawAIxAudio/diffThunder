import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Constants
c = 343  # speed of sound (m/s)
r = 5000  # distance to listener in meters
l = 3  # length of segment (in meters)
T = 0.005  # duration of wave (usually 5ms)
A = 1  # Arbitrary scale factor

def safe_div(x, y):

    epsilon = 1e-7

    if y == 0:
        return x / epsilon
    else:
        return x / y

def calculate_wm(t, theta, r = 5000, r0 = 1000):
    tau = ((c * t) - r) / l
    psi = (c * T) / l
    B = (A * l ** 2) / (2 * r * c * T)
    sin_theta = math.sin(math.radians(theta))

    # Atmospheric refraction
    a = 10
    k = math.log(r0 / a) / math.log(r / a)
    psi = (k ** (1/2)) * psi
    B = k * B

    if sin_theta < psi:
        if (-psi - sin_theta) < tau <= (-psi + sin_theta):
            return safe_div(-B * (psi ** 2 - (tau + sin_theta) ** 2), sin_theta)
        elif (-psi + sin_theta) < tau <= (psi - sin_theta):
            return -4 * B * tau
        elif (psi - sin_theta) < tau <= (psi + sin_theta):
            return safe_div(B * (psi ** 2 - (tau - sin_theta) ** 2), sin_theta)
    else:
        if (-psi - sin_theta) < tau <= (psi - sin_theta):
            return safe_div(B * (psi ** 2 - (tau + sin_theta) ** 2), sin_theta)
        elif (psi - sin_theta) < tau <= (-psi + sin_theta):
            return 0
        elif (-psi + sin_theta) < tau <= (psi + sin_theta):
            return safe_div(-B * (psi ** 2 - (tau - sin_theta) ** 2), sin_theta)

    return 0

def calculate_wm_from_3D_point(t, x, y, z):
    # gets theta and r from x y z
    # then performs calculate_wm
    pass

# Create 90 plots
t_values = np.linspace(14.5625, 14.5925, 1000)

for angle in range(91):
    wm_values = [calculate_wm(t, angle) for t in t_values]

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, wm_values)
    plt.title(f"WM vs t (Angle: {angle}°)")
    plt.xlabel("Time (t)")
    plt.ylabel("WM")
    plt.grid(True)
    #plt.savefig(f"wm_plot_angle_{angle}.png")
    #plt.show()

print("All plots have been generated and saved.")

# Create 90 WM waves at same distance, then add them up and export to .wav
SAMPLE_RATE = 44100
SECONDS = 20
NUM_SAMPLES = 44100 * SECONDS

t_values = np.linspace(0, SECONDS, NUM_SAMPLES)
combined_wm_values = np.zeros(NUM_SAMPLES)

for angle in range(91):
    wm_values =  np.array([calculate_wm(t,angle) for t in t_values])
    combined_wm_values += wm_values

# Normalize the combined wave
max_amplitude = np.max(np.abs(combined_wm_values))
normalized_wm_values = combined_wm_values / max_amplitude

# Convert to 16-bit PCM
audio_data = (normalized_wm_values * 32767).astype(np.int16)

# Export as .wav file
wavfile.write("combined_wm_wave.wav", SAMPLE_RATE, audio_data)

print("Combined WM wave has been generated and saved as 'combined_wm_wave.wav'.")