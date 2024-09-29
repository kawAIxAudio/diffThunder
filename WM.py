import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

SAMPLE_RATE = 44100
SECONDS = 20
NUM_SAMPLES = SAMPLE_RATE * SECONDS

# Constants
c = 343  # speed of sound (m/s)
r = 5000  # distance to listener in meters
l = 3  # length of segment (in meters)
T = 0.005  # duration of wave (usually 5ms)
A = 1  # Arbitrary scale factor

#r0 = 1000

def calculate_wm(t,
                 theta,
                 distance_to_listener,
                 distance_listener_to_original_point,
                 wave_duration):
    tau = ((c * t) - distance_to_listener) / l
    psi = (c * wave_duration) / l
    B = (A * l ** 2) / (2 * distance_to_listener * c * wave_duration)
    sin_theta = math.sin(math.radians(theta))

    # Atmospheric refraction
    a = 10
    k = math.log(distance_listener_to_original_point / a) / math.log(distance_to_listener / a)
    psi = (k ** (1/2)) * psi
    B = k * B

    def safe_div(x, y):
        epsilon = 1e-7
        if y == 0:
            return x / epsilon
        else:
            return x / y

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

def calculate_wm_from_3Dsegment(t,
                                point_1: np.array, point_2: np.array,
                                distance_listener_to_original_point,
                                wave_duration,
                                listener: np.array = None):

    if point_1.shape != point_2.shape:
        raise ValueError("Point should have the same number of dimensions.")

    def calculate_distance(first_point, second_point):
        # Calculate the distance from the midpoint to the listener
        return np.linalg.norm(first_point - second_point)
    def calculate_angle(vector_1, vector_2):
        # Normalize the vectors for angle calculation
        norm_vector_1 = np.linalg.norm(vector_1)
        norm_vector_2 = np.linalg.norm(vector_2)

        if norm_vector_1 == 0 or norm_vector_2 == 0:
            raise ValueError("Vectors must not be zero length.")

        dot_product = np.dot(vector_1, vector_2)
        angle = np.arccos(dot_product / (norm_vector_1 * norm_vector_2))

        return angle

    # Vector representing the line segment from point_1 to point_2
    segment_vector = point_2 - point_1

    # Calculate the midpoint of the segment
    midpoint = (point_1 + point_2) / 2

    # Vector from the midpoint to the listener
    midpoint_to_listener = listener - midpoint

    # Use the internal functions to calculate distance and angle
    distance = calculate_distance(midpoint, listener)
    angle = calculate_angle(midpoint_to_listener, segment_vector)

    return calculate_wm(t, angle, distance, distance_listener_to_original_point, wave_duration)

def calculate_wm_wave_from_points(points, listener, output_filename):

    if listener is None:
        listener = np.zeros(shape=points[0].shape)

    t_values = np.linspace(0, SECONDS, NUM_SAMPLES)
    combined_wm_values = np.zeros(NUM_SAMPLES)

    r0 = np.linalg.norm(points[0] - listener)

    # Iterate over each pair of points (segments)
    for point_1, point_2 in points:
        wm_values = np.array([calculate_wm_from_3Dsegment(t_values, point_1, point_2, r0, T, listener)])
        combined_wm_values += wm_values

    # Normalize the combined wave
    max_amplitude = np.max(np.abs(combined_wm_values))
    normalized_wm_values = combined_wm_values / max_amplitude

    # Convert to 16-bit PCM
    audio_data = (normalized_wm_values * 32767).astype(np.int16)

    # Export as .wav file
    wavfile.write(output_filename, SAMPLE_RATE, audio_data)
    print(f"Combined WM wave has been generated and saved as '{output_filename}'.")
