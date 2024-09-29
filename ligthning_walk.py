import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from WM import calculate_wm_wave_from_points

def generate_random_walk(num_points, step_size):
    points = np.zeros((num_points, 3))
    for i in range(1, num_points):
        # Generate a random direction
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)

        # Move in that direction
        points[i] = points[i - 1] + direction * step_size

    return points

# Generate 10,000 random points separated by approximately 3 meters
num_points = 10000
step_size = 3  # 3 meters between each point
points = generate_random_walk(num_points, step_size)

# Create segments from these points
segments = [(points[i], points[i + 1]) for i in range(num_points - 1)]

# Define the listener position
listener = np.mean(points, axis=0)  # Place listener at the center of the points

# Generate the WM wave
output_filename = "wm_wave_output_10000points_3m.wav"
calculate_wm_wave_from_points(segments, listener, output_filename)

# Read and print information about the generated wave
sample_rate, audio_data = wavfile.read(output_filename)
print(f"Generated WM wave from 10,000 random points separated by 3 meters:")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
print(f"Shape of audio data: {audio_data.shape}")

# Plot a subset of the points to visualize
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot first 1000 points for visualization
subset = points[:1000]
ax.plot(subset[:, 0], subset[:, 1], subset[:, 2], 'b-', alpha=0.5)
ax.scatter(subset[0, 0], subset[0, 1], subset[0, 2], c='g', s=50, label='Start')
ax.scatter(subset[-1, 0], subset[-1, 1], subset[-1, 2], c='r', s=50, label='End')
ax.scatter(listener[0], listener[1], listener[2], c='y', s=100, label='Listener')

ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title('Subset of Random Points (first 1000)')
ax.legend()

plt.savefig('random_points_3m_subset.png')
print("Subset of points visualized and saved as 'random_points_3m_subset.png'")

# Print some statistics about the points
distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
print(f"Average distance between points: {np.mean(distances):.2f} meters")
print(f"Standard deviation of distances: {np.std(distances):.2f} meters")
print(f"Total path length: {np.sum(distances):.2f} meters")