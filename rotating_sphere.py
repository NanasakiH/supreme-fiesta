import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create sphere coordinates
phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2*np.pi:100j]
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Base color (teal)
base_color = np.array([0.0, 0.5, 0.5])

# Fixed light direction from upper-left (normalized)
light_dir = np.array([-1.0, 1.0, 1.0])
light_dir /= np.linalg.norm(light_dir)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])

# Hide the axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])

# Compute normals
normals = np.stack((x, y, z), axis=-1)

# Initial shading
dot = np.einsum('ijk,k->ij', normals, light_dir)
intensity = np.clip(dot, 0, 1)
colors = base_color * intensity[..., None]

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, linewidth=0, antialiased=False, shade=False)

# Animation function to rotate sphere around z-axis

def update(frame):
    angle = np.radians(frame)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y
    normals_rot = np.stack((x_rot, y_rot, z), axis=-1)
    dot = np.einsum('ijk,k->ij', normals_rot, light_dir)
    intensity = np.clip(dot, 0, 1)
    colors = base_color * intensity[..., None]
    surf.set_facecolors(colors)
    surf.remove()
    return ax.plot_surface(x_rot, y_rot, z, rstride=1, cstride=1,
                           facecolors=colors, linewidth=0,
                           antialiased=False, shade=False)

ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 120), interval=50)

plt.show()
