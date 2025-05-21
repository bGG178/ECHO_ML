# File: electrode_circle_animation.py
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
import random
from shapely.geometry import Point, Polygon as ShapelyPolygon
from shapely.ops import unary_union
import json
import os
from matplotlib.animation import PillowWriter


def functiontimer(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds of function {func.__name__}")
        return result
    return wrapper

def generate_circle_points(radius, num_points):
    """Generates evenly spaced points on a circle."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    return points

def create_inward_control_point(p0, p1, center, shrink_factor=0.5):
    """Computes a control point for a bezier arc that curves inward."""
    midpoint = (p0 + p1) / 2
    direction = midpoint - center
    control = midpoint - direction * shrink_factor
    return control

def create_bezier_arc(p0, p1, control, num_points=100):
    """Creates a quadratic bezier arc from p0 to p1 using a control point."""
    t = np.linspace(0, 1, num_points)
    arc = (1 - t)[:, None] ** 2 * p0 + 1 * (1 - t)[:, None] * t[:, None] * control + t[:, None] ** 2 * p1
    return arc


def generate_sync_scan_sequence(num_electrodes, num_excitation):
    """Generates a list of electrode pair scans (avoiding adjacent ones).

    - If num_excitation == num_electrodes: full scan is returned.
    - If num_excitation == 1: only the first 11 pairs are returned.
    """
    pairs = []
    for i in range(num_electrodes):
        for j in range(i + 1, num_electrodes + i):
            j_mod = j % num_electrodes
            if j_mod > i:
                pair = ((i, (i + 1) % num_electrodes), (j_mod, (j_mod + 1) % num_electrodes))
                pairs.append(pair)
                if (num_excitation == 1) and (len(pairs) == (num_electrodes-1)):
                    return pairs
    return pairs


def create_polygon_between_arcs(arc1, arc2):
    """Forms a closed polygon between two arcs."""
    return np.vstack([arc1, arc2[::-1]])

def generate_random_shapes(num_shapes, area_radius):
    """Generates random circular obstacles in the sensing area."""
    shapes = []
    for _ in range(num_shapes):
        r = random.uniform(0.5, 1.5)
        x = random.uniform(-area_radius + r, area_radius - r)
        y = random.uniform(-area_radius + r, area_radius - r)
        circle = Point(x, y).buffer(r, resolution=50)
        shapes.append((circle, x, y, r))
    return shapes

def compute_scan_polygons(points, center, scan_pairs, num_electrodes):
    """Computes all scan polygons and their sensing values."""
    sensing_values = []
    polygons = []

    for (elec1, elec2) in scan_pairs:
        start_idx1, end_idx1 = elec1
        start_idx2, end_idx2 = elec2


        #Handles electrode 1 to final electrode.
        if (start_idx1==0) & (end_idx1==1)&(start_idx2==(num_electrodes-1))&(end_idx2==0):
            start_idx1, end_idx1, start_idx2, end_idx2 = start_idx2, end_idx2, start_idx1, end_idx1



        # Special case: adjacent electrodes
        if (start_idx2 == end_idx1) or (end_idx2 == start_idx1):
            # Arc from start_idx1 to end_idx2 only, use midpoint
            arc = create_bezier_arc(points[start_idx1], points[end_idx2],
                                    create_inward_control_point(points[start_idx1], points[end_idx2], center))
            midpoint = (points[end_idx2] + points[start_idx1]) / 2
            # Include midpoint of adjacent electrode for fullness
            midpoint_extra = points[end_idx1]  # e.g., point 2 when scanning 1 to 3
            polygon_coords = np.vstack([arc, midpoint_extra[None, :], arc[0]])

        else:
            arc1 = create_bezier_arc(points[start_idx1], points[end_idx2],
                                     create_inward_control_point(points[start_idx1], points[end_idx2], center))
            arc2 = create_bezier_arc(points[end_idx1], points[start_idx2],
                                     create_inward_control_point(points[end_idx1], points[start_idx2], center))
            polygon_coords = create_polygon_between_arcs(arc1, arc2)



        scan_poly = ShapelyPolygon(polygon_coords)
        polygons.append((polygon_coords, scan_poly))
        sensing_values.append(scan_poly)

    return polygons, sensing_values

def compute_sensing_values(scan_polygons, shapes):
    """Computes overlap ratios between each scan polygon and the union of shapes.
    If a polygon is invalid, assigns value 0.0.
    """
    union_shapes = unary_union([s[0] for s in shapes])
    values = []
    for poly in scan_polygons:
        if not poly.is_valid:
            values.append(0.0)
            continue
        intersection_area = union_shapes.intersection(poly).area
        area = poly.area if poly.area > 0 else 1
        ratio = round(intersection_area / area, 6)
        values.append(ratio)
    return values

def perform_scan_on_phantoms(phantom_list, radius=5, num_electrodes=15, num_excitation=15):
    """
    Given a list of phantom objects (x, y, r), simulate scans and return sensing values.

    Args:
        phantom_list: List of tuples like (x, y, r) for circular phantoms
        radius: Radius of the sensing region
        num_electrodes: Number of electrodes
        num_excitation: Number of emitting electrodes

    Returns:
        List of sensing values
    """

    # Convert phantom tuples to Shapely circles with resolution 50
    shapes = [(Point(x, y).buffer(r, resolution=50), x, y, r) for (x, y, r) in phantom_list]

    # Generate electrode positions
    points = generate_circle_points(radius, num_electrodes)
    center = [0, 0]

    # Generate scan pairs
    scan_pairs = generate_sync_scan_sequence(num_electrodes, num_excitation)

    # Generate scan polygons
    polygons, _ = compute_scan_polygons(points, center, scan_pairs, num_electrodes)

    # Extract just the Shapely polygons from the output
    poly_only = [p[1] for p in polygons]

    # Compute sensing values
    values = compute_sensing_values(poly_only, shapes)

    return values

@functiontimer
def main(radius=5, num_electrodes=12, num_excitation=12, fade=True, animate=False, savegif=False):
    """Main function to generate electrodes, simulate scans, and optionally animate."""
    points = generate_circle_points(radius, num_electrodes)
    center = np.array([0, 0])
    scan_pairs = generate_sync_scan_sequence(num_electrodes,num_excitation)

    # Generate shapes
    shapes = generate_random_shapes(random.randint(1, 4), radius)

    # Compute scan polygons
    polygon_data, shapely_polys = compute_scan_polygons(points, center, scan_pairs, num_electrodes)

    # Compute values
    sensing_values = compute_sensing_values(shapely_polys, shapes)

    if animate:

        # Setup plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.add_artist(plt.Circle((0, 0), radius, color='black', fill=False, lw=2))
        ax.scatter(points[:, 0], points[:, 1], color='green', zorder=5)
        for idx, (x, y) in enumerate(points):
            ax.text(x * 1.05, y * 1.05, str(idx + 1), ha='center', va='center')
        for _, x, y, r in shapes:
            ax.add_patch(Circle((x, y), r, color='gray', alpha=0.3))

        ax.set_xlim(-radius - 1, radius + 1)
        ax.set_ylim(-radius - 1, radius + 1)
        ax.set_aspect('equal')
        ax.set_title('Electrode Sync Scanning')
        plt.grid(False)

        value_text = ax.text(-radius, -radius - 0.5, '', fontsize=12, ha='left')
        lines, patches = [], []

        @functiontimer
        def update(frame):
            nonlocal lines, patches
            if fade:
                for line in lines:
                    line.set_alpha(max(line.get_alpha() - 0.1, 0))
                for patch in patches:
                    patch.set_alpha(max(patch.get_alpha() - 0.1, 0))
            else:
                for line in lines:
                    line.remove()
                for patch in patches:
                    patch.remove()
                lines.clear()
                patches.clear()

            polygon_coords, poly = polygon_data[frame]
            coverage = sensing_values[frame]

            patch = PathPatch(Path(polygon_coords, [Path.MOVETO] + [Path.LINETO] * (len(polygon_coords) - 2) + [Path.CLOSEPOLY]),
                              facecolor='cyan', edgecolor='black', lw=1, alpha=0.6)
            ax.add_patch(patch)
            lines.append(ax.plot(polygon_coords[:, 0], polygon_coords[:, 1], color='black', lw=1, alpha=0.6)[0])
            patches.append(patch)
            value_text.set_text(f'Sensing value: {coverage:.6f}')
            return lines + patches + [value_text]

    if animate:
        ani = FuncAnimation(fig, update, frames=len(scan_pairs), interval=500, blit=True, repeat=True)
        plt.show()
        if savegif:
            output_dir = r"C:\Users\welov\PycharmProjects\ECHO_ML\DATA\GrantGeneratedData\gifs"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = os.path.join(output_dir, f"scan_animation_{timestamp}_{num_electrodes}.gif")

            ani.save(gif_path, writer=PillowWriter(fps=2))
            print(f"Saved animation to {gif_path}")
    else:
        pass
        #print("Scan values:")
        #for i, val in enumerate(sensing_values):
            #print(f"Scan {i+1:03d}: {val:.6f}")


    return sensing_values, shapes #shapes structure:(circle_geometry, x, y, r)




if __name__ == "__main__":
    numberSamples = 1
    numberElectrodes = 15
    numberExcitationElectrodes = 15
    save = False #save as file. If doing large batches, turn savegif and animate false
    savegif = False #DO NOT DO THIS UNLESS YOU ARE SURE AND HAVE NUMBERSAMPLES BELOW 10
    animate = True #In order for animations for savegif to work, this must also be true
    fade = True
    # shapes structure:(circle_geometry, x, y, r)
    data = []

    for i in range(numberSamples):
        print(f"Sample {i + 1}")
        sensing_values, shapes = main(radius=5, num_electrodes=numberElectrodes,num_excitation=numberExcitationElectrodes, fade=fade, animate=animate, savegif=savegif)

        obj_list = [{"center": [round(x, 6), round(y, 6)], "radius": round(r, 6)} for (_, x, y, r) in shapes]
        data.append({
            "objects": obj_list,
            "measurements": [round(val, 6) for val in sensing_values]
        })

    if save:

        output_dir = r"C:\Users\welov\PycharmProjects\ECHO_ML\DATA\GrantGeneratedData"  # Absolute path to the desired directory
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"npg_{numberSamples}_{numberElectrodes}e{numberExcitationElectrodes}_traintest.json") #format as 'npg_{number of samples}_{number of electrodes}e{number of emitting electrodes}_{purpose(test, train, traintest)}.json'

            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(data, f)
            print(f"Data successfully saved to {output_file}")
        except Exception as e:
            print(f"Error saving file: {e}")