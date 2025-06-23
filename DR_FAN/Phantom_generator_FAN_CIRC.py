# File: electrode_circle_animation.py
import datetime
from shapely.geometry import Point, box

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
import random
from shapely import affinity
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
        chance = random.random()
        if 1==1: # do a circle
            shape = 'circle'
            r = random.uniform(0.5, 2)
            x = random.uniform(-area_radius + r, area_radius - r)
            y = random.uniform(-area_radius + r, area_radius - r)
            circle = Point(x, y).buffer(r, resolution=50)
            rotation = random.uniform(0, 360)
            shapes.append((shape,circle, x, y, r, 0, 0, rotation))
        """elif (chance >= 55) & (chance <55): #do a square
            shape = 'square'
            r = random.uniform(0.5, 2)
            x = random.uniform(-area_radius + r, area_radius - r)
            y = random.uniform(-area_radius + r, area_radius - r)
            square = box(x - r, y - r, x + r, y + r)
            rotation = random.uniform(0, 360)
            shapes.append((shape,square, x, y, r, 0, 0, rotation))
        if 1==1: #do on oval
            shape = 'oval'
            r = random.uniform(0.5, 1.5)
            x = random.uniform(-area_radius + r, area_radius - r)
            y = random.uniform(-area_radius + r, area_radius - r)
            circle = Point(x, y).buffer(r, resolution=50)
            xfact = random.uniform(0.5, 2.5)
            yfact = random.uniform(0.5, 2.5)
            rotation = random.uniform(0, 360)
            oval = affinity.scale(circle,xfact=xfact, yfact=yfact, origin =(x, y))
            shapes.append((shape,oval, x, y, r, xfact, yfact,rotation))"""

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

def compute_sensing_values(scan_polygons, shapes, epsilon=1e-5):
    """
    Computes overlap ratios between each scan polygon and the union of shapes.
    If a polygon is invalid or intersection is negligible, assigns 0.0.
    """
    # Fix and unify all shape geometries first
    cleaned_shapes = []
    print("Shapes, ", shapes)
    for s in shapes:

        geom = s[1] if isinstance(s, tuple) else s[0]
        if not geom.is_valid:
            geom = geom.buffer(0)
        cleaned_shapes.append(geom)

    union_shapes = unary_union(cleaned_shapes)
    if not union_shapes.is_valid:
        union_shapes = union_shapes.buffer(0)

    values = []
    for poly in scan_polygons:
        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid:
                values.append(0.0)
                continue

        intersection = union_shapes.intersection(poly)
        intersection_area = intersection.area
        area = poly.area if poly.area > 0 else 1

        # Suppress very small ratios
        ratio = intersection_area / area
        if ratio < epsilon:
            ratio = 0.0
        values.append(round(ratio, 6))

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
    shapes = []
    #print(phantom_list)
    for shape_type, shape_geom,x, y, r, xfact,yfact, rotation in phantom_list:
        print(f"Shape type: {shape_type},shape geom: {shape_geom}, x: {x}, y: {y}, r: {r}, xfact: {xfact}, yfact: {yfact}, rotation: {rotation}")
        if shape_type == 'circle':
            shape_geom = Point(x, y).buffer(r, resolution=50)
        elif shape_type == 'square':
            shape_geom = affinity.rotate(box(x - r, y - r, x + r, y + r), rotation, origin=(x, y))
        elif shape_type == 'oval':
            shape_geom = Point(x, y).buffer(r, resolution=50)
            shape_geom = affinity.scale(shape_geom, xfact=xfact, yfact=yfact, origin=(x, y))
            shape_geom = affinity.rotate(shape_geom, rotation, origin=(x, y))
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        shapes.append((shape_type, shape_geom,x, y, r, xfact,yfact, rotation))

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
        for type,_, x, y, r,xf,yf, rot in shapes:
            if type == 'circle':
                ax.add_patch(Circle((x, y), r, color='gray', alpha=0.3))
            elif type == 'square':
                square = box(x - r, y - r, x + r, y + r)
                square = affinity.rotate(square, rot, origin=(x, y))
                ax.add_patch(PathPatch(Path(np.array(square.exterior.coords), closed=True), color='gray', alpha=0.3))
            elif type == 'oval':
                print("OVAL")
                print(f"Shape type: {type}, x: {x}, y: {y}, r: {r}, xfact: {xf}, yfact: {yf}, rotation: {rot}")
                circle = Point(x, y).buffer(r, resolution=50)
                oval = affinity.scale(circle, xfact=xf, yfact=yf, origin=(x, y))
                oval = affinity.rotate(oval, rot, origin=(x, y))
                ax.add_patch(PathPatch(Path(np.column_stack(oval.exterior.xy), closed=True), color='gray', alpha=0.3))

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
    else:
        pass
        #print("Scan values:")
        #for i, val in enumerate(sensing_values):
            #print(f"Scan {i+1:03d}: {val:.6f}")


    return sensing_values, shapes #shapes structure:(circle_geometry, x, y, r)




if __name__ == "__main__":
    numberSamples = 10000
    numberElectrodes = 15
    numberExcitationElectrodes = 15
    save = True #save as file. If doing large batches, turn savegif and animate false
    savegif = False #DO NOT DO THIS UNLESS YOU ARE SURE AND HAVE NUMBERSAMPLES BELOW 10
    animate = False #In order for animations for savegif to work, this must also be true
    fade = False
    # shapes structure:(circle_geometry, x, y, r)
    data = []

    for i in range(numberSamples):
        print(f"Sample {i + 1}")
        sensing_values, shapes = main(radius=5, num_electrodes=numberElectrodes,num_excitation=numberExcitationElectrodes, fade=fade, animate=animate, savegif=savegif)

        obj_list = [{"center": [round(x, 8), round(y, 8)], "radius": round(r, 8), "type":type, "xf": round(xf,8), "yf":round(yf,8),"rotation":rot} for (type,_, x, y, r,xf,yf, rot) in shapes]
        data.append({
            "objects": obj_list,
            "measurements": [round(val, 6) for val in sensing_values]
        })

    if save:

        output_dir = "Data"  # relavite path to the desired directory
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"npgC_{numberSamples}_{numberElectrodes}e{numberExcitationElectrodes}_traintest.json") #format as 'npg_{number of samples}_{number of electrodes}e{number of emitting electrodes}_{purpose(test, train, traintest)}.json'

            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(data, f)
            print(f"Data successfully saved to {output_file}")
        except Exception as e:
            print(f"Error saving file: {e}")