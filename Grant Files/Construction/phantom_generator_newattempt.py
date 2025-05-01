import time

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.animation import FuncAnimation
import numpy as np
from pyparsing import Empty
from shapely.geometry import Polygon, Point
from matplotlib.patches import Polygon as mpl_polygon
import math




class ElectrodeArcPlotter:
    def __init__(self, num_arcs=12, circradius=5, center=(0, 0)):
        self.num_arcs = num_arcs
        self.circradius = circradius
        self.center = center
        self.pointdict = {
            "Electrode": {str(i + 1): {"start": (0, 0), "end": (0, 0)} for i in range(num_arcs)}
        }
        self.fig, self.ax = plt.subplots()

    def plot_polygon(self, polygon, color='blue', alpha=0.5):
        """
        Plot a Shapely Polygon using Matplotlib.

        :param polygon: A Shapely Polygon object.
        :param color: Color for the polygon.
        :param alpha: Transparency level (0 = fully transparent, 1 = fully opaque).
        """
        # Convert the polygon to a list of coordinates (vertices)
        x, y = polygon.exterior.xy
        # Create a Matplotlib polygon and add it to the plot
        mpl_patch = mpl_polygon(list(zip(x, y)), closed=True, color=color, alpha=alpha)
        self.ax.add_patch(mpl_patch)



    def generate_sensing_polygon(self, start_electrode, end_electrode, num_points=100):
        # Corrected point selection based on actual physical meaning
        # Inner arc: from END of start electrode to START of end electrode
        sx_inner, sy_inner = self.pointdict["Electrode"][str(start_electrode)]["end"]
        ex_inner, ey_inner = self.pointdict["Electrode"][str(end_electrode)]["start"]

        # Outer arc: from START of start electrode to END of end electrode
        sx_outer, sy_outer = self.pointdict["Electrode"][str(start_electrode)]["start"]
        ex_outer, ey_outer = self.pointdict["Electrode"][str(end_electrode)]["end"]

        # --- Compute distance for inner arc ---
        d_inner = math.hypot(ex_inner - sx_inner, ey_inner - sy_inner)

        # Apply heuristic center calculation based on distance d_inner
        if 2.58 < d_inner < 2.6:
            cx_inner = (sx_inner + ex_inner) / 1.9
            cy_inner = (sy_inner + ey_inner) / 1.9
        elif 4.99 < d_inner < 5.01:
            cx_inner = (sx_inner + ex_inner) / 1.5
            cy_inner = (sy_inner + ey_inner) / 1.5
        elif 7.06 < d_inner < 7.08:
            cx_inner = (sx_inner + ex_inner)
            cy_inner = (sy_inner + ey_inner)
        elif 8.65 < d_inner < 8.67:
            cx_inner = (sx_inner + ex_inner) * 2
            cy_inner = (sy_inner + ey_inner) * 2
        elif 9.64 < d_inner < 9.66:
            cx_inner = (sx_inner + ex_inner) * 5
            cy_inner = (sy_inner + ey_inner) * 5
        else:
            cx_inner = (sx_inner + ex_inner) / 2
            cy_inner = (sy_inner + ey_inner) / 2

        r_inner = math.hypot(sx_inner - cx_inner, sy_inner - cy_inner)

        # --- Compute angles for the inner arc ---
        start_ang_inner = math.degrees(math.atan2(sy_inner - cy_inner, sx_inner - cx_inner))
        end_ang_inner = math.degrees(math.atan2(ey_inner - cy_inner, ex_inner - cx_inner))

        # Normalize angles to handle wraparound
        theta1_inner = start_ang_inner % 360
        theta2_inner = end_ang_inner % 360
        if abs(theta2_inner - theta1_inner) > 180:
            # Ensure the arc doesn't go beyond the expected angle range
            if theta2_inner > theta1_inner:
                theta1_inner, theta2_inner = theta2_inner, theta1_inner - 360
            else:
                theta1_inner, theta2_inner = theta2_inner + 360, theta1_inner

        # Sample points along the inner arc
        arc_angles_inner = np.linspace(theta1_inner, theta2_inner, num_points)
        arc_points_inner = [(cx_inner + r_inner * np.cos(np.radians(a)), cy_inner + r_inner * np.sin(np.radians(a))) for
                            a in arc_angles_inner]

        # --- Compute distance for outer arc ---
        d_outer = math.hypot(ex_outer - sx_outer, ey_outer - sy_outer)

        # Apply heuristic center calculation based on distance d_outer
        if 2.58 < d_outer < 2.6:
            cx_outer = (sx_outer + ex_outer) / 1.9
            cy_outer = (sy_outer + ey_outer) / 1.9
        elif 4.99 < d_outer < 5.01:
            cx_outer = (sx_outer + ex_outer) / 1.5
            cy_outer = (sy_outer + ey_outer) / 1.5
        elif 7.06 < d_outer < 7.08:
            cx_outer = (sx_outer + ex_outer)
            cy_outer = (sy_outer + ey_outer)
        elif 8.65 < d_outer < 8.67:
            cx_outer = (sx_outer + ex_outer) * 2
            cy_outer = (sy_outer + ey_outer) * 2
        elif 9.64 < d_outer < 9.66:
            cx_outer = (sx_outer + ex_outer) * 5
            cy_outer = (sy_outer + ey_outer) * 5
        else:
            cx_outer = (sx_outer + ex_outer) / 2
            cy_outer = (sy_outer + ey_outer) / 2

        r_outer = math.hypot(sx_outer - cx_outer, sy_outer - cy_outer)

        # --- Compute angles for the outer arc ---
        start_ang_outer = math.degrees(math.atan2(sy_outer - cy_outer, sx_outer - cx_outer))
        end_ang_outer = math.degrees(math.atan2(ey_outer - cy_outer, ex_outer - cx_outer))

        # Normalize angles to handle wraparound
        theta1_outer = start_ang_outer % 360
        theta2_outer = end_ang_outer % 360
        if abs(theta2_outer - theta1_outer) > 180:
            # Ensure the arc doesn't go beyond the expected angle range
            if theta2_outer > theta1_outer:
                theta1_outer, theta2_outer = theta2_outer, theta1_outer - 360
            else:
                theta1_outer, theta2_outer = theta2_outer + 360, theta1_outer

        # Sample points along the outer arc
        arc_angles_outer = np.linspace(theta1_outer, theta2_outer, num_points)
        arc_points_outer = [(cx_outer + r_outer * np.cos(np.radians(a)), cy_outer + r_outer * np.sin(np.radians(a))) for
                            a in arc_angles_outer]

        # Handle the straight case (line and arc together)
        if abs(d_inner - d_outer) < 0.1:
            A = (sx_inner, sy_inner)
            B = (ex_inner, ey_inner)
            D = (ex_outer, ey_outer)
            C = (sx_outer, sy_outer)
            pts = [B, D, C, A]
            return Polygon(pts)

        if (abs(start_electrode - end_electrode) == 5):
            A = (sx_inner, sy_inner)
            B = (ex_inner, ey_inner)
            D = (ex_outer, ey_outer)
            C = (sx_outer, sy_outer)
            pts = arc_points_inner + [B, D, C, A]
            return Polygon(pts)

        if (abs(start_electrode - end_electrode) == 7):
            A = (sx_inner, sy_inner)
            B = (ex_inner, ey_inner)
            D = (ex_outer, ey_outer)
            C = (sx_outer, sy_outer)
            pts = arc_points_outer+[D, B, A, C]
            return Polygon(pts)

        # For arcs (normal case)
        else:
            sensing_polygon_points = arc_points_inner + arc_points_outer[::-1]  # Combine arcs
            sensing_polygon = Polygon(sensing_polygon_points)
            return sensing_polygon

    def detect_overlap(self, objects, start_electrode, end_electrode, startorendpoint="start"):
        # Generate the sensing polygon for the current scan
        sensing_polygon = self.generate_sensing_polygon(start_electrode, end_electrode, startorendpoint=startorendpoint)

        overlaps = []
        for obj in objects:
            # Compute the intersection of the sensing polygon and the object
            intersection = sensing_polygon.intersection(obj)

            if intersection.is_empty:
                overlaps.append(0)  # No overlap
            else:
                # Calculate the area of the intersection
                overlap_area = intersection.area
                object_area = obj.area
                overlap_percentage = (overlap_area / object_area)
                overlaps.append(overlap_percentage)

        return overlaps

    # Detect overlap percentages for each start (se) and end (ee) electrode pair
    def detect_all_overlaps(self,objects):
        overlap_results = []

        for se in range(1, 13):  # Start electrode goes from 1 to 12
            for ee in range(se + 1, 13):  # End electrode goes from se+1 to 12
                overlap_percentages = self.detect_overlap(objects, start_electrode=se, end_electrode=ee)
                overlap_results.append((se, ee, overlap_percentages))

        if overlap_results is Empty:
            print("No overlaps detected.")
        return overlap_results

    def generate_arcs(self):
        for i in range(self.num_arcs):
            start_angle = (360 / self.num_arcs) * i
            end_angle = start_angle + (360 / self.num_arcs)

            arc = Arc(self.center,
                      width=2 * self.circradius,
                      height=2 * self.circradius,
                      angle=0,
                      theta1=start_angle,
                      theta2=end_angle,
                      color='blue')
            self.ax.add_patch(arc)


            sx = self.center[0] + self.circradius * np.cos(np.radians(start_angle))
            sy = self.center[1] + self.circradius * np.sin(np.radians(start_angle))
            ex = self.center[0] + self.circradius * np.cos(np.radians(end_angle))
            ey = self.center[1] + self.circradius * np.sin(np.radians(end_angle))

            self.pointdict["Electrode"][str(i+1)] = {"start": (sx, sy), "end": (ex, ey)}

            self.ax.plot(sx, sy, 'ro', label='_nolegend_')
            self.ax.plot(ex, ey, 'go', label='_nolegend_')

    def add_scanning_arc(self, start_electrode, end_electrode, startorendpoint="start"):
        # Choose start and end points
        if startorendpoint == "end":
            sx, sy = self.pointdict["Electrode"][str(start_electrode)]["start"]
            ex, ey = self.pointdict["Electrode"][str(end_electrode)]["end"]
            color = 'blue'
        else:
            sx, sy = self.pointdict["Electrode"][str(start_electrode)]["end"]
            ex, ey = self.pointdict["Electrode"][str(end_electrode)]["start"]
            color = 'green'

        # Debugging output

        # Plot start and end points
        self.ax.plot(sx, sy, 'o', color=color, label=f"Start ({startorendpoint})")
        self.ax.plot(ex, ey, 'o', color=color, label=f"End   ({startorendpoint})")

        # Calculate distance between the points
        d = math.hypot(ex - sx, ey - sy)
        print(f"Distance (d) between electrodes: {d:.2f}")

        # Check if distance is too small for arcs, just plot a line
        if abs(end_electrode - start_electrode) == 6:
            # Short arc (if distance is small, just plot a line)
            print("Short arc detected. Plotting line.")
            self.ax.plot([sx, ex], [sy, ey], color='red', label='Line')
            return

        # Center calculations based on distance heuristics (for arcs)
        if 2.58 < d < 2.6:
            cx = (sx + ex) / 1.9
            cy = (sy + ey) / 1.9
        elif 4.99 < d < 5.01:
            cx = (sx + ex) / 1.5
            cy = (sy + ey) / 1.5
        elif 7.06 < d < 7.08:
            cx = (sx + ex)
            cy = (sy + ey)
        elif 8.65 < d < 8.67:
            cx = (sx + ex) * 2
            cy = (sy + ey) * 2
        elif 9.64 < d < 9.66:
            cx = (sx + ex) * 5
            cy = (sy + ey) * 5
        elif 9.9 < d < 10.1:
            print("Long arc detected. Plotting line.")
            self.ax.plot([sx, ex], [sy, ey], color='red', label='Line')
            return
        else:
            cx = (sx + ex) / 2
            cy = (sy + ey) / 2

        # Calculate the radius of the circle
        r = math.hypot(sx - cx, sy - cy)

        # Calculate angles for the arc
        start_ang = math.degrees(math.atan2(sy - cy, sx - cx))
        end_ang = math.degrees(math.atan2(ey - cy, ex - cx))

        # Normalize angles to minor arc (interior)
        theta1 = start_ang % 360
        theta2 = end_ang % 360
        diff = (theta2 - theta1) % 360
        if diff > 180:
            # Swap to take the shorter path
            theta1, theta2 = theta2, theta1 + 360

        # Debugging output for angles

        # Draw the arc between the electrodes
        arc = Arc((cx, cy), width=2 * r, height=2 * r,
                  angle=0, theta1=theta1, theta2=theta2, color='red')
        self.ax.add_patch(arc)

    def plot(self):
        self.ax.set_xlim(-self.circradius-1, self.circradius+1)
        self.ax.set_ylim(-self.circradius-1, self.circradius+1)
        self.ax.set_aspect('equal', 'box')
        plt.legend(loc='upper right')
        plt.show()

if __name__ == "__main__":
    electrodecount = 12
    startelectrode = 1
    scancount = 0
    roundcount = 1
    totalscancount = 0
    isfirstscan = True
    previous_start_electrode = None

    plotter = ElectrodeArcPlotter()
    current_start_electrode = startelectrode  # Initialize the starting electrode




    def update(frame, hits=[0]):
        global current_start_electrode, scancount, roundcount, totalscancount, isfirstscan, previous_start_electrode  # Use a global variable to track the start electrode
        plotter.ax.clear()  # Clear the previous frame

        plotter.generate_arcs()

        # Reset isfirstscan only when transitioning from the last electrode to the first
        if previous_start_electrode is not None and current_start_electrode == 1 and previous_start_electrode == electrodecount:
            isfirstscan = True




        # Skip the first scan
        if isfirstscan:
            print("Skipping the first scan.")
            isfirstscan = False
            scancount += 1

            return

        # Calculate the current end electrode
        current_end_electrode = (current_start_electrode + scancount) % electrodecount or electrodecount



        # Skip logic when the current electrode has scanned the required number of times
        if scancount >= (electrodecount - current_start_electrode + 1):
            print(f"Electrode {current_start_electrode} completed scanning.")
            scancount = 0
            previous_start_electrode = current_start_electrode  # Update the previous electrode
            current_start_electrode = (current_start_electrode % electrodecount) + 1
            roundcount += 1
            isfirstscan = True  # <<< ADD THIS HERE!

            if current_start_electrode == electrodecount:
                ani.event_source.stop()

            return

        print(
            f"Scanning from Electrode {current_start_electrode} to {current_end_electrode} (Scancount: {scancount}), totalscancount: {totalscancount}")

        # Add scanning arcs



        plotter.add_scanning_arc(current_start_electrode, current_end_electrode, "start")
        plotter.add_scanning_arc(current_start_electrode, current_end_electrode, "end")

        # Plot sensing polygon for the current start and end electrodes
        polygon = plotter.generate_sensing_polygon(current_start_electrode, current_end_electrode)
        plotter.plot_polygon(polygon, color='orange', alpha=0.3)  # Plot the sensing polygon with transparency

        scancount += 1  # Increment scancount after processing the current scan

        # Stop after completing the required number of rotations
        if current_start_electrode == 1 and scancount == 0:
            hits[0] += 1
            if hits[0] == electrodecount + 2:  # Stop after completing two full rotations
                ani.event_source.stop()

        totalscancount += 1

        # Update plot limits and aspect ratio
        plotter.ax.set_xlim(-plotter.circradius - 1, plotter.circradius + 1)
        plotter.ax.set_ylim(-plotter.circradius - 1, plotter.circradius + 1)
        plotter.ax.set_aspect('equal', 'box')

        previous_start_electrode = current_start_electrode

        time.sleep(0.5)  # Optional: Add a small delay for better visualization

    ani = FuncAnimation(plotter.fig, update, frames=electrodecount, repeat=True)
    plt.show()
    print(f"totalscancount: {totalscancount}")


    objects = []

    # Get all overlap percentages for all start and end electrode combinations
    overlap_results = plotter.detect_all_overlaps(objects)

    # Print the results
    print(overlap_results)
    for se, ee, overlap_percentages in overlap_results:
        print(f"Start Electrode: {se}, End Electrode: {ee}")
        for i, overlap in enumerate(overlap_percentages):
            print(f"  Object {i + 1} overlap: {overlap:.2f}%")