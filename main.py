import tkinter as tk
from tkinter import ttk
import pyvista as pv
from pyvista import Plotter
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

bone_color = ['yellow', 'orange', 'green']

class PyVistaApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PyVista Plotter")
        self.geometry("600x600")

        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.label = ttk.Label(self.scrollable_frame, text="Click the button to display the plot")
        self.label.pack(pady=20)

        self.plot_button = ttk.Button(self.scrollable_frame, text="Display Plot", command=self.display_plot)
        self.plot_button.pack(pady=10)

        self.bones = ['patella', 'femur', 'tibia']
        self.solids = []
        self.landmark_buttons = {}
        self.landmark_checkbuttons = {}
        self.landmark_vars = {}
        self.landmark_points = {}
        self.glyph_actors = {}
        self.line_actors = {}
        self.label_actors = {}
        bone1 = ['tibia', 'femur']
        self.hide_buttons = []
        self.transparency_sliders = []
        self.rotation_angle = 2  # Angle to rotate each step

    def display_plot(self):
        kinds = ['/Users/dweejpandya/Downloads/SRI - Copy - Copy/patella.obj', '/Users/dweejpandya/Downloads/SRI - Copy - Copy/femur.obj', '/Users/dweejpandya/Downloads/SRI - Copy - Copy/tibia.obj']
        centers = [
            (0, 1, 0),
            (0, 0, 0),
            (0, 2, 0),
        ]

        self.plotter = Plotter()

        for kind, bone, bc in zip(kinds, self.bones, bone_color):
            solid = pv.read(kind)
            if bone == "tibia":
                self.tibia_actor = solid
                translation_matrix = np.eye(4)
                solid.transform(translation_matrix)
            self.solids.append(solid)
            self.plotter.add_mesh(solid, name=bone, color=bc, show_edges=False)

        self.add_landmarks_and_lines()

        self.plotter.view_vector((5.0, 2, 3))

        self.plotter.show(interactive_update=True)

        self.add_controls()

    def add_landmarks_and_lines(self):
        # Load the CSV files
        tibia_csv_file = "/Users/dweejpandya/Downloads/SRI - Copy - Copy/LM_INFO_LPS_Coord_tibia.csv"
        femur_csv_file = "/Users/dweejpandya/Downloads/SRI - Copy - Copy/LM_INFO_LPS_Coord_femur.csv"

        tibia_df = pd.read_csv(tibia_csv_file)
        femur_df = pd.read_csv(femur_csv_file)

        # Extract points and labels from the tibia CSV
        tibia_points = tibia_df[['x', 'y', 'z']].values
        tibia_labels = tibia_df['label'].tolist()

        # Extract points and labels from the femur CSV
        femur_points = femur_df[['x', 'y', 'z']].values
        femur_labels = femur_df['label'].tolist()

        # Create point clouds for both tibia and femur
        tibia_point_cloud = pv.PolyData(tibia_points)
        femur_point_cloud = pv.PolyData(femur_points)

        # Store landmark points and actors for buttons
        self.landmark_points = {label: point for label, point in zip(tibia_labels, tibia_points)}
        self.landmark_points.update({label: point for label, point in zip(femur_labels, femur_points)})

        # Create glyphs for the landmark points and store actors
        self.glyph_actors = {}
        for label, point in zip(tibia_labels + femur_labels, tibia_points.tolist() + femur_points.tolist()):
            sphere = pv.Sphere(radius=2, center=point)
            actor = self.plotter.add_mesh(sphere, name=label, color='red')
            self.glyph_actors[label] = actor

        # Add point labels for both tibia and femur
        tibia_label_actor = self.plotter.add_point_labels(tibia_point_cloud, tibia_labels, font_size=10, text_color='black', point_size=0, always_visible=True)
        femur_label_actor = self.plotter.add_point_labels(femur_point_cloud, femur_labels, font_size=10, text_color='black', point_size=0, always_visible=True)

        self.label_actors['tibia'] = tibia_label_actor
        self.label_actors['femur'] = femur_label_actor

        # Define function to get a point by label from a dataframe
        def get_point_by_label(df, label):
            point = df[df['label'] == label][['x', 'y', 'z']].values
            if len(point) == 0:
                raise ValueError(f"Label '{label}' not found in dataframe.")
            return point[0]

        # Get specific points for tibia
        tkc = get_point_by_label(tibia_df, 'TKC')
        ta = get_point_by_label(tibia_df, 'TA')
        tmc = get_point_by_label(tibia_df, 'TMC')
        tlc = get_point_by_label(tibia_df, 'TLC')

        # Get specific points for femur
        fkc = get_point_by_label(femur_df, 'FKC')
        fhp = get_point_by_label(femur_df, 'FHP')

        # Define function to extend a line between two points
        def extend_line(start, end, extension=50):
            direction = end - start
            direction = direction / np.linalg.norm(direction)
            extended_start = start - direction * extension
            extended_end = end + direction * extension
            return extended_start, extended_end

        # Extend lines for tibia points
        tkc_ext, ta_ext = extend_line(tkc, ta)
        tmc_ext, tlc_ext = extend_line(tmc, tlc)

        # Extend lines for femur points
        fkc_ext, fhp_ext = extend_line(fkc, fhp)

        # Prepare line points and connectivity for tibia
        tibia_line_points = np.array([tkc_ext, ta_ext, tmc_ext, tlc_ext])
        tibia_lines = np.array([2, 0, 1, 2, 2, 3])

        tibia_pdata = pv.PolyData(tibia_line_points)
        tibia_pdata.lines = tibia_lines

        # Prepare line points and connectivity for femur
        femur_line_points = np.array([fkc_ext, fhp_ext])
        femur_lines = np.array([2, 0, 1])

        femur_pdata = pv.PolyData(femur_line_points)
        femur_pdata.lines = femur_lines

        # Add the extended lines to the plotter for tibia and femur
        tibia_lines_actor = self.plotter.add_mesh(tibia_pdata, name="tibia_lines", color='blue', line_width=5)
        femur_lines_actor = self.plotter.add_mesh(femur_pdata, name="femur_lines", color='green', line_width=5)

        # Store line actors for toggling visibility
        self.line_actors['tibia'] = tibia_lines_actor
        self.line_actors['femur'] = femur_lines_actor

    def add_controls(self):
        tibia_frame = ttk.Frame(self.scrollable_frame)
        tibia_frame.pack(side=tk.LEFT, fill='y', padx=10)

        femur_frame = ttk.Frame(self.scrollable_frame)
        femur_frame.pack(side=tk.LEFT, fill='y', padx=10)

        for label in self.landmark_points.keys():
            if label.startswith('T'):
                frame = ttk.Frame(tibia_frame)
                frame.pack(pady=2, fill='x')

                button = ttk.Button(frame, text=label, command=lambda l=label: self.change_color(l))
                button.pack(side='left')

                var = tk.IntVar()
                checkbox = ttk.Checkbutton(frame, variable=var, command=lambda l=label, v=var: self.toggle_color(l, v))
                checkbox.pack(side='right')

                self.landmark_buttons[label] = button
                self.landmark_checkbuttons[label] = checkbox
                self.landmark_vars[label] = var
            elif label.startswith('F'):
                frame = ttk.Frame(femur_frame)
                frame.pack(pady=2, fill='x')

                button = ttk.Button(frame, text=label, command=lambda l=label: self.change_color(l))
                button.pack(side='left')

                var = tk.IntVar()
                checkbox = ttk.Checkbutton(frame, variable=var, command=lambda l=label, v=var: self.toggle_color(l, v))
                checkbox.pack(side='right')

                self.landmark_buttons[label] = button
                self.landmark_checkbuttons[label] = checkbox
                self.landmark_vars[label] = var

        control_frame = ttk.Frame(self.scrollable_frame)
        control_frame.pack(side=tk.LEFT, fill='y', padx=10)

        for bone in self.bones:
            hide_button = ttk.Button(control_frame, text=f"Hide {bone}", command=lambda b=bone: self.hide_bone(b))
            hide_button.pack(pady=5)
            self.hide_buttons.append(hide_button)

            transparency_label = ttk.Label(control_frame, text=f"{bone} Transparency")
            transparency_label.pack(pady=5)

            transparency_slider = ttk.Scale(control_frame, from_=1, to=0, orient='horizontal', command=lambda val, b=bone: self.set_transparency(b, val))
            transparency_slider.pack(pady=5)
            self.transparency_sliders.append(transparency_slider)

        rotation_frame = ttk.Frame(control_frame)
        rotation_frame.pack(pady=5)

        rotation_label = ttk.Label(rotation_frame, text="Tibia Rotation")
        rotation_label.pack(pady=5)

        rotate_positive_button = ttk.Button(rotation_frame, text="Rotate +", command=lambda: self.rotate_tibia(self.rotation_angle))
        rotate_positive_button.pack(pady=5)

        rotate_negative_button = ttk.Button(rotation_frame, text="Rotate -", command=lambda: self.rotate_tibia(-self.rotation_angle))
        rotate_negative_button.pack(pady=5)

    def change_color(self, label):
        var = self.landmark_vars[label]
        var.set(1)  # Check the checkbox
        self.glyph_actors[label].GetProperty().SetColor(0, 1, 0)  # Change color to green

    def toggle_color(self, label, var):
        if var.get() == 0:
            self.glyph_actors[label].GetProperty().SetColor(1, 0, 0)  # Change color to red
        else:
            self.glyph_actors[label].GetProperty().SetColor(0, 1, 0)  # Change color to green

    def hide_bone(self, bone):
        # Toggle the visibility of the bone
        actor = self.plotter.renderer._actors[bone]
        visibility = not actor.GetVisibility()
        actor.SetVisibility(visibility)

        # Toggle the visibility of the landmark points
        if bone in ['femur', 'tibia']:  # Only toggle landmark points for femur and tibia
            for label, actor in self.glyph_actors.items():
                if label.startswith(bone[0].upper()):
                    actor.SetVisibility(visibility)

            # Toggle the visibility of the point labels
            self.label_actors[bone].SetVisibility(visibility)

            # Toggle the visibility of the lines
            if bone in self.line_actors:
                self.line_actors[bone].SetVisibility(visibility)

        elif bone == 'patella':
            # No landmarks for patella, so only toggle the bone itself
            pass

        self.plotter.update()

    def set_transparency(self, bone, value):
        actor = self.plotter.renderer._actors[bone]
        actor.GetProperty().SetOpacity(float(value))
        self.plotter.update()

    def rotate_tibia(self, angle):
        if self.tibia_actor:
            r = R.from_euler('y', angle, degrees=True)
            rotation_matrix = r.as_matrix()

            # Create a homogeneous transformation matrix
            homogeneous_matrix = np.eye(4)
            homogeneous_matrix[:3, :3] = rotation_matrix

            # Reapply the initial translation and then rotate
            translation_matrix = np.eye(4)
            transformation_matrix = translation_matrix @ homogeneous_matrix  

            self.tibia_actor.points = self.tibia_actor.points @ transformation_matrix[:3, :3]

            self.plotter.update()

if __name__ == "__main__":
    app = PyVistaApp()
    app.mainloop()
