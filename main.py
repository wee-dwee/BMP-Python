import tkinter as tk
import vtk
from tkinter import ttk
import pyvista as pv
from pyvista import Plotter
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
import os
import requests
import json
import cv2
import random
import base64
import numpy as np

bone_color = ['yellow', 'orange', 'green']

api_end_point = "http://127.0.0.1:5000"

def record_coordinates(label, x, y, z):
        file_name = 'records.csv'
        
        # Check if the file already exists
        file_exists = os.path.isfile(file_name)
        
        # Open the file in append mode
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header if the file does not exist
            if not file_exists:
                writer.writerow(['Label', 'X', 'Y', 'Z'])
            
            # Write the label and coordinates
            writer.writerow([label, x, y, z])
            
        print(f"Record added: {label}, {x}, {y}, {z}")

    
def base64_to_image(image_base64):
    image_data = base64.b64decode(image_base64)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)  # Convert to JPEG format
    image_base64 = base64.b64encode(buffer).decode('utf-8')  # Encode to base64
    return image_base64


def show_image(captured_image):
    if captured_image is None:
        print("Error: No image to display. Capture an image first.")
        return
    
    # Display the captured image in a window
    cv2.imshow("Captured Image :", captured_image)
    # Wait for any key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_image():
    response = requests.get(api_end_point+"/capture_image")
    json_response = response.json()
    if "image" in json_response:
        image_base64 = json_response["image"]
        image = base64_to_image(image_base64)
        return image
    else :
        return None
        
def get_points(image_base64):
    payload = {
        "image": image_base64
    }

    response = requests.post(api_end_point + "/calculate_points" , json=payload)
    json_response = response.json()
    if "points" in json_response:
        return json_response["points"]
    else:
        return None


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

        self.style = ttk.Style()
        self.style.configure('White.TButton' , foreground = 'white')
        self.style.configure('Green.TButton', foreground='yellow', font=('TkDefaultFont',13,'bold'))

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
        kinds = ['patella.obj', 'femur.obj', 'tibia.obj']
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

        # Hide the label and button after displaying the plot
        self.label.pack_forget()
        self.plot_button.pack_forget()


    

    def add_landmarks_and_lines(self):
        # Load the CSV files
        tibia_csv_file = "LM_INFO_LPS_Coord_tibia.csv"
        femur_csv_file = "LM_INFO_LPS_Coord_femur.csv"

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
        # Tibia frame and title
        tibia_frame = ttk.Frame(self.scrollable_frame)
        tibia_frame.pack(side=tk.LEFT, fill='y', padx=10, pady=10)

        tibia_title = ttk.Label(tibia_frame, text="Tibia", font=("Arial", 12, "bold"))
        tibia_title.pack(pady=(0, 10))  # Add some padding below the title

        # Femur frame and title
        femur_frame = ttk.Frame(self.scrollable_frame)
        femur_frame.pack(side=tk.LEFT, fill='y', padx=10, pady=10)

        femur_title = ttk.Label(femur_frame, text="Femur", font=("Arial", 12, "bold"))
        femur_title.pack(pady=(0, 10))  # Add some padding below the title

        # Add buttons and controls for Tibia
        for label in self.landmark_points.keys():
            if label.startswith('T'):
                frame = ttk.Frame(tibia_frame)
                frame.pack(pady=2, fill='x')

                button = ttk.Button(frame, text=label, style='White.TButton', command=lambda l=label: self.change_color(l))
                button.pack(side='left')

                var = tk.IntVar()
                self.landmark_buttons[label] = button
                self.landmark_vars[label] = var

        # Add buttons and controls for Femur
        for label in self.landmark_points.keys():
            if label.startswith('F'):
                frame = ttk.Frame(femur_frame)
                frame.pack(pady=2, fill='x')

                button = ttk.Button(frame, text=label, style='White.TButton', command=lambda l=label: self.change_color(l))
                button.pack(side='left')

                var = tk.IntVar()
                self.landmark_buttons[label] = button
                self.landmark_vars[label] = var

            # Control frame
        control_frame = ttk.Frame(tibia_frame)
        control_frame.pack(pady=(50, 10))  # Increase the top padding for more space between frames
        
        control_title = ttk.Label(control_frame, text="Controls", font=("Arial", 12, "bold"))
        control_title.pack(pady=(0, 10))  # Add some padding below the title

        for bone in self.bones:
            hide_button = ttk.Button(control_frame, text=f"Hide {bone}", command=lambda b=bone: self.hide_bone(b))
            hide_button.pack(pady=5)
            self.hide_buttons.append(hide_button)

            transparency_label = ttk.Label(control_frame, text=f"{bone} Transparency")
            transparency_label.pack(pady=5)

            transparency_slider = ttk.Scale(control_frame, from_=1, to=0, orient='horizontal', command=lambda val, b=bone: self.set_transparency(b, val))
            transparency_slider.pack(pady=5)
            self.transparency_sliders.append(transparency_slider)

        # Rotation controls
        rotation_frame = ttk.Frame(control_frame)
        rotation_frame.pack(pady=5)

        rotation_label = ttk.Label(rotation_frame, text="Tibia Rotation")
        rotation_label.pack(pady=5)

        rotate_positive_button = ttk.Button(rotation_frame, text="Rotate +", command=lambda: self.rotate_tibia(self.rotation_angle))
        rotate_positive_button.pack(pady=5)

        rotate_negative_button = ttk.Button(rotation_frame, text="Rotate -", command=lambda: self.rotate_tibia(-self.rotation_angle))
        rotate_negative_button.pack(pady=5)


    def change_color(self, label):
        btn = self.landmark_buttons[label]
        currStyle = btn.cget('style')
        if(currStyle=='Green.TButton'):
            btn.config(style = 'White.TButton')
            self.glyph_actors[label].GetProperty().SetColor(1, 0, 0)  # Change color to red
        else:
            btn.config(style = 'Green.TButton')
            self.glyph_actors[label].GetProperty().SetColor(0, 1, 0)  # Change color to green
        self.plotter.update()

        if(currStyle=='Green.TButton'):
            return

        print(f"Label : {label}")

        image = get_image()
        show_image(image)
        if image is None:
            print("Error Getting Image.")
            return
        image_base64 = image_to_base64(image)
        if image_base64 is None:
            print("Error converting image to base64.")
            return
        
        points = get_points(image_base64)
        if points is None:
            print("Error calculating points")
            return
        
        record_coordinates(label , points[0] , points[1] , points[2])




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
