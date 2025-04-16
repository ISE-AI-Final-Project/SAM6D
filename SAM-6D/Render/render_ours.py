import os
import subprocess


def run_blender_proc(cad_path, output_dir, blender_path):
    # # Set environment variables
    # os.environ['OUTPUT_DIR'] = output_dir
    # os.environ['BLENDER_PATH'] = blender_path

    # Build the command
    command = [
        "blenderproc",
        "run",
        "--custom-blender-path",
        blender_path,
        "render_custom_templates.py",
        "--output_dir",
        output_dir,
        "--cad_path",
        cad_path,
    ]

    # Execute the command
    subprocess.run(command)


OUTPUT_DIR = "/home/icetenny/senior-1/SAM-6D/SAM-6D/Data/ours/templates"
BLENDER_PATH = "/home/icetenny/senior-1/blender-3.1.1-linux-x64/"
CAD_FOLDER = "/home/icetenny/senior-2/ours_dataset"

for obj_name in os.listdir(CAD_FOLDER):
    cad_path = os.path.join(CAD_FOLDER, obj_name)

    output_path = f"{OUTPUT_DIR}/{obj_name.rstrip('.ply')}"

    # Create folder
    os.makedirs(output_path, exist_ok=True)

    print(f"Running {cad_path}")
    run_blender_proc(cad_path, output_path, BLENDER_PATH)
