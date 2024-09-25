import subprocess
import os

def run_blender_proc(cad_path, output_dir, blender_path):
    # Set environment variables
    os.environ['OUTPUT_DIR'] = output_dir
    os.environ['BLENDER_PATH'] = blender_path

    # Build the command
    command = [
        'blenderproc', 'run',
        '--custom-blender-path', blender_path,
        'render_custom_templates.py',
        '--output_dir', output_dir,
        '--cad_path', cad_path
    ]

    # Execute the command
    subprocess.run(command)


OUTPUT_DIR = '/home/icetenny/senior-1/SAM-6D/SAM-6D/Data/linemod-ism-eval/templates'
BLENDER_PATH = '/home/icetenny/senior-1/blender-3.1.1-linux-x64/'
CAD_FOLDER = '/home/icetenny/senior-1/Linemod_preprocessed/models'


for obj_id in range(1, 16):
    cad_path = f"{CAD_FOLDER}/obj_{obj_id:02d}.ply"

    output_path = f"{OUTPUT_DIR}/{obj_id:02d}"

    # Create folder
    os.makedirs(output_path, exist_ok=True)

    print(f"Running {cad_path}")
    run_blender_proc(cad_path, output_path, BLENDER_PATH)
