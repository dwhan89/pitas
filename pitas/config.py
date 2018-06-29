import os
import pitas_io

CUSPS_ROOT         = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
DEFAULT_OUTPUT_DIR = os.path.join(CUSPS_ROOT, "output")

pitas_io.create_dir(DEFAULT_OUTPUT_DIR)

def get_output_dir():
    return DEFAULT_OUTPUT_DIR
