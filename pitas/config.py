import os
import argparse
import pitas_io

PITAS_ROOT           = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
DEFAULT_OUTPUT_DIR   = os.path.join(PITAS_ROOT, "output")
DEFAULT_RESOURCE_DIR = os.path.join(PITAS_ROOT, "resource")

pitas_io.create_dir(DEFAULT_OUTPUT_DIR)

argparser = argparse.ArgumentParser(
        prog='pitas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve'
        )


def get_output_dir():
    return DEFAULT_OUTPUT_DIR

def get_resource_dir():
    return DEFAULT_RESOURCE_DIR
