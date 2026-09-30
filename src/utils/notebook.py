"""Utility functions supporting jupyter notebooks."""
from utils.timing import tic, toc
from utils.logging import printf
import subprocess
import shlex
import sys


def isnotebook():
    """Returns True if code is being executed interactively as a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def call_notebook(ntbk, extra_args="", nbconvert=True):
    """Converts a jupyter notebook to a Python script.

    Executes notebook with commandline arguments

    Args:
        ntbk (str): file name of jupyter notebook ending in .ipynb
        extra_args (str): string containing extra command line args to pass to notebook
        nbconvert (bool): if False, skips nbconvert step and calls an existing Python script
    """
    if nbconvert:
        tic()
        # Convert jupyter notebook to script and remove extraneous folders generated
        # by nbconvert
        subprocess.call(f'jupyter nbconvert --to script "{ntbk}"; '
                        r'rm -rf nbconvert; mv \~ deleteme; rm -rf deleteme',
                        shell=True)
        toc()
    # Reconstruct command-line arguments and pass to script
    cmd_args = " ".join(map(shlex.quote, sys.argv[1:]))
    script = ntbk.replace(".ipynb", ".py")
    cmd = f"python \"{script}\" {cmd_args} {extra_args}"
    printf(f"Executing {cmd}")
    subprocess.call(cmd, shell=True)
