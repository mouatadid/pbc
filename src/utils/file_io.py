# File input and output utility functions
import os
import subprocess
from pathlib import Path

def make_directories(dirname):
    """Creates directory and parent directories with 777 permissions
    if they do not exist

    Args:
      dirname - directory path string or pathlib.Path
    """
    if dirname != '' and dirname != Path(''):
        os.umask(0)
        os.makedirs(dirname, exist_ok=True, mode=0o777)

def make_parent_directories(file_path):
    """Creates parent directories of a given file with 777 permissions
    if they do not exist

    Args:
      file_path - file path string or pathlib.Path
    """
    make_directories(os.path.dirname(file_path))

def symlink(src, dest, use_abs_path=False):
    """Symlinks dest to point to src; if dest was previously a symlink,
    unlinks src first

    Args:
      src - source file name
      dest - target file name
      use_abs_path - if True, links dest to the real absolute path of src,
       resolving all symlinks (useful when src is not expressed relative to dest)
    """
    # n and f flags ensure that prior symlink is overwritten by new one
    if use_abs_path:
        # realpath resolves symlinks and returns absolute path
        src = os.path.realpath(src)
    cmd = "ln -nsf {} {}".format(src, dest)
    subprocess.call(cmd, shell=True)


def set_file_permissions(file_path, skip_if_exists=False, throw=False,
                         mode=0o777):
    """Set file/folder permissions.

    Args:
      file_path : file path string or pathlib.Path

      skip_if_exists : boolean
        If True, skips setting permissions if file exists

      throw : boolean
        If True, throws exception if cannot set permissions

      mode : which file permissions to set
    """
    if not skip_if_exists or (skip_if_exists and not os.path.exists(file_path)):
        # Set permissions
        try:
            os.chmod(file_path, mode)
        except Exception as err:
            if throw:
                raise err
            else:
                pass

def set_recursive_permissions(dir_path, skip_if_exists=False, throw=False,
                              permissions="a+rwx"):
    """Recursively set permissions for a directory and all elements thereof.

    Args:
      dir_path : directory path string or pathlib.Path

      skip_if_exists : boolean
        If True, skips setting permissions if file exists

      throw : boolean
        If True, throws exception if cannot set permissions

      mode : which file permissions to set
    """
    if not skip_if_exists or not os.path.exists(dir_path):
        # Set permissions
        try:
            os.system(f"chmod -Rf {permissions} {dir_path}")
        except Exception as err:
            if throw:
                raise err
            else:
                pass
