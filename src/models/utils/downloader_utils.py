import os
import sys
import subprocess
import warnings
from os.path import expanduser
from subprocess import CalledProcessError

# Globals
DEFAULT_SUBSEASONAL_DATA_DIR = "subseasonal_data"
SUBSEASONAL_DATA_SUBDIRS = ["dataframes", "combined_dataframes", "masks"]
SUBSEASONAL_DATA_BLOB = "https://subseasonalusa.blob.core.windows.net/subseasonalusa"


def download(verbose=True):
    """Download or sync the entire subseasonal dataset from Azure storage.

    Download requires the Azure Storage CLI ``azcopy`` which can be installed from
    https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy. To check
    whether ``azcopy`` was installed correctly, see :func:`~subseasonal_data.downloader.check_azcopy_install`.

    The data is organized into the following subdirectories:
        * **dataframes**: individual dataframes containing ground truth, climatology, etc. data
        * **combined_dataframes**: lat-lon-day dataframes that merge individual dataframes
        * **masks**: lat-lon filters for Western U.S. and contiguous U.S.

    To get a list of all available files, see :func:`~subseasonal_data.downloader.list_subdir_files`.

    The data will be downloaded in the location given by :func:`~subseasonal_data.downloader.list_subdir_files`.

    If the data was downloaded before, this function will instead sync the modified files.

    Parameters
    ----------
    verbose: bool, default True
        Whether to redirect download progress messages to stdout.

    """
    # Get data path
    data_path = get_subseasonal_data_path()
    # Check azcopy is installed
    check_azcopy_install()
    # Sync data
    for data_subdir in SUBSEASONAL_DATA_SUBDIRS:
        # Make dirs
        print(f"Downloading data from the '{data_subdir}' directory...")
        data_subdir_path = os.path.join(
            data_path, data_subdir)
        if not os.path.exists(data_subdir_path):
            os.makedirs(data_subdir_path)
        # Run azcopy sync
        # Use Popen to access logs in real time
        azcopy_cmd = f"azcopy sync {
            SUBSEASONAL_DATA_BLOB}/{data_subdir} {data_subdir_path} --recursive"
        _subprocess_with_realtime_log(cmd=azcopy_cmd, verbose=verbose)


def download_file(data_subdir, filename, verbose=True, allow_write=False):
    """Download or sync one subseasonal data file from Azure storage.

    Behavior and is similar to :func:`~subseasonal_data.downloader.download`.

    If the file was downloaded before, this function will instead sync the target file.

    Parameters
    ----------
    data_subdir: {'dataframes', 'combined_dataframes', 'masks'}
        Azure data directory of target file.

    filename: string
        Name of target file.

    verbose: bool, (default=True)
        Whether to redirect download progress messages to stdout.

    allow_write: bool, (default=False)
        Whether to give write permissions to all users when syncing files.
        Recommended if working in shared directories. Users must be allowed
        to set permissions.

    """
    # Check data_subdir is valid
    if data_subdir not in SUBSEASONAL_DATA_SUBDIRS:
        raise ValueError(
            f"The data_subdir '{data_subdir}' does not exist. Valid choices are {SUBSEASONAL_DATA_SUBDIRS}.")
    # Get data path
    data_path = get_subseasonal_data_path()
    # Check azcopy is installed
    check_azcopy_install()
    # Copy or sync data
    data_subdir_path = os.path.join(
        data_path, data_subdir)
    filepath = os.path.join(data_subdir_path, filename)
    if not os.path.exists(filepath):
        cmd = "copy"
    else:
        cmd = "sync"
    # Run azcopy
    # Use Popen to access logs in real time
    azcopy_cmd = f"azcopy {cmd} {SUBSEASONAL_DATA_BLOB}/{data_subdir}/{filename} {filepath}"
    _subprocess_with_realtime_log(cmd=azcopy_cmd, verbose=verbose)
    if allow_write:
        try:
            os.chmod(filepath, 0o777)
        except Exception:
            warnings.warn(f'Changing file permissions of {filepath} failed.')


def get_subseasonal_data_path():
    """Get local path for doanloaded subseasonal data files.

    By default, the path is the :envvar:`$HOME`/:const:`subseasonal_data.download.DEFAULT_SUBSEASONAL_DATA_DIR`.

    You can change the default behavior by defining :envvar:`$SUBSEASONALDATA_PATH` as the target I/O folder.
    """
    # Look up data path and convert ~ to home directory
    data_path = expanduser(os.environ.get("SUBSEASONALDATA_PATH"))
    if not data_path:
        # Set default to user's home
        # Get home for local install
        data_path = os.path.join(expanduser("~"), DEFAULT_SUBSEASONAL_DATA_DIR)
    return data_path


def get_local_file_path(data_subdir, fname, sync=False, allow_write=False):
    """Get the local path of a directory/file combo.

    If ``sync=True``, it will also sync the target file.

    Parameters
    ----------
    data_subdir: {'dataframes', 'combined_dataframes', 'masks'}
        Azure data directory of target file.

    fname: string
        Name of target file.

    sync: bool, (default=True)
        Whether to download/sync the target file.

    allow_write: bool, (default=False)
        Whether to give write permissions to all users when syncing files.
        Recommended if working in shared directories. Users must be allowed
        to set permissions.
    """
    if sync:
        print("Syncing data....Set sync=False to avoid this step.")
        download_file(data_subdir, fname, verbose=True,
                      allow_write=allow_write)
    data_path = get_subseasonal_data_path()
    return os.path.join(data_path, data_subdir, fname)


def check_azcopy_install():
    """Check ``azcopy`` is installed correctly.

    Raises :exc:`~subprocess.CalledProcessError` if ``azcopy`` is not installed correctly.
    """
    try:
        s = subprocess.check_output("azcopy", shell=True)
    except CalledProcessError as e:
        print("An error has occured while calling 'azcopy'. "
              "Try first installing 'azcopy' from https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy.")
        raise e


def list_subdir_files(data_subdir):
    """List files in a data subdirectory in Azure.

    Requires ``azcopy`` to run.

    Parameters
    ----------
    data_subdir: {'dataframes', 'combined_dataframes', 'masks'}
        Azure data directory of target file.
    """
    check_azcopy_install()
    azcopy_cmd = f"azcopy list {SUBSEASONAL_DATA_BLOB}/{data_subdir}"
    _subprocess_with_realtime_log(cmd=azcopy_cmd)


def _subprocess_with_realtime_log(cmd, verbose=True):
    """Run subprocess with realtime log."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    # Reroute log
    if verbose:
        for c in iter(lambda: p.stdout.read(1), b''):
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout.buffer.write(c)
    # Parse errors
    _, stderr = p.communicate()
    if p.returncode != 0 or stderr:
        raise CalledProcessError(
            returncode=p.returncode, cmd=cmd, output=stderr)
