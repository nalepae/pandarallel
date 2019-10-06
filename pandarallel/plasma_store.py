import tempfile
import os
import pyarrow.plasma as plasma
import subprocess

PREFIX_TEMPFILE = "pandarallel-"
PLASMA_DIR = os.path.dirname(plasma.__file__)


def start_plasma_store(plasma_store_memory, verbose=True):
    tmpdir = tempfile.mkdtemp(prefix=PREFIX_TEMPFILE)

    plasma_store_name = os.path.join(tmpdir, "plasma_sock")

    # Pyarrow version > 0.14
    plasma_store_executable = os.path.join(PLASMA_DIR, "plasma-store-server")

    if not os.path.exists(plasma_store_executable):
        # Pyarrow version <= 0.14
        plasma_store_executable = os.path.join(PLASMA_DIR,
                                               "plasma_store_server")

    command = [plasma_store_executable,
               "-s", plasma_store_name,
               "-m", str(plasma_store_memory)]

    stdout = stderr = None if verbose else subprocess.DEVNULL

    proc = subprocess.Popen(command, stdout=stdout, stderr=stderr)

    return plasma_store_name, proc
