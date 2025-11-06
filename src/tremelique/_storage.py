from contextlib import contextmanager
import h5py

@contextmanager 
def open_store(path: str, mode: str = "r"):
        f = h5py.File(path, mode)
        try:
               yield f
        finally:
                f.close()
