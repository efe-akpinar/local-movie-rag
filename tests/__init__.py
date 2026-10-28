import os
import sys

if sys.platform == "darwin":
    # PyTorch and faiss-cpu each bundle their own libomp on macOS; two OpenMP runtimes in one process
    # abort or deadlock. Keep local test runs stable; Linux (CI, Docker) is not affected.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
