import os
import sys

# CRITICAL FIX for Windows OpenMP / PyTorch / Streamlit crashes
# This MUST be set before ANY python process loads numpy, faiss, or torch.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# By importing these here, we force the C++ runtimes to initialize on the 
# absolute MAIN thread of the process, BEFORE Streamlit starts any worker threads.
import torch
import faiss
try:
    import FlagEmbedding
except ImportError:
    pass

from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())
