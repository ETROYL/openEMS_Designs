import os

NUM_THREADS = int(os.environ.get("EM_THREADS", 4))
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)