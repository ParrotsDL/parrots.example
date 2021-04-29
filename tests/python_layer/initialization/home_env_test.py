import os
print("Begin testing......")
os.environ.pop("HOME")
try:
    import torch
    print("Congratulations! Test pass!")
except Exception:
    print("Unfortunately! Test fail, cannot import parrots without “HOME” envrionment variable.")
