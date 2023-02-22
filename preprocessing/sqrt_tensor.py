import torch
import sys
import os

if __name__ == "__main__":
    filename = sys.argv[1]
    new_filename = os.path.splitext(filename)[0] + "_sqrt.pt"
    x = torch.load(filename)
    x = torch.sqrt(x)
    torch.save(x, new_filename)
