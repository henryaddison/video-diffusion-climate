import sys
from matplotlib import pyplot as plt
import numpy as np
import re


def extract_loss_values(file_name):
    loss_values = []
    with open(file_name, "r") as f:
        for line in f:
            match = re.match(r'(\d+): (\d+\.\d+)', line)
            if match:
                loss_values.append(float(match.group(2)))
    return loss_values


def plot_loss_curve(loss_values, window_size=1000):
    ma = np.convolve(
        loss_values,
        np.ones(window_size) / window_size,
        mode="valid"
    )
    ma_start = window_size // 2
    ma_end = len(loss_values) - window_size // 2 + 1
    # plt.plot(loss_values, label="Loss")
    plt.plot(range(ma_start, ma_end), ma, label="Moving Average")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")


if __name__ == "__main__":
    file_name = sys.argv[1]
    window_size = 1000
    if len(sys.argv) > 2:
        window_size = int(sys.argv[2])
    loss_values = extract_loss_values(file_name)
    plot_loss_curve(loss_values, window_size)
