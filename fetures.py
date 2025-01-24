# features.py
import matplotlib.pyplot as plt

def plot_image_channels(img):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(img[0])  # отображение канала R
    ax1.set_title("R")
    ax2.imshow(img[1])  # отображение канала G
    ax2.set_title("G")
    ax3.imshow(img[2])  # отображение канала B
    ax3.set_title("B")
    fig.show()