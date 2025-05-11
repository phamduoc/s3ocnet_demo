import matplotlib.pyplot as plt
import numpy as np
import torchvision

def show_images(images, labels, classes):
    grid = torchvision.utils.make_grid(images)
    npimg = grid.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title("Ground truth: " + " ".join(f"{classes[int(label)]}" for label in labels))
    plt.axis("off")
    plt.show()
