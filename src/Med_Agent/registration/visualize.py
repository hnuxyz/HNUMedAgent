import matplotlib.pyplot as plt
import numpy as np


def plot_slice(moving, fixed, flow, axis=0):

    mid = moving.shape[axis] // 2

    if axis == 0:
        m = moving[mid,:,:]
        f = fixed[mid,:,:]
        d = flow[0,mid,:,:]

    elif axis == 1:
        m = moving[:,mid,:]
        f = fixed[:,mid,:]
        d = flow[1,:,mid,:]

    else:
        m = moving[:,:,mid]
        f = fixed[:,:,mid]
        d = flow[2,:,:,mid]

    fig = plt.figure(figsize=(9,3))

    plt.subplot(1,3,1)
    plt.imshow(m, cmap="gray")
    plt.title("moving")

    plt.subplot(1,3,2)
    plt.imshow(f, cmap="gray")
    plt.title("fixed")

    plt.subplot(1,3,3)
    plt.imshow(d, cmap="jet")
    plt.title("deformation")

    plt.tight_layout()

    return fig