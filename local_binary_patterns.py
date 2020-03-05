import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.color
import skimage.transform
import itertools
from scipy import interpolate
import urllib.request as request

def plot_neighbourhood(x, y, P, R):
    plt.scatter(x, y)
    plt.axis('square')
    plt.grid(True)
    plt.title('Circle with P={p} and R={r}'.format(p=P, r=R))
    plt.xticks(np.arange(-2, 3, 1,0))
    plt.yticks(np.arange(-2, 3, 1.0))
    plt.show()

def load_image(path, as_gray=False):
    return skimage.io.imread(path, as_gray=as_gray, plugin='pil')

def create_index(s_T):
    n_ones=np.sum(s_T)
    s_T_size=len(s_T)

    if 1 in s_T:
        first_one=list(s_T).index(1)
    else:
        first_one=-1
    if 0 in s_T:
        first_zero=list(s_T).index(0)
    else:
        first_zero=-1

    if n_ones==0:
        return 0
    elif n_ones==s_T_size:
        return s_T_size*(s_T_size-1)+1
    else:
        if first_one==0:
            rot_index=n_ones-first_zero
        else:
            rot_index=s_T_size-first_one
        return 1+(n_ones-1)*s_T_size+rot_index

# #testing
# img=load_image("resources/red_bricks.jpg", False)
# img=skimage.transform.rescale(img, scale=(1/2, 1/2), anti_aliasing=True, mode='reflect', multichannel=True)
# img_gray=skimage.color.rgb2gray(img)
# plt.imshow(img)
# plt.show()
# plt.imshow(img_gray)
# plt.show()
