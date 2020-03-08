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
    plt.xticks(np.arange(-2, 3, 1.0))
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

'''
local binary patterns

local binary patterns is a theoretically simple, yet efficient approach \
to gray scale and rotation invariant textur classification. \
'''

'''
circularly symmetrical neighbor set

a circularly symmetric neighbor set for a given pixel g_c is defined \
by the points with coordinates (i, j) that surround the central point \
on a circle of radius R, and number of elements P.
'''
def neighbourhood(P, R):
    x=np.arange(0, P)
    x=R*np.cos(2*np.pi*x/P)
    y=np.arange(0, P)
    y=-R*np.sin(2*np.pi*y/P)
    return x, y

# #testing
# R=2
# P=8
# x,y=neighbourhood(P, R)
# plot_neighbourhood(x, y, P, R)

'''
texture

it is the collection of pixels in a gray scale image
'''

'''
interpolation

when a neighbor is not located in the center of a pixel, that \
neighbor gray value should be calculated by interpolation. \
'''
def interpolate2d(gray_img, kind='cubic'):
    #should be a 2d image,checking
    assert gray_img.ndim==2
    h, w=gray_img.shape

    x=np.arange(0, w)
    y=np.arange(0, h)

    return interpolate.interp2d(x, y, gray_img, kind=kind)

def calculate_neighborhood_values(x, y, interpolation_function):
    gray_values=map(lambda pt: interpolation_function(*pt), zip(x, y))
    return np.fromiter(gray_values, float)

# #testing
# x0=400
# y0=400
# xp=x+x0
# yp=y+y0
# f=interpolate2d(img_gray, kind='cubic')
# print("\nNeighborhood gray values:\n", img_gray[y0-R: y0+R+1, x0-R: x0+R+1])
# print("\nNeighborhood interpolations:\n", calculate_neighborhood_values(xp, yp, f))
