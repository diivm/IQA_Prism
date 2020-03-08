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

'''
joint difference distribution

used to turn texture into joint difference. \
to calculate it, subtract the gray value of the central pixel to all neighbor set. \
the joint difference distribution is a highly discriminative texture operator.
'''
def joint_difference_distribution(gray_img, gc, x, y, interpolation_function):
    xc, yc=gc
    xp=xc+x
    yp=yc+y
    g_p=calculate_neighborhood_values(xp, yp, interpolation_function)
    g_c=interpolation_function(xc, yc)
    return np.round(g_p-g_c, 15)

# #testing
# print("The joint difference distribution is:\n", joint_difference_distribution(img_gray, (x0, y0), x, y, f))

'''
local binary pattern operator

LBP operator is by definition invariant against any monotonic transformation of the gray scale. 
'''
def binary_joint_distribution(gray_img, gc, x, y, interpolation_function):
    T=joint_difference_distribution(gray_img, gc, x, y, interpolation_function)
    return np.where(T>=0, 1, 0)
    
def LBP(gray_img, gc, x, y, interpolation_function):
    s=binary_joint_distribution(gray_img, gc, x, y, interpolation_function)
    p=np.arange(0, P)
    binomial_factor=2**p
    return np.sum(binomial_factor*s)

# #testing
# print('The binary joint distribution is:\n', binary_joint_distribution(img_gray, (x0, y0), x, y, f))
# print('LBP:\n', LBP(img_gray, (x0, y0), x, y, f))

'''
uniform local binary patterns

LBP is not a good discriminator. \
use a set of local binary patterns such that no of spatial transitions \
does not exceed 2. \
to each uniform pattern, a unique index is associated.
'''
def is_uniform(pattern):
    count=0
    for idx in range(len(pattern)-1):
        count+=pattern[idx]^pattern[idx+1]
        return ~(count>2)

def uniform_patterns(P):
    patterns=itertools.product([0, 1], repeat=P)
    u_patterns=[pattern for pattern in patterns if is_uniform(pattern)]
    return [''.join(str(ele) for ele in eles) for eles in u_patterns]

def LBP_uniform(gray_img, gc, x, y, interpolation_function, uniform_patterns):
    s=binary_joint_distribution(gray_img, gc, x, y, interpolation_function)
    pattern=''.join([str(ele) for ele in s])
    return create_index(s) if pattern in uniform_patterns else 2+P*(P-1)

# #testing
# u_patterns=uniform_patterns(P)
# s_T = binary_joint_distribution(img_gray, (x0, y0), x, y, f)
# print('Is {} a uniform pattern: {}\n'.format(s_T, is_uniform(s_T)))
# print('LBP_uniform:', LBP_uniform(img_gray, (x0, y0), x, y, f, u_patterns))
