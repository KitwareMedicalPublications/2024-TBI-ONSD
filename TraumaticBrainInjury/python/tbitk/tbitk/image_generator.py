
import itk
import numpy as np
import scipy
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
import numpy as np
from skimage.draw import circle, polygon, ellipse, disk, polygon2mask
from tbitk.util import overwrite_mask

class IncrementingName:
    def __init__(self, outdir, image_prefix, image_suffix, label_prefix, label_suffix):
        '''
        Path(self.image_prefix2 + str(self.num) + self.image_suffix), Path(self.label_prefix2 + str(self.num) + self.label_suffix)
        
        
        Parameters
        ==========
        outdir
        image_prefix
        image_suffix
        
        '''
        self.num = 0
        self.outdir = outdir
        self.image_prefix2 = outdir + "/" + image_prefix
        self.image_suffix = image_suffix
        self.label_prefix2 = outdir + "/" + label_prefix
        self.label_suffix = label_suffix
    
    def get_next(self):
        '''
        Return next integer-labeled filepath image/label tuple.
        
        Returns
        =======
        (Path(), Path())
        '''
        ans = Path(self.image_prefix2 + str(self.num) + self.image_suffix), Path(self.label_prefix2 + str(self.num) + self.label_suffix)
        self.num += 1
        return ans

def uniform_bounded_points(bounds, size):
    '''
    Returns uniformly distributed points within shape.  Dimension of points is len(shape).
    
    Parameters
    ==========
    bounds (list/tuple of ndarray) - ([x1min, x1max], [x2min, x2max], ...)
    size - number of points to sample 
    
    Returns
    ==========
    len(bounds) x size array
    '''
    ans = None
    for i in range(len(bounds)):
        x = np.expand_dims(uniform.rvs(loc=bounds[i][0], scale=bounds[i][1] - bounds[i][0], size=size), axis=-1)
        ans = x if ans is None else np.concatenate((ans, x), axis=-1)
        
    return ans

def centered_rotation(pts, center, theta):
    '''
    2D-centered rotation
    
    Parameters
    ==========
    pts(nd.array) : Nx2
    '''
    x = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    t1 = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
    t2 = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    t3 = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
    y = t3 @ t2 @ t1 @ x.T
    return y.T[:, 0:2]

def uniform_random_disk(dim, rbounds):
    '''
    Returns disk mask image, location, and radius
    
    Parameters
    ==========
    shape (np.array) : size image for mask and to keep disk inside
    rbounds (np.array) : [rmin, rmax] uniform distibution
    '''
    shape = np.flip(dim)
    r = uniform.rvs(loc=rbounds[0], scale=rbounds[1]-rbounds[0], size=1)[0]
    xy = uniform_bounded_points(([r, dim[0]-1-r], [r, dim[1]-1-r]), size=1).squeeze()
    yy,xx = disk(np.flip(xy), r, shape=shape)
    npimg = np.zeros(shape)
    npimg[yy,xx] = 1
    return npimg, xy, r
    

def uniform_random_rectangle(dim, wbounds, hbounds, thetabounds):
    shape = np.flip(dim)
    w = uniform.rvs(loc=wbounds[0], scale=wbounds[1]-wbounds[0], size=1)[0]
    h = uniform.rvs(loc=hbounds[0], scale=hbounds[1]-hbounds[0], size=1)[0]
    theta = uniform.rvs(loc=thetabounds[0], scale=thetabounds[1]-thetabounds[0], size=1)[0]
    pts = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    pts = centered_rotation(pts, np.array([w/2.0, h/2.0]), theta)
    w2 = np.max(pts[:,0]) - np.min(pts[:,0])
    h2 = np.max(pts[:,1]) - np.min(pts[:,1])
    xy = uniform_bounded_points(([w2/2.0, dim[0] - w2/2.0], [h2/2.0, dim[1] - h2/2.0]), 1).squeeze()
    pts = pts + xy
    npimg = polygon2mask(shape, np.fliplr(pts))
   
    return npimg, pts, w, h, theta

# based on Perreualt, Auclair-Fortier 2007
def sample(npimg, n, m, theta, y0, dmin, dmax):
    assert False, 'Not implemented'
    pass

def interpolate(npimg, n, m, theta, y0, dmin, dmax):
    assert False, 'Not implemented'
    pass

def speckle_noise(npimg, a=0, b=10, alpha=0.5):
    '''
    Generates speckle noise on npimg.  TODO: there's issues with this.  Not sure how to scale the noise
    to images larger that 96 x 96 unless I implement some sort of upsampling scheme.  It also doesn't handle
    magnitude well - so the magnitude of the noise is correlated with alpha and can get nuts (like adding 1000 for an intensity value.)
    
    Parameters
    ==========
    npimg (ndarray) : 2D image
    a = minimum number of phasors per pixel
    b = maximum number of phasors per pixel
    alpha = normal distr variance
    
    based on Perreualt, Auclair-Fortier 2007
    '''
    ans = np.zeros(npimg.shape)
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            m = int(np.round(uniform.rvs(loc=a, scale=b, size=1))[0])
            x_re = npimg[i,j] + np.sum(norm.rvs(loc=0, scale=alpha, size=m))
            x_im = np.sum(norm.rvs(loc=0, scale=alpha, size=m))
            ans[i,j] = x_re**2 + x_im**2
    return ans