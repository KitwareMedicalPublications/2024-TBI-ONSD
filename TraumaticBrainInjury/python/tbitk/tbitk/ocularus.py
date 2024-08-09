from dataclasses import dataclass
from typing import List
from scipy.ndimage import binary_hit_or_miss
import numpy as np
import scipy.interpolate
import skimage.measure
import skvideo
import skvideo.io
from collections import defaultdict
from abc import ABC, abstractmethod
from scipy import integrate
import math
import itk
import pickle
import os.path as path
import matplotlib.pyplot as plt
from datetime import datetime
import os
from IPython.core.debugger import set_trace
import zipfile
import tempfile
import sys

from tbitk.util import transform_to_indices, transform_to_physical, get_pts_from_binary_mask, polygon_from_array, image_from_spatial

'''
Module: ocularus (Ocular Ultrasound)

Notes
-----
Be aware there are 4 coordinate systems.  1. ITK's physical coordinate system.  2. ITK's index coordinate system.  3.  Numpy's index coordinate system.  4.  pyplot's
coordinate system.  ITK's index coordinate system (#2) is the raw video coordinate system, even if we crop the image.

'''

# Abstractions
# Identify optic nerve
    # Identify optic nerve width
# Identify eye
    # Identify eye center, eye socket, etc
# will need
# all nerves from video
# all eyes from video
# stuff?

# first cludgy part, handle clarius dimension and cropping

Dimension = 2
InputPixelType = itk.F
ImageType = itk.Image[InputPixelType, Dimension]

class CropPreprocess:
    def __init__(self, ref, percentage_x, percentage_y):
        '''
        Crops any black rows/columns (i.e. letterboxing) and a fixed percentage of the image

        Parameters
        ----------
        ref (itk.Image): reference image to define crop
        percentage_x (real): percentage to remove from either side (2.5% * 2 = 5% removal post-letterbox removal)
        percentage_y (real):
        '''
        npimg = itk.array_from_image(ref)
        nz_cols = np.nonzero(np.amax(npimg,axis=0))[0] # only 1 dimension returned
        nz_sc = nz_cols[0]
        nz_ec = nz_cols[-1]
        padc = round((nz_ec-nz_sc)*percentage_x)

        nz_rows = np.nonzero(np.amax(npimg,axis=1))[0]
        nz_sr = nz_rows[0]
        nz_er = nz_rows[-1]
        padr = round((nz_er-nz_sr)*percentage_y)


        self.crop_region = itk.ImageRegion[2]()
        self.crop_region.SetIndex(0, int(nz_sc+padc))
        self.crop_region.SetIndex(1, int(nz_sr+padr))

        self.crop_region.SetSize(0, int(nz_ec-nz_sc -2*padc + 1))
        self.crop_region.SetSize(1, int(nz_er-nz_sr -2*padr + 1))

        # it's confusing to me whether to use this or RegionOfInterestFilter
        # we'll see what the index consequences are when bouncing between this and numpy arrays
        # annotations will be specified in indices of the uncropped image
        # so how do we deal with that?

        # OK, ExtractImage filter and the index being weird is too difficult to track.  I'll normalize later.
        self.filter = itk.ExtractImageFilter[ImageType, ImageType].New(ExtractionRegion=self.crop_region)
        #self.filter = itk.RegionOfInterestImageFilter[ImageType, ImageType].New(RegionOfInterest=self.crop_region)

    def crop(self, image):
        '''
        Maintains image origin and spacing but will create a new image with a different LargestPossibleRegion covering the
        cropped region.
        '''
        self.filter.SetInput(image)
        self.filter.Update()
        return self.filter.GetOutput()

def image_from_array(array, ref, ttype=None):
    '''
    TODO maybe deprecate this once xarray support has stabilized.
    Returns an ITK image from a numpy array with the correct metadata.

    TODO consider this with indexing (i.e. the extractimagefilter effect on ImageRegion.Index)
    This doesn't work that great.  The index of the region isn't maintained.

    Parameters
    ----------
    array (np.ndarray): array to convert
    ref (itk.Image): reference image to copy spacing and coordinates from
    '''
    ans = itk.image_from_array(array, ttype=ttype)
    ans.SetOrigin(ref.GetOrigin())
    ans.SetSpacing(ref.GetSpacing())
    ans.SetDirection(ref.GetDirection())
    ans.SetLargestPossibleRegion(ref.GetLargestPossibleRegion())
    return ans

# def transform_to_physical(indices, image):
#     '''
#     Transform [y,x] indices to physical locations in an image.  Note, this is not the same as ITK's index scheme.
#     '''
#     start_index = np.asarray(image.GetLargestPossibleRegion().GetIndex())
#     return np.apply_along_axis(lambda x: np.array(image.TransformIndexToPhysicalPoint(wrap_itk_index(x))), 1, np.fliplr(indices) + start_index[np.newaxis,:])

# def transform_to_indices(pts, image):
#     # TODO find usage of this and figure out if I want to get rid of it
#     '''
#     Transform ITK's physical locations to [y,x] indices.  Note, this is not the same as ITK's index scheme.
#     '''
#     start_index = np.asarray(image.GetLargestPossibleRegion().GetIndex())
#     return np.fliplr(np.apply_along_axis(lambda x: np.array(image.TransformPhysicalPointToIndex(wrap_itk_point(x))), 1, pts) - start_index[np.newaxis,:])

def normalize_angle(t):
    '''
    Transform t to [0, 2pi]
    '''
    ans = t
    if np.abs(ans) > 2*np.pi:
        ans = np.fmod(ans, 2*np.pi)
    if t < 0:
        ans = 2*np.pi + ans
    return ans

def wrap_itk_index(x):
        idx = itk.Index[2]()
        idx.SetElement(0, int(x[0]))
        idx.SetElement(1, int(x[1]))
        return idx

def wrap_itk_point(x):
    # TODO, why itk.F?
    pt = itk.Point[itk.F,2]()
    pt.SetElement(0, x[0])
    pt.SetElement(1, x[1])
    return pt

def arclength(theta1, theta2, a, b):
    '''
    Returns arc length of ellipse perimeter from theta1 to theta2.  To get the length in the other direction reverse the assignments of theta1 and theta2.

    Parameters
    ---------
    theta1 (float):
    theta2 (float):
    a: major axis (1/2 ellipse width)
    b: minor axis (1/2 ellipse height)

    Returns
    -------
    arc length (float):  i.e. Integral[theta2, theta1]
    '''
    #assert not reverse, "reverse=True not implemented"
    assert 0 <= theta1 and theta1 <= 2*np.pi and 0 <= theta2 and theta2 <= 2*np.pi, "0 <= theta1, theta2, <= 2*np.pi"

    def foo(theta, a, b):
        return a*np.sqrt(1 - (1 - (b/a)**2)*np.sin(theta)**2)

    if theta1 == theta2:
        return 0
    elif theta1 < theta2:
        return integrate.quad(foo, theta1, theta2, args=(a, b))[0]
    else: # theta1 > theta2
        return integrate.quad(foo, theta1, 2*np.pi, args=(a,b))[0] + integrate.quad(foo, 0, theta2, args=(a,b))[0]

class Curve:
    '''
    Represents a curve from a list of nodes

    Parameters
    -------
    nodes : List of Node
        List of nodes comprising the curve before downsampling
    downsample : int
        Factor to downsample the list of nodes by. One out of every
        `downsample` nodes will be kept in the list
    extrapolatesample : int, optional
        Number of samples to average when calculating
        extrapolation at endpoints

    Attributes
    -------
    nodes : List of Node
        List of nodes comprising the curve before downsampling
    downsample : int
        Factor to downsample the list of nodes by. One out of every
        `downsample` nodes will be kept in the list
    extrapolatesample : int
        Number of samples to average when calculating
        extrapolation at endpoints
    spline_points : np.ndarray of itk.Point
        The points to fit a spline from
    deriv1_1 : np.ndarray of float
        Numpy array containing the x and y value of the derivative at the top endpoint
    deriv1_2 : np.ndarray of float
        Numpy array containing the x and y value of the derivative at the bottom endpoint
    length : float
        Length of the curve
    spline : scipy.interpolate.PPoly
        Fitted spline
    '''
    def __init__(self, nodes, downsample, extrapolatesample=2):
        self.nodes = nodes
        self.downsample = downsample
        self.extrapolatesample = extrapolatesample

        # guh, there's probably a better way to do the subsampling, this will add random asymmetry in the sampling distance
        self.spline_points = np.array([x.point for x in self.nodes[::downsample]])
        if not np.array_equal(self.spline_points[-1,:], np.array(self.nodes[-1].point)): # make sure we keep 0 and last points no matter downsample
            self.spline_points = np.vstack([self.spline_points, self.nodes[-1].point])

        # diffs is difference vectors between points
        # norms is the magnitudes of those vectors
        # want to average the diffs at the beginning and end of that curve, then just use that 1st derivative vector as our extrapolation vector
        diffs = np.vstack([[0,0], np.diff(self.spline_points, axis=0)])
        norms = np.linalg.norm(diffs, axis=1)
        norm_diffs = diffs / norms[:,np.newaxis]
        self.deriv1_1 = np.mean(-norm_diffs[1:(self.extrapolatesample+1),:], axis=0)

        deriv1_2_start_idx = np.max([-self.extrapolatesample, -len(norm_diffs) + 1])
        self.deriv1_2 = np.mean(norm_diffs[deriv1_2_start_idx::,:], axis=0)


        spline_t = np.cumsum(norms)
        self.length = spline_t[-1]
        self.spline = scipy.interpolate.CubicSpline(spline_t, self.spline_points)

    def reverse(self):
        '''
        Return a new reversed curve.  May affect spline near ends of curve, flips handedness of the derivative, flips order of vertices.
        '''
        return Curve(self.nodes[::-1], self.downsample, self.extrapolatesample)

    def evaluate(self, t):
        #if t.ndim == 1:
        #    t = t[:,np.newaxis]
        ans = np.zeros([len(t), 2])

        idx = (0 <= t) & (t <= self.length)
        ans[idx,:] = self.spline(t[idx])

        idx = (t < 0)
        ans[idx,:] = (self.deriv1_1[:,np.newaxis]*(-t[idx]) + self.spline_points[0,np.newaxis].T).T

        idx = (t > self.length)
        ans[idx,:] = (self.deriv1_2[:,np.newaxis]*(t[idx]-self.length) + self.spline_points[-1,np.newaxis].T).T
        return ans

    def normal(self, t):
        ans = np.zeros([len(t), 2])

        idx = (0 <= t) & (t <= self.length)
        tmp = self.spline.derivative(1)(t[idx])
        ans[idx,0] = tmp[:,1]
        ans[idx,1] = -tmp[:,0]
        idx = (t < 0)
        ans[idx,:] = np.array([-self.deriv1_1[1], self.deriv1_1[0]])

        idx = (t > self.length)
        ans[idx,:] = np.array([self.deriv1_2[1], -self.deriv1_2[0]])

        return ans

    @property
    def vertices(self):
        return (self.nodes[0], self.nodes[-1])

class Node:
    '''

    '''
    def __init__(self, index, point, connectivity=None, neighbors=None):
        self.index = index
        self.point = point
        #self.connectivity = connectivity
        self.neighbors = neighbors if neighbors is not None else set()
        self.is_vertex = False

    def __eq__(self, other):
        return self.index[0] == other.index[0] and self.index[1] == other.index[1]

    @property
    def connectivity(self):
        return len(self.neighbors)

    def get_other_neighbors(self, source):
        return [x for x in self.neighbors if x != source]

    def __hash__(self):
        return id(self)

    def __str__(self):
        return '({},{}), connectivity: {}, neighbors: {}'.format(self.index[0], self.index[1], self.connectivity, len(self.neighbors))


class CurveGraphFactory:
    '''
    Generates a CurveGraph from a skeletonized image

    Parameters
    -------
    itkimage : itk.Image[,2]
        Skeleton image to generate the curvegraph from
    curve_downsample : int
        Factor to downsample the list of nodes by. See `Curve.downsample`

    Attributes
    -------
    nodes : dict of tuple[2] to Node
        Dictionary mapping x, y positions in the image to node objects
    itkimage : itk.Image[,2]
        Skeleton image to generate the curvegraph from
    image : np.ndarray
        `itkimage` converted to a numpy array
    curve_downsample : int
        Factor to downsample the list of nodes by. See `Curve.downsample`
    '''
    R1_NEIGHBORHOOD_CIRC = np.array([[-1, 0, 1, 1, 1, 0, -1, -1, -1], [-1, -1, -1, 0, 1, 1, 1, 0, -1]]).T # clockwise starting upper left and overlap with begin == end
    R1_NEIGHBORHOOD = np.array([[-1, 0, 1, 1, 1, 0, -1, -1], [-1, -1, -1, 0, 1, 1, 1, 0]]).T # clockwise starting upper left
    VISITED = np.array([[-1, -1, 0, 1], [0, -1, -1, -1]]).T
    CORNER1 = (np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]]), np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]))
    CORNER2 = (np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]), np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]))
    CORNER3 = (np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]), np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]))
    CORNER4 = (np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]]))

    def __init__(self, itkimage, curve_downsample):
        self.nodes = dict()
        self.itkimage = itkimage
        self.image = itk.array_from_image(itkimage)
        self.curve_downsample = curve_downsample

        # cheap way of handling border, TODO: REMOVE
        self.image[0,:] = 0
        self.image[:,0] = 0
        self.image[self.image.shape[0]-1,:] = 0
        self.image[:,self.image.shape[1]-1] = 0

        self.to_connect8(self.image) # preprocess the binary thinned image

        self.pass1(self.image) # create nodes with connected neighbors

        self.merge_adjacent_junctions() # combine adjacent junctions to avoid weird edge cases

        # not sure if the binary thinning allows this, but we can't do a one-point curve
        salt = {n for n in self.nodes.values() if n.connectivity == 0}
        for n in salt:
            self.nodes.pop((n.index[0], n.index[1]))

        edges = {n for n in self.nodes.values() if n.connectivity == 2}
        vertices = {n for n in self.nodes.values() if n.connectivity != 2}
        self.pass2(edges, vertices) # break loops, trace from all end points and junctions, create curves

    def set_connectivity(self, image, n):
        #y = CurveGraphFactory.R1_NEIGHBORHOOD + n.index
        #n.connectivity = np.sum(image[y[:,1], y[:,0]])

        z = CurveGraphFactory.VISITED + n.index
        q = image[z[:,1], z[:,0]] > 0
        r = z[q,:]
        for i in range(r.shape[0]):
            k = self.nodes[(r[i,0], r[i,1])]
            n.neighbors.add(k)
            k.neighbors.add(n)

#     def set_connectivity4(self, image, n):
#         '''
#         This is an untested connectivity measure for 4-connected
#         '''
#         y = CurveGraphFactory.R1_NEIGHBORHOOD_CIRC + n.index
#         n.connectivity = np.sum(np.abs(np.diff(image[y[:,1], y[:,0]])))/2

#         z = CurveGraphFactory.VISITED + n.index
#         q = image[z[:,1], z[:,0]] > 0
#         r = z[q,:]
#         for i in range(r.shape[0]):
#             k = self.nodes[(r[i,0], r[i,1])]
#             n.neighbors.add(k)
#             k.neighbors.add(n)


    def to_connect8(self, image):
        self.image = image
        self.image = self.image - binary_hit_or_miss(self.image, CurveGraphFactory.CORNER1[0], CurveGraphFactory.CORNER1[1])
        self.image = self.image - binary_hit_or_miss(self.image, CurveGraphFactory.CORNER2[0], CurveGraphFactory.CORNER2[1])
        self.image = self.image - binary_hit_or_miss(self.image, CurveGraphFactory.CORNER3[0], CurveGraphFactory.CORNER3[1])
        self.image = self.image - binary_hit_or_miss(self.image, CurveGraphFactory.CORNER4[0], CurveGraphFactory.CORNER4[1])

    def pass1(self, image):
        '''
        Computes nodes and their connectivity (first guess)
        '''
        for y in range(1, image.shape[0]-1):
            for x in range(1, image.shape[1]-1):
                if image[y, x] > 0:
                    index = np.array([x, y]) # i think this is the only place where the index is read
#                     itkindex = itk.Index[2]()
#                     itkindex.SetElement(0, int(index[0]))
#                     itkindex.SetElement(1, int(index[1]))
                    n = Node(index, transform_to_physical(index[np.newaxis,::-1], self.itkimage).flatten())
                    self.nodes[(x, y)] = n
                    self.set_connectivity(image, n)

    def pass2(self, edges, vertices):
        self.curvegraph = CurveGraph()
        while len(vertices) > 0:
            v = vertices.pop()
            for x2 in v.neighbors:
                if x2 in edges: # important check, for example, junction with a loop, or the other end of the curve has already been visited
                    nodes = [v]
                    x1 = v
                    while x2.connectivity == 2 and x2 != v:
                        edges.remove(x2)
                        nodes.append(x2)
                        tmp = x2.get_other_neighbors(x1)[0] # know there's only 1 cuz of connectivity constraint
                        x1 = x2
                        x2 = tmp
                    if x2 != v:
                        nodes.append(x2)
                    else: # x2 == v, we have a cycle and we'll break it
                        x1.neighbors.remove(x2)
                    curve = Curve(nodes, self.curve_downsample)
                    self.curvegraph.add(curve)
        if len(edges) > 0: # we have a free loop, break it
            e1 = edges.pop()
            e2 = e1.neighbors.pop() # get arbitraty neighbor and remove e1's connection
            e2.neighbors.remove(e1) # remove e2's link to e1
            vertices.add(e1)
            vertices.add(e2)
            self.pass2(edges, vertices) # in case we have more than one loop



    def merge_adjacent_junctions(self):
        '''
        Merge any adjacent junctions into one junction.

        Imagine a T-junction.  Left, right, and middle nodes all think they are 3-junctions due to 8-connectedness.  Merge these into one junction.
        '''
        merged = True
        while merged:
            merged = False
            junctions = {n for n in self.nodes.values() if n.connectivity > 2}
            for j in junctions:
                for n in j.neighbors:
                    if n.connectivity > 2:
                        self.combine_with(n, j)
                        merged = True
                        break



    def combine_with(self, target, source):
        '''
        Connects all of source's neighbors with target and then removes source from the self.nodes
        '''
        for n in source.neighbors:
            n.neighbors.remove(source)
            if n != target:
                n.neighbors.add(target)
                target.neighbors.add(n)
        self.nodes.pop((source.index[0], source.index[1]))


    # consider loops
    # consider junctions
    # blergh
#     def first_pass(self):
#         for i
#         return

#     def second_pass(self):
#         return

class CurveGraph:
    # TODO: get vertex by index or point
    # algorithm:
    # mark all vertices as curves or junctions
    # group all vertices in a neighborhood
    # mark lines that only have one connection or are connected to a junction as an endpoint
    def __init__(self, vertices=None, adjacency_list=None):
        self.vertices = dict() if vertices is None else vertices
        self.adjacency_list = defaultdict(dict)
        self.curves = []
    def add(self, curve):
        for i in range(2):
            v = curve.vertices[i]
            idx = (v.index[0], v.index[1])
            if idx not in self.vertices:
                self.vertices[idx] = v
            self.adjacency_list[idx][curve.vertices[i-1]] = curve # the i-1 let's me swap between first and last element
        self.curves.append(curve)

    def create_curve_plot_on_image(self, image, ax=None, num_pts_per_curve=50):
        if ax is None:
            ax = plt.gca()
        ax.imshow(image, cmap="gray")
        for c in self.curves:
            s = 0
            e = c.length
            cs = np.linspace(s, e, num=num_pts_per_curve)
            pts_physical_space = c.evaluate(cs)
            pts_index_space = transform_to_indices(pts_physical_space, image)
            ax.plot(pts_index_space[:, 1], pts_index_space[:, 0])

        return ax

    def display_curve_plot_on_image(self, image, ax=None, num_pts_per_curve=50):
        self.create_curve_plot_on_image(image, ax, num_pts_per_curve).show()

# need closest point search
def nearest_curve(curvegraph, eye, debug_obj=None):
    '''
    Return the nearest curve in the correct orientation (closest vertex first) relative to eye model

    Parameters
    ----------
    curvegraph (ocularus.CurveGraph)
    eye (skimage.measure.EllipseModel) or numpy array
    debug_obj (ONSDDebugInfo)
    '''

    min_dist = np.Inf
    min_curve = None
    if isinstance(eye, np.ndarray) and eye.size == 0:
        return min_curve, min_dist

    for c in curvegraph.curves:
        v1 = np.asarray(c.vertices[0].point)
        v2 = np.asarray(c.vertices[1].point)

        if isinstance(eye, skimage.measure.EllipseModel):
            eye_pt_closest_v1 = eye.predict_xy(nearest_angle(v1, eye))
            eye_pt_closest_v2 = eye.predict_xy(nearest_angle(v2, eye))
            d1 = np.linalg.norm(eye_pt_closest_v1 - v1)
            d2 = np.linalg.norm(eye_pt_closest_v2 - v2)
        elif isinstance(eye, np.ndarray):
            distances_from_v1 = np.array([np.linalg.norm(pt - v1) for pt in eye])
            distances_from_v2 = np.array([np.linalg.norm(pt - v2) for pt in eye])

            index_min_d1 = np.argmin(distances_from_v1)
            index_min_d2 = np.argmin(distances_from_v2)

            d1 = distances_from_v1[index_min_d1]
            d2 = distances_from_v2[index_min_d2]

            eye_pt_closest_v1 = eye[index_min_d1]
            eye_pt_closest_v2 = eye[index_min_d2]

        if d1 < min_dist:
            min_dist = d1
            min_curve = c
            if debug_obj:
                debug_obj.closest_eye_pt = eye_pt_closest_v1
        if d2 < min_dist:
            min_dist = d2
            min_curve = c.reverse()
            if debug_obj:
                debug_obj.closest_eye_pt = eye_pt_closest_v2
    return (min_curve, min_dist)


def linspace_ellipse(model, theta1, theta2, arc_step, reverse=False, fast=True):
    '''
    Returns regularly sampled thetas from theta1 to theta2 with an arc_step arc length spacing between them.

    Parameters
    ----------
    model (EllipseModel): really only need a, b from the model
    theta1 (float):
    theta2 (float):
    arc_step (float): distance between points (> 0)

    Returns
    -------
    (np.array) : [theta1, x2, x3, ..., xn] where arclength(xn, theta2) < arc_step and 0 <= xi <= 2PI
    '''

    assert arc_step > 0

    a, b = model.params[2:4]
    thetac = theta1
    if not fast:
        ans = []
        while np.abs(thetac - theta2) > arc_step:
            ans.append(thetac)
            thetan = normalize_angle(scipy.optimize.minimize_scalar(lambda x: (arclength(thetac, normalize_angle(x), a, b) - arc_step)**2, method='Brent').x)
            thetac = thetan
        return np.array(ans)
            #print(thetac)
    else:
        arc_len = arclength(theta1, theta2, a, b)
        ans = np.linspace(theta1, theta2, num=int(np.ceil(arc_len / arc_step)))
        return ans





def nearest_angle(x, model):
    '''
    Given a point returns the parametric angle of the nearest point on the ellipse.
    '''
    return normalize_angle(scipy.optimize.minimize_scalar(lambda t: np.linalg.norm(model.predict_xy(np.array([t])) - x), method='Brent').x)

def is_inside(pt, image):
    idx = image.TransformPhysicalPointToIndex(pt)
    return image.GetLargestPossibleRegion().IsInside(idx)

def resample_by_grid(grid, image):
    '''
    grid is a MxNx2
    '''
    interp = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
    interp.SetInputImage(image)
    ans = np.zeros([grid.shape[0], grid.shape[1]], dtype='float32')
    good = np.repeat(True, grid.shape[1])
    for c in range(ans.shape[1]):
        for r in range(ans.shape[0]):
            if is_inside(grid[r,c], image):
                pt = itk.Point[itk.D, 2](grid[r, c])
                ans[r, c] = interp.Evaluate(pt)
            else:
                good[c] = False
                break
    return ans[:,good], grid[:,good]
# inliers
#
# inliers appear to be presorted by x
# cut-off a few points on either end in case an outlier snuck in


def fig_to_array(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data



# Note that we use an approximation here.
def get_mask_from_ellipse(img, model):
    if model is None:
        return None
    ts = np.linspace(0, 2*np.pi, num=1000)
    eye_indices = transform_to_indices(model.predict_xy(ts), img)
    eye_indices = np.fliplr(eye_indices).astype('float64')
    eye_spatial = polygon_from_array(eye_indices)
    return image_from_spatial(eye_spatial, reference_image=img)


def create_eye_figure(img, eye, ax=None):
    model = eye.ellipse_model
    inliers = eye.eyesocket_points
    filtered_pts = eye.nerve_search_points
    nerve_seed_pt = eye.nerve_seed_point
    eye_seed_pt = eye.eye_seed_point

#     plt.ioff()
    if (ax is None):
        fig, ax = plt.subplots()
    else:
        fig = None
    ax.set_aspect('equal', 'box') # make sure this is displayed correctly
    ax.imshow(img, cmap='gray')
    img_xlim = plt.xlim()
    img_ylim = plt.ylim()
    if model is not None:
        ts = np.linspace(0, 2*np.pi, num=1000)
        pts2 = transform_to_indices(model.predict_xy(ts), img)
        ax.fill(pts2[::,1], pts2[::,0], facecolor='red', alpha=0.2, zorder=2)

        # print inliers
        pts3 = transform_to_indices(inliers, img)
        ax.scatter(pts3[:,1], pts3[:,0], color='pink', alpha=0.8)

        if filtered_pts is not None:
            tmp = transform_to_indices(np.array([filtered_pts[:,:,0].flatten(), filtered_pts[:,:,1].flatten()]).T, img)
            ax.scatter(tmp[:,1], tmp[:,0], color='green', alpha=0.01)

        #tmp4 = transform_to_indices(inliers, img)
        if nerve_seed_pt is not None:
            seed_idx = transform_to_indices(nerve_seed_pt[np.newaxis,:], img).flatten()
            ax.scatter(seed_idx[1], seed_idx[0], color='blue')

        if eye_seed_pt is not None:
            pts_indices = transform_to_indices([eye_seed_pt], img)[0]
            ax.scatter(pts_indices[1], pts_indices[0], color='cyan')

    plt.xlim(img_xlim)
    plt.ylim(img_ylim)
#     plt.ion()
    return fig



def find_ellipse_of_best_fit(img, edge_tolerance=None):
    '''
    Returns the ellipse of best fit from a binary image. Option to remove
    points close to the edges of the image
    '''
    binaryContourFilter = itk.BinaryContourImageFilter[type(img), type(img)].New()
    binaryContourFilter.SetInput(img)
    binaryContourFilter.SetForegroundValue(1)
    binaryContourFilter.SetBackgroundValue(0)
    binaryContourFilter.Update()
    binary_output = binaryContourFilter.GetOutput()
    npimg = itk.array_from_image(binary_output)
    indices = np.argwhere(npimg > 0)
    if indices.size == 0:
        return None, None

    if edge_tolerance is not None:
        # TODO: Do the same for the height?
        _, w = npimg.shape
        wmax = w - edge_tolerance
        wmin = 0 + edge_tolerance

        c_indices = indices[:, 1]
        # Exclude points that are within edge_tolerance pixels of the edge
        indices = indices[(wmin < c_indices) & (c_indices < wmax), :]
        if indices.size == 0:
            return None, None

    pts = transform_to_physical(indices, img)
    model = skimage.measure.EllipseModel()
    try:
        model_estimate_success = model.estimate(pts)
    except:
        return None, None

    if model_estimate_success:
        return model, pts

    return None, None

def overlay(img, objects, colors, alphas=None):
    '''
    Creates an overlay of objecs on top of img.  Sequential blending of pixels according to color and alpha value.

    Parameters
    ----------
    img (numpy 2D array): normalized pixel values 0 to 1
    objects (list of numpy 2D arrays): overlay labels the same size of img.  Non-zero values will be overlayed.
    colors (list of numpy 3-element arrays): RGB colors, each element 0 to 1
    alphas (list of alpha values or None=1):
    '''

#     imgrgb = np.stack((img, img, img), axis=2)
#     imgones = np.ones(imgrgb.shape)
#     imgzeros = np.zeros(imgrgb.shape)
    alphas = alphas if alphas is not None else np.ones(len(colors))

    imgflat = img.flatten()
    ansflat = np.stack((imgflat, imgflat, imgflat), axis=1)

    for i in range(len(objects)):
        obj = objects[i].flatten() > 0
        c = colors[i]

        overlay = np.zeros((obj.shape[0], 3))
        overlay[obj,:] = c

        ansflat[obj,:] = (1-alphas[i])*ansflat[obj,:] + alphas[i]*overlay[obj,:]

    return np.stack((ansflat[:,0].reshape(img.shape), ansflat[:,1].reshape(img.shape), ansflat[:,2].reshape(img.shape)), axis=2) # probably could do this with one reshape



    # combining colors = 0 0 1 and 1 0 0 to .5 0 .5
    # what operation is that?
    # just add then divide by norm?
    # flatten and split out channels?
    # is it additive or subtractive?
    # image is pure white [1 1 1] and overlay is red [1 0 0]
    # now what?  now i've blended and want [0.5 0 0.5], now what?
    # make 1 default and add to zeros, unmasked pixels are set to white at the end?
    # what about black?  no?  does the mask have to add to 1?

#     overr
#     overg
#     overb


    # ok, we are shifting the original image from white to color, so labeling is just a shift on the rgb
    # so, the label color sum = 1

    # maybe calculate intensity first, then weight by colors, then multiple by masks, then set pixels in image

def _map_nerve(nerve_mask, input_image, eye, nerve_offset=1,
               nerve_image_dimension=np.asarray([6, 12]), nerve_image_sampling=np.asarray([50, 100]),
               dilation_rad=3, debug_obj=None, curve_downsample=None):
    '''
    Generates a straightened nerve image (zoomed in on just the nerve),
    skeletonization of the nerve, and a corresponding CurveGraph.

    Parameters
    -------
    nerve_mask : itk.Image[,2]
        Mask image that is 1 only where the nerve is present
    input_image : itk.Image[, 2]
        Original input image
    eye : np.ndarray
        Array of points (represented by numpy arrays with size 2) specifying
        the outline of the eye
    nerve_offset : float, optional
        offset in mm from the closest eye point towards the nerve,
        where the nerve image will start. Defaults to 1
    nerve_image_dimension : np.array, optional
        Nerve image size in mm. Defaults to np.array([6, 12])
    nerve_image_sampling : np.ndarray, optional
        Nerve image size in pixels (width, height). Defaults to
        np.array([50, 100])
    dilation_rad : int, optional
        Dilation radius to apply to the nerve mask. Defaults to 3.
    debug_obj : ONSDDebugInfo, optional
        Optional debug object for this frame. Will populate the appropriate fields
        of `debug_obj` with the results from this step of the pipeline.
    curve_downsample : float, optional
        Factor to downsample the curves in the curvegraph by. Will sample
        one point from every `curve_downsample` mm for each curve
        in the curvegraph. Defaults to 1 mm.

    Returns
    -------
    itk.Image[,2]
        The straightened nerve image, sampled along and orthogonal to
        the medial axis
    CurveGraph
        Contains the curves comprising the skeletonization of the nerve mask
    itk.image[,2]
        Skeletonization of the nerve mask
    '''
    if curve_downsample is None:
        curve_downsample = 1

    # Dilate the nerve mask a bit
    # This removes corners and other small edge artifacts that would
    # effect the skeleton.
    im_dim = nerve_mask.GetImageDimension()
    StructuringElementType = itk.FlatStructuringElement[im_dim]
    structuringElement = StructuringElementType.Ball(dilation_rad)
    DilateFilterType = itk.BinaryDilateImageFilter[
        type(nerve_mask), type(nerve_mask), StructuringElementType
    ]

    dilate_filter = DilateFilterType.New(Input=nerve_mask,
        Kernel=structuringElement,
        ForegroundValue=1)


    binary = itk.BinaryThinningImageFilter[type(nerve_mask), type(nerve_mask)].New(Input=dilate_filter.GetOutput())
    binary.Update()
    binary_output = binary.GetOutput()

    # Convert curve_downsample to pixel space

    curve_downsample = round(curve_downsample / binary_output.GetSpacing()[1])
    cfg = CurveGraphFactory(binary_output, curve_downsample)

    # TODO
    c, dist = nearest_curve(cfg.curvegraph, eye, debug_obj=debug_obj)
    if debug_obj is not None:
        debug_obj.medial_curve = c
        debug_obj.nerve_offset = nerve_offset
        debug_obj.dist = dist
        debug_obj.nerve_image_dimension = nerve_image_dimension

    if c is None:
        nerve_image = None
    else:
        ts = np.linspace(-dist+nerve_offset,  nerve_image_dimension[0]-dist+nerve_offset, nerve_image_sampling[0]) # 6mm
        ys = c.evaluate(ts)
        zs = c.normal(ts)

        # resample everything, do the medial projection, wow numpy broadcasting!
        qs = np.linspace(-nerve_image_dimension[1]/2, nerve_image_dimension[1]/2, nerve_image_sampling[1])
        transform_pts = qs[:,np.newaxis,np.newaxis] * zs + ys[np.newaxis,:,:]

        nerve_image = itk.image_from_array(resample_by_grid_point(transform_pts, input_image))
        nerve_image.SetSpacing(nerve_image_dimension / nerve_image_sampling) # this is a bit false as this is really a distorted mesh

    return nerve_image, cfg.curvegraph, binary_output


def create_border_of_straightened_nerve_plot(nerve_image, ax=None, bottom_pts=None, top_pts=None, onsd_sample_position=None, mid=None,
                                            onsd=None, score=None, perc=None):
    '''
    Creates a plot of the straightened nerve image. Can optionally add
    the calculated border points, the beginning and end of the sample
    position range as vertical lines, and the middle of the nerve.

    Parameters
    ----------
    nerve_image : itk.Image[,2]
        The straightened nerve image
    ax : matplotlib.axes.Axes
        Optional axes to use
    bottom_pts : np.ndarray[2]
        numpy array of points (x and y) specifying bottom of the nerve in
        index space
    top_pts : np.ndarray[2]
        numpy array of points (x and y) specifying top of the nerve in
        index space
    onsd_sample_position : List[2]
        Lower and upper end of the range to sample the nerve sheath diameter
        in physical coordinates
    mid: int
        The middle of the nerve (default to the middle of the image)
    onsd: float
        onsd for this frame. Will be displayed in top right corner
    score: float
        score for this frame. Will be displayed in top right corner
    perc: float
        percentile for this frame. Will be displayed in top right corner

    Returns
    -------
    ax: matplotlib.axes.Axes
    '''

    if ax is None:
        ax = plt.gca()

    ax.axis("off")


    _, h = nerve_image.GetLargestPossibleRegion().GetSize()
    mid = h // 2

    if bottom_pts is not None and top_pts is not None and bottom_pts.size != top_pts.size:
        raise ValueError("Expected equal number of bottom and top nerve points")

    ax.imshow(nerve_image)
    if bottom_pts is not None:
        ax.scatter(x=bottom_pts[:, 0], y=bottom_pts[:, 1], c="orange", alpha=0.2)
    if top_pts is not None:
        ax.scatter(x=top_pts[:, 0], y=top_pts[:, 1], c="red", alpha=0.2)
    ax.axhline(y=mid)
    if onsd_sample_position is not None:
        x0 = onsd_sample_position[0]
        x1 = onsd_sample_position[1]
        spacing = nerve_image.GetSpacing()
        c0 = int(x0 / spacing[0])
        c1 = int(x1 / spacing[0])
        ax.axvline(x=c0, c="yellow")
        ax.axvline(x=c1, c="yellow")

        if bottom_pts is not None and top_pts is not None:
            assert (bottom_pts[:, 0] == top_pts[:, 0]).all()

            xs_phys = bottom_pts[:, 0] * spacing[0]
            idxs = np.argwhere((x0  < xs_phys) & (xs_phys < x1))
            av_y_bottom = np.mean(bottom_pts[idxs, 1])
            av_y_top = np.mean(top_pts[idxs, 1])
            if onsd is not None:
                recalculated_onsd = (av_y_bottom - av_y_top) * spacing[1]
                np.testing.assert_almost_equal(recalculated_onsd, onsd)
            ax.axhline(y=av_y_bottom, c="white")
            ax.axhline(y=av_y_top, c="white")

    text = ""
    for k, v in {"onsd": onsd, "score": score, "perc": perc}.items():
        if v is not None:
            text += f"{k} = {v:.3f}\n"
    if text:
        ax.text(0, 8, text, backgroundcolor="white")
    return ax

def plot_border_of_straightened_nerve_plot(nerve_image, bottom_pts=None, top_pts=None, onsd_sample_position=None, mid=None):
    '''
    Calls create_border_of_straightened_nerve_plot and displays the resulting image.

    Parameters
    ----------
    nerve_image : itk.Image[,2]
        The straightened nerve image
    bottom_pts : np.ndarray[2]
        numpy array of points (x and y) specifying bottom of the nerve in
        index space
    top_pts : np.ndarray[2]
        numpy array of points (x and y) specifying top of the nerve in
        index space
    onsd_sample_position : List[2]
        Lower and upper end of the range to sample the nerve sheath diameter
        in physical coordinates
    mid: int
        The middle of the nerve (default to the middle of the image)

    Returns
    -------
    None
    '''

    # Just call the above function and show the result.
    axes = plt.gca()
    create_border_of_straightened_nerve_plot(nerve_image, axes, bottom_pts, top_pts, onsd_sample_position, mid)
    plt.show()


def _find_top_and_bottom_of_nerve_grad(nerve_grad, mid):
    top = np.argmin(nerve_grad[0:mid,:], axis=0)
    bottom = mid + np.argmax(nerve_grad[mid::, :], axis=0)
    xindices = np.arange(0, nerve_grad.shape[1])

    return xindices, top, bottom

def _find_top_and_bottom_nerve_width_pf_middle_rectangles(nerve_grad, blurred_im, mid, grad_frac=0.75):
    top = []
    bottom = []
    spacing = blurred_im.GetSpacing()
    blurred_np = itk.array_from_image(blurred_im)

    sample_rect_width = int(0.25 / spacing[1])
    val_outside_nerve_top = np.mean(blurred_np[:sample_rect_width, :])
    val_outside_nerve_bottom = np.mean(blurred_np[-sample_rect_width:, :])
    # Take intensity samples 0.25mm around the middle of the nerve
    mid_sample_start_index = mid - sample_rect_width // 2
    mid_sample_stop_index = mid + sample_rect_width // 2
    val_inside_nerve = np.mean(blurred_np[mid_sample_start_index:mid_sample_stop_index, :])

    intensity_diff_top = val_outside_nerve_top - val_inside_nerve
    intensity_diff_bottom = val_outside_nerve_bottom - val_inside_nerve
    xindices = []
    top_prom_threshold = grad_frac * intensity_diff_top
    bottom_prom_threshold = grad_frac * intensity_diff_bottom
    for i, grad_col in enumerate(nerve_grad.T):
        top_candidates, _ = scipy.signal.find_peaks(-grad_col[:mid], prominence=top_prom_threshold)
        bottom_candidates, _ = scipy.signal.find_peaks(grad_col[mid:], prominence=bottom_prom_threshold)

        if (top_candidates.size and bottom_candidates.size):
            xindices.append(i)
            top_candidate_closest_to_mid = np.max(top_candidates)
            bottom_candidate_closest_to_mid = np.min(bottom_candidates)
            top.append(top_candidate_closest_to_mid)
            bottom.append(bottom_candidate_closest_to_mid)

    return np.array(xindices), np.array(top), mid + np.array(bottom)


def _nerve_border_from_straightened_image(nerve_image, width_sigma=3):
    '''
    Finds the border of the nerve from the straightened image

    Parameters
    -------
    nerve_image : itk.Image[, 2]
        Straightened nerve image
    width_sigma : int, optional
        Radius for applying blurring

    Returns
    -------
    np.ndarray
        Array containing the top points of the nerve
    np.ndarray
        Array containing the bottom points of the nerve
    np.ndarray
        Array containing the gradient values at the top and bottom points
    '''
    # OK, there are two ways of doing this:
    # could look for the first peak from the middle
    # or look for the max gradient (max peak)
    # max gradient is simpler (to code), this is what this is
    # TODO: don't really want this, I want only the magnitude, or the heck, the value, in y
    # gradmag = itk.GradientMagnitudeRecursiveGaussianImageFilter[ImageType, ImageType].New(InputImage=nerve_image, Sigma=self.width_sigma)
    # gradmag.Update()
    # blur = itk.RecursiveGaussianImageFilter[ImageType, ImageType].New(Input=nerve_image, sigma=self.width_sigma)
    # TODO: More idiomatic way of doing this?

    blur = itk.MedianImageFilter[ImageType, ImageType].New(Input=nerve_image, Radius=width_sigma)
    grad = itk.GradientImageFilter[ImageType, itk.F, itk.F].New(Input=blur.GetOutput())
    grad.Update()
    npgrad = itk.array_from_image(grad.GetOutput())[:,:,1]

    mid = round(npgrad.shape[0]/2)
    # TODO: uncomment below when comparing and contrasting
    xindices, top, bottom = _find_top_and_bottom_of_nerve_grad(npgrad, mid)
    # xindices, top, bottom = _find_top_and_bottom_nerve_width_pf_middle_rectangles(npgrad, blur.GetOutput(), mid, grad_frac=grad_frac)
    top_pts = np.array([xindices, top]).T
    bottom_pts = np.array([xindices, bottom]).T

    values = np.array([])
    if 0 in (len(top_pts), len(bottom_pts)):
        return None, None, None

    values = np.array([npgrad[top, xindices], npgrad[bottom, xindices]])
    return top_pts, bottom_pts, values.T


def _nerve_width_and_grad_vals_from_straight_image(nerve_image, **kwargs):
    top_pts, bottom_pts, values = _nerve_border_from_straightened_image(nerve_image, **kwargs)
    if values is not None:
        nerve_widths = _nerve_width_from_border_pts(top_pts, bottom_pts, nerve_image.GetSpacing())
    return nerve_widths, values

def _nerve_width_from_border_pts(top_pts, bottom_pts, spacing):
    if top_pts is None or bottom_pts is None:
        return None
    xpts = bottom_pts[:, 0] * spacing[0]
    ydiffs = (bottom_pts[:, 1] - top_pts[:, 1]) * spacing[1]

    return np.array([xpts, ydiffs]).T

def _calculate_onsd(nerve_width, edge_values, x0, x1):
    '''
    Calculate the ONSD from the widths and edge gradient magnitude values between x0 and x1 (mm) on the straightened nerve.

    Parameters
    ==========
    nerve_width (numpy array Mx2)
    edge_values (numpy array M)
    x0 (float) : position in mm, e.g. 2.5
    x1 (float) : position in mm, e.g., 3.5

    Returns
    =======
    onsd (float), score (float)
    '''
    onsd = None
    score = None
    if nerve_width is not None and edge_values is not None:
        idxs = np.argwhere((x0  < nerve_width[::,0]) & (nerve_width[::,0] < x1))
        if len(idxs) > 0:
            onsd = np.mean(nerve_width[idxs,1])
            score = np.median(np.abs(edge_values[idxs,::])) # should this be limited to the sample position window?  in the paper it was not
    return onsd, score


def _convert_multi_label_mask_to_single_label(multi_label_mask, foreground_value=1):
    thresh = itk.BinaryThresholdImageFilter[type(multi_label_mask), type(multi_label_mask)].New()
    thresh.SetInput(multi_label_mask)
    thresh.SetLowerThreshold(foreground_value)
    thresh.SetUpperThreshold(foreground_value)
    thresh.SetInsideValue(1)
    thresh.SetOutsideValue(0)
    thresh.Update()
    return thresh.GetOutput()

def markup_input_image(input_image, mask=None, closest_eye_pt=None, normal_left_endpt=None,
                        normal_right_endpt=None, meas_loc=None, skeleton=None, ax=None, alpha=0.4):

    '''
    Markup the passed input image with information from various stages of the onsd estimation algorithm

    Parameters
    ==========
    input_image : itk.Image[, 2]
        Input frame from an ultrasound video
    mask : itk.Image[, 2]
        The mask corresponding to the input image
    closest_eye_pt : itk.Index[2]
        Point on the eye that is closest to the medial axis
    normal_left_endpt : itk.Index[2]
        Left endpoint of the measurement line
    normal_right_endpt : itk.Index[2]
        Right endpoint of the measurement line
    meas_loc : itk.Index[2]
        Point down the medial axis where the measurement line should be drawn.
        If the closest_eye_pt is specified, another will be drawn connecting
        that point and the meas_loc
    skeleton : CurveGraph
        Result of the erosion on the binary mask
    ax : matplotlib.axes.Axes
        Pre-specified axes to use. If None, will use plt.gca()
    alpha : float
        alpha value for displaying the mask on top of the image
        Should be between 0-1

    Returns
    =======
    plt.axes corresponding to the marked up input image
    '''
    if ax is None:
        ax = plt.gca()

    ax.axis("off")

    if skeleton is None:
        ax.imshow(input_image, cmap="gray")
    else:
        skeleton.create_curve_plot_on_image(input_image, ax=ax)

    if mask is not None:
        # Eye first
        eye_mask_arr = itk.array_from_image(mask) == 1
        eye_mask_alphas = eye_mask_arr * alpha
        ax.imshow(eye_mask_arr, cmap="Greens", alpha=eye_mask_alphas)

        # Next is the nerve
        nerve_mask_arr = itk.array_from_image(mask) == 2
        nerve_mask_alphas = nerve_mask_arr * alpha
        ax.imshow(nerve_mask_arr, cmap="Blues", alpha=nerve_mask_alphas)

    if normal_left_endpt is not None and normal_right_endpt is not None:
        ax.plot([normal_left_endpt[0], normal_right_endpt[0]], [normal_left_endpt[1], normal_right_endpt[1]])

    if closest_eye_pt is not None:
        ax.plot(closest_eye_pt[0], closest_eye_pt[1], marker="o")

        if meas_loc is not None:
            xs = [closest_eye_pt[0], meas_loc[0]]
            ys = [closest_eye_pt[1], meas_loc[1]]

            # Should plot a line connected the closest point on the eye to
            # the measurement point
            ax.plot(xs, ys)
    return ax

@dataclass
class ONSDDebugInfo:
    # TODO: More specific itk image type
    input_image: itk.Image = None
    mask: itk.Image = None
    nerve_image: itk.Image = None
    skeleton: itk.Image = None
    top_pts: np.ndarray = None
    bottom_pts: np.ndarray = None
    onsd_sample_position: List[float] = None
    onsd: float = None
    score: float = None
    percentile: float = None
    medial_curve: Curve = None
    nerve_offset: float = None
    dist: float = None
    closest_eye_pt: np.ndarray = None

    def complete_prediction(self):
        '''
        Determines whether this instance has all fields populated
        from all parts of the algorithm

        Returns
        -------
        (bool) True / False value whether all fields are populated
        '''
        return self.onsd is not None

    def calculate_meas_dist_along_curve(self):
        '''
        Calculates the position down the medial curve corresponding
        to the middle of the measurement range

        Returns
        -------
        (float) Distance down the medial curve
        '''
        if None not in [self.dist, self.nerve_offset, self.onsd_sample_position]:
            return -self.dist + self.nerve_offset + np.mean(self.onsd_sample_position)

        return None

    def create_border_of_straightened_nerve_plot(self, ax=None):
        '''
        Creates a plot of the straightened nerve image
        using the instance fields

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Optional axes to use

        Returns
        ----------
        ax: matplotlib.axes.Axes
        '''
        return create_border_of_straightened_nerve_plot(
            self.nerve_image,
            ax=ax,
            bottom_pts=self.bottom_pts,
            top_pts=self.top_pts,
            onsd_sample_position=self.onsd_sample_position,
            onsd=self.onsd,
            score=self.score,
            perc=self.percentile,
        )

    def plot_im_with_mask_and_meas(self, plot_normal=True, ax=None, alpha=0.4):
        '''
        Marks up the input image with the predicted mask, measurement line,
        and a line connecting the closest point on the eye to the middle of the
        measurement line.

        Parameters
        ----------
        plot_normal : bool
            Whether or not to plot the measurement line
        ax : matplotlib.axes.Axes
            Optional axes to use
        alpha : float
            Transparency value for the mask. Should be between 0 and 1

        Returns
        ----------
        ax: matplotlib.axes.Axes
        '''
        meas_dist = self.calculate_meas_dist_along_curve()
        meas_loc = None
        if self.medial_curve is not None:
            meas_loc = self.medial_curve.evaluate(np.array([meas_dist]))[0]
            meas_loc = self.input_image.TransformPhysicalPointToIndex(meas_loc)


        normal_left_endpt, normal_right_endpt = None, None
        if plot_normal:
            if meas_dist is not None:
                normal_left_endpt, normal_right_endpt = self.calculate_endpts_normal_along_curve(meas_dist)

        return markup_input_image(
            self.input_image,
            mask=self.mask,
            normal_left_endpt=normal_left_endpt,
            normal_right_endpt=normal_right_endpt,
            meas_loc=meas_loc,
            closest_eye_pt=self.input_image.TransformPhysicalPointToIndex(self.closest_eye_pt),
            ax=ax,
            alpha=alpha
            )

    def markup_skeleton_image(self, ax=None):
        '''
        Marks up the input image with the curvegraph and measurement line.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Optional axes to use

        Returns
        ----------
        ax: matplotlib.axes.Axes
        '''
        if ax is None:
            ax = plt.gca()

        ax.axis("off")

        meas_loc = self.calculate_meas_dist_along_curve()
        if meas_loc is not None:
            normal_left_endpt, normal_right_endpt = self.calculate_endpts_normal_along_curve(meas_loc)
        return markup_input_image(
            self.input_image,
            normal_left_endpt=normal_left_endpt,
            normal_right_endpt=normal_right_endpt,
            ax=ax,
            skeleton=self.skeleton,
            )

    def calculate_endpts_normal_along_curve(self, meas_dist):
        '''
        Calculates the endpoints of the measurement line at the measurement location
        on the curve.

        Parameters
        ----------
        meas_dist : float
            Distance down the curve to calculate the endpoints at
            This is also the middle of the line.

        Returns
        ----------
        meas_line_left_endpt (itk.Index[2] or None), meas_line_right_endpt (itk.Index[2] or None)
        '''
        if None not in [meas_dist, self.medial_curve, self.onsd]:
            meas_loc_image_space = self.medial_curve.evaluate(np.array([meas_dist]))[0]
            ortho_vector_at_meas_loc = self.medial_curve.normal(np.array([meas_dist]))[0]

            meas_line_left_endpt = ortho_vector_at_meas_loc * (-self.onsd / 2)
            meas_line_right_endpt = ortho_vector_at_meas_loc * (self.onsd / 2)

            meas_line_left_endpt += meas_loc_image_space
            meas_line_right_endpt += meas_loc_image_space

            meas_line_left_endpt = self.input_image.TransformPhysicalPointToIndex(meas_line_left_endpt)
            meas_line_right_endpt = self.input_image.TransformPhysicalPointToIndex(meas_line_right_endpt)

            return meas_line_left_endpt, meas_line_right_endpt
        return None, None

    def create_debug_obj_figure(self, alpha=0.4, blank_nerve_image_size=None):
        '''
        Display a single debug object.

        Resulting figure contains three images stacked in a row. First, the
        original ultrasound image with the eye and nerve masks overlaid on the
        image. Second, the input image with the medial axis skeleton overalaid.
        Lastly, The straightened nerve image with the outline points of the nerve
        plotted, along with a blue line along the middle of the nerve, and two
        yellow lines specifying the range where the ONSD will be measured


        Parameters
        ----------
        debug_obj : ONSDDebugInfo
            Debug object to display
        alpha: float
            Transparency of the overlaid eye and nerve mask. Ranges from 0-1
        blank_nerve_image_size: tuple[2]
            If there is no nerve_image to be shown, this specifies the size of the
            blank image to be shown instead

        Returns
        -------
        None
        '''
        if blank_nerve_image_size is None:
            blank_nerve_image_size = (512, 512)

        f, axes = plt.subplots(1, 3)

        # Figure 1 is the original image with both masks overlayed
        self.plot_im_with_mask_and_meas(ax=axes[0], alpha=alpha)

        # Figure 2 is the nerve mask with the skeleton overlayed
        if self.skeleton is not None:
            self.markup_skeleton_image(axes[1])
        else:
            original_im_size = tuple(self.input_image.GetLargestPossibleRegion().GetSize())
            axes[1].imshow(np.zeros(original_im_size[::-1]))

        # Figure 3 is the nerve_image with the top and bottom border points
        if self.nerve_image is not None:
            self.create_border_of_straightened_nerve_plot(ax=axes[2])
        else:
            axes[2].imshow(np.zeros(blank_nerve_image_size))

        f.set_size_inches(18.5, 10.5)

        return f



class ONSDDebugInfoCollection:
    def __init__(self, debug_objs=None):
        self.debug_objs = list() if debug_objs is None else debug_objs

    def append(self, debug_obj):
        self.debug_objs.append(debug_obj)

    def calculate_percentiles(self):
        all_scores = []
        for dobj in self.debug_objs:
            if dobj.score:
                all_scores.append(dobj.score)

        # Now that we have all of the scores, we can update the individual objects
        all_scores.sort()
        all_scores = np.array(all_scores)
        n_scores = all_scores.shape[0]
        for dobj in self.debug_objs:
            if dobj.score:
                perc = (all_scores < dobj.score).sum()
                perc /= n_scores
                dobj.percentile = perc

    def __iter__(self):
        return iter(self.debug_objs)

def calculate_onsd_from_mask(input_image, mask, onsd_sample_position=[2.5, 3.5], debug_obj=None, **kwargs):
    '''
    Main entrypoint to the onsd estimation algorithm. Estimates the
    ONSD for one frame.

    Parameters
    -------
    input_image : itk.Image[itk.F, 2]
        Input frame
    mask : itk.Image[itk.UC, 2]
        Multilabel mask corresponding to `input_image`
    onsd_sample_position : List[2] of float, optional
        Min and max values of the sampling range to average ONSD measurements
        from.
    debug_obj : ONSDDebugInfo, optional
        Optional argument that will have its fields populated with
        intermediate values calculated at each stage of the pipeline.
    **kwargs : dict, optional
        Additional arguments that can be passed to other functions called
        in the pipeline. See the `curve_downsample` parameter of `_map_nerve`,
        and the `width_sigma` argument to
        `_nerve_border_from_straightened_image`
    Returns
    -------
    float
        The onsd
    float
        The "score", of this onsd calculation. Calculated using the median
        strength of the gradient at each sampling position.

    '''

    if debug_obj is not None:
        debug_obj.input_image = input_image
        debug_obj.mask = mask

    # First get the individual masks for the eye and the nerve
    eye_mask = _convert_multi_label_mask_to_single_label(mask)
    nerve_mask = _convert_multi_label_mask_to_single_label(mask, foreground_value=2)

    # We need a list of points defining the eye
    eye_pts = get_pts_from_binary_mask(eye_mask)

    _map_nerve_kwarg_keys = ["curve_downsample"]
    _map_nerve_kwargs = {k: kwargs[k] for k in _map_nerve_kwarg_keys if k in kwargs}
    nerve_image, skeleton, skeleton_image = _map_nerve(nerve_mask, input_image, eye_pts, debug_obj=debug_obj, **_map_nerve_kwargs)
    if nerve_image is None:
        nerve_width, edge_values = None, None
        top_pts, bottom_pts = None, None
    else:
        # TODO: More idiomatic
        _nerve_width_kwarg_keys = ["width_sigma"]
        _nerve_width_kwargs = {k: kwargs[k] for k in _nerve_width_kwarg_keys if k in kwargs}

        top_pts, bottom_pts, edge_values = _nerve_border_from_straightened_image(nerve_image, **_nerve_width_kwargs)
        nerve_width = _nerve_width_from_border_pts(top_pts, bottom_pts, nerve_image.GetSpacing())

    onsd, score = _calculate_onsd(nerve_width, edge_values, onsd_sample_position[0], onsd_sample_position[1])
    if debug_obj:
        # TODO: Make a method that does this
        debug_obj.input_image = input_image
        debug_obj.nerve_image = nerve_image
        debug_obj.skeleton = skeleton
        debug_obj.top_pts = top_pts
        debug_obj.bottom_pts = bottom_pts
        debug_obj.onsd_sample_position = onsd_sample_position
        debug_obj.onsd = onsd
        debug_obj.score = score

    return onsd, score

def create_nerve_figure(img, nerve, ax=None):
    myimg =  overlay(itk.array_from_image(img), [itk.array_from_image(nerve.nerve_mask), itk.array_from_image(nerve.skeleton_image)], [np.array([0, 1, 1]), np.array([1, 0, 1])], [0.5, 1])
    if ax is None:
        return plt.imshow(myimg)
    else:
        ax.imshow(myimg)
        return None

def aggregate_onsds(onsds, scores, lower_perc=.9, upper_perc=1):
    if len(onsds) == 0:
        return None

    if isinstance(onsds, list):
        onsds = np.array(onsds)

    if isinstance(scores, list):
        scores = np.array(scores)

    lower = round(len(onsds) * lower_perc)
    upper = round(len(onsds) * upper_perc)

    # Not enough data? Don't throw any out
    if lower == upper:
        return np.mean(onsds)

    sort_idx = scores.argsort()
    good_pts = onsds[sort_idx[lower:upper]]

    return np.mean(good_pts)

#     model = eye.ellipse_model
#     inliers = eye.eyesocket_points
#     filtered_pts = eye.nerve_search_points
#     seed_pt = eye.nerve_seed_point

#     fig, ax = plt.subplots()
#     ax.set_aspect('equal', 'box') # make sure this is displayed correctly
#     plt.imshow(img)
#     img_xlim = plt.xlim()
#     img_ylim = plt.ylim()
#     if model is not None:
#         ts = np.linspace(0, 2*np.pi, num=1000)
#         pts2 = transform_to_indices(model.predict_xy(ts), img)
#         ax.fill(pts2[::,1], pts2[::,0], facecolor='red', alpha=0.2, zorder=2)

#         # print inliers
#         pts3 = transform_to_indices(inliers, img)
#         ax.scatter(pts3[:,1], pts3[:,0], color='pink', alpha=0.8)

#         if filtered_pts is not None:
#             tmp = transform_to_indices(np.array([filtered_pts[:,:,0].flatten(), filtered_pts[:,:,1].flatten()]).T, img)
#             ax.scatter(tmp[:,1], tmp[:,0], color='green', alpha=0.01)

#         #tmp4 = transform_to_indices(inliers, img)
#         if seed_pt is not None:
#             seed_idx = transform_to_indices(seed_pt[np.newaxis,:], img).flatten()
#             ax.scatter(seed_idx[1], seed_idx[0], color='blue')

#     plt.xlim(img_xlim)
#     plt.ylim(img_ylim)
#     return fig

class VideoReader(ABC):
    @abstractmethod
    def get_next(self):
        pass
    def at_end(self):
        pass

def load_results(fp):
    '''
    Loads the entirety of the results in directory, fp.
    '''
    i = 0
    prefix = fp + '/' + str(i)
    img_path = prefix + '.mha'
    results = []
    while os.path.exists(img_path):
        img = itk.imread(img_path)
        eye = EyeSegmentationRANSAC.Eye.load(prefix)
        nerve = NerveSegmentationSkeleton.Nerve.load(prefix)
        with open(prefix + '-duration.p', 'rb') as f:
            duration = pickle.load(f)

        results.append((img, eye, nerve, duration))

        i += 1
        prefix = fp + '/' + str(i)
        img_path = prefix + '.mha'
    return results

def write_result(fp, i, img, eye, nerve, duration):
    os.makedirs(fp, exist_ok=True)

    if img is not None:
        itk.imwrite(img, fp + '/' + str(i) + '.mha')
    if eye is not None:
        eye.save(fp + '/' + str(i))
    if nerve is not None:
        nerve.save(fp + '/' + str(i))
    with open(fp + '/' + str(i) + '-duration.p', 'wb') as f:
        pickle.dump(duration, f)


def estimate_width(results, lower_perc=0.02, upper_perc=0.1):
    nerve_plot = np.array([[np.median(r[2].nerve_width[:,1]), np.median(np.abs(r[2].edge_values.flatten()))] for r in results if r[2] is not None and r[2].nerve_width is not None])
    lower = round(nerve_plot.shape[0] * lower_perc)
    upper = round(nerve_plot.shape[0] * upper_perc)
    sort_idx = np.flip(nerve_plot[:,1].argsort())
    good_pts = nerve_plot[sort_idx[lower:upper], :]
#     plt.scatter(good_pts[:,0], good_pts[:,1])
    estimate = np.mean(good_pts[:,0])
    return estimate

def vidread(filepath):
    return (skvideo.io.vread(filepath)[::,::,::,1].squeeze() / 255.0).astype('float32')

class ClariusOfflineReader(VideoReader):
    # TODO, get rid of this hardcoded stuff
    EYE_WIDTH_MM = 24.2
    EYE_HEIGHT_MM = 23.7
    CLARIUS_EYE_WIDTH = 400


    def __init__(self, filepath, probe_width=37.57, crop=None):
        self.video = (skvideo.io.vread(filepath)[::,::,::,1].squeeze() / 255.0).astype('float32')
        if crop is not None:
            self.video = self.video[::,crop[0]:crop[1], crop[2]:crop[3]]

        npimg = np.amax(self.video, axis=0)
        nz_cols = np.nonzero(np.amax(npimg,axis=0))[0] # only 1 dimension returned
        nz_sc = nz_cols[0]
        nz_ec = nz_cols[-1]
#         set_trace()
        s = probe_width/(nz_ec-nz_sc)
        self.spacing = [s, s]
        self.current = 0

    def get_next(self):
        image = self.get(self.current)
        self.current += 1
        return image

    def get(self, i):
        image = itk.image_from_array(self.video[i])
        self.set_clarius_dimension(image)
        return image

    def size(self):
        return self.video.shape[0]

    def at_end(self):
        return self.current >= self.video.shape[0]

    def set_clarius_dimension(self, img):
        '''
        Hard-coded guesstimate for the CLARIUS spacing.  TODO: get this somehow from Clarius.
        '''
        img.SetSpacing(self.spacing)

class ONSDFrame:
    def __init__(self, image, preprocess, eye_segmentation, nerve_segmentation):
        self.image = image
        self.eye_segmentation = eye_segmentation
        self.nerve_segmentation = nerve_segmentation
        # etc etc etc
    def process(self):
        self.eye = None
        self.nerve = None

        self.input_image = self.preprocess(self.image)
        self.eye = self.eye_segmentation.process(self.input_image)
        if self.eye.nerve_point is not None:
            self.nerve = self.nerve_segmentation.process(self.input_image, self.eye)

class EyeSegmentationRANSAC:
    class Eye:
        OBJECT_SUFFIX = '-eye.p'
        def __init__(self, ellipse_model, eyesocket_points, nerve_search_points, nerve_seed_point, eye_seed_point,
                           mask=None):
            self.ellipse_model = ellipse_model
            self.eyesocket_points = eyesocket_points
            self.nerve_search_points = nerve_search_points
            self.nerve_seed_point = nerve_seed_point
            self.eye_seed_point = eye_seed_point
            self.mask = mask

        def save(self, prefix):
            # pickle object
            with open(prefix + self.OBJECT_SUFFIX, 'wb') as f:
                pickle.dump(self, f)

        def found(self):
            return self.nerve_seed_point is not None

        def __str__(self):
            xc, yc, a, b, theta = self.ellipse_model.params
            return 'xc: {}, yc: {}, a: {}, b: {}, theta: {}'.format(xc, yc, a, b, theta)

        @classmethod
        def load(cls, prefix):
            if not path.exists(prefix + cls.OBJECT_SUFFIX):
                return None

            with open(prefix + cls.OBJECT_SUFFIX, 'rb') as f:
                ans = pickle.load(f)

            return ans
            # load pickle
            # load nerve_image

    def __init__(self, blur_sigma=[1,1], downscale=[6,6], canny_threshold=[0.03, 0.06],
                use_active_contour=False, debug=False, edge_angle_arcs=[(.176,np.pi-.176)], ransac_min_samples=5,
                ransac_residual_threshold=1, ransac_max_trials=200,
                ellipse_width_threshold=[8.47, 21.78],
                ellipse_height_threshold=[8.295, 21.33], peak_sigma=10,
                radius=0, ac_sigma=.6, initial_distance=5, alpha=-0.03, beta=0,
                propagation_scaling=8, curvature_scaling=1.33, advection_scaling=1,
                speed_constant=1, maximum_rms_error=0.004, number_of_iterations=1600,
                edge_tolerance=None, output_min=0, output_max=1, eye_seed_sigma=None):
        self.blur_sigma = blur_sigma
        self.downscale = downscale
        self.canny_threshold = canny_threshold
        self.use_active_contour = use_active_contour
        self.debug = debug
        self.edge_angle_arcs = edge_angle_arcs
        self.ransac_min_samples = ransac_min_samples
        self.ransac_residual_threshold = ransac_residual_threshold
        self.ransac_max_trials = ransac_max_trials
        self.ellipse_width_threshold = ellipse_width_threshold
        self.ellipse_height_threshold = ellipse_height_threshold
        self.peak_sigma = peak_sigma

        self.radius = radius
        self.ac_sigma = ac_sigma
        self.initial_distance = initial_distance
        self.alpha = alpha
        self.beta = beta

        self.propagation_scaling = propagation_scaling
        self.curvature_scaling = curvature_scaling
        self.advection_scaling = advection_scaling
        self.speed_constant = speed_constant
        self.maximum_rms_error = maximum_rms_error
        self.number_of_iterations = number_of_iterations
        self.output_min = output_min
        self.output_max = output_max
        self.eye_seed_sigma = eye_seed_sigma

        self.edge_tolerance = edge_tolerance


    def load_eye(self, prefix):
        return self.Eye.load(prefix)

    def _good_eye(self, model, data):
    # TODO: also, what is height and what is width (a and b) might be arbitrary in the EllipseModel
        xc, yc, a, b, theta = model.params
        ans1 = self.ellipse_width_threshold[0] < a and a < self.ellipse_width_threshold[1] and self.ellipse_height_threshold[0] < b and b < self.ellipse_height_threshold[1]
        return ans1

    def _find_edges(self, input_image, debug=False):
#         blur_filter = itk.SmoothingRecursiveGaussianImageFilter[ImageType, ImageType].New(SigmaArray=self.blur_sigma, Input=input_image)
        blur_filter = itk.MedianImageFilter[ImageType, ImageType].New(Radius=int(self.blur_sigma[0]), Input=input_image)
        blur_filter.Update()

        shrink_filter = itk.ShrinkImageFilter[ImageType, ImageType].New(ShrinkFactors=self.downscale, Input=blur_filter.GetOutput())
        shrink_filter.Update()

        canny_filter = itk.CannyEdgeDetectionImageFilter[ImageType, ImageType].New(Input=shrink_filter.GetOutput(), LowerThreshold=self.canny_threshold[0], UpperThreshold=self.canny_threshold[1])
        canny_filter.Update()

        grad_filter = itk.GradientImageFilter[ImageType, itk.F, itk.F].New(Input=shrink_filter.GetOutput())
        grad_filter.Update()

        #TODO: ADD CHECKS THAT THESE AREN'T EMPTY ARRAYS (i.e. no points matching criteria, then return None model)
        indices = np.argwhere(itk.array_from_image(canny_filter.GetOutput()) > 0)
        if indices.shape[0] == 0:
            return None, None, None, None

        grad = -itk.array_from_image(grad_filter.GetOutput()) # negate so eye gradients point inward

        return grad, indices, canny_filter.GetOutput(), shrink_filter.GetOutput()


    def _fit_ellipse_ransac(self, input_image):
        grad, indices, canny_image, shrink_image = self._find_edges(input_image)

        if grad is None:
            return None, None
        edge_grad = grad[indices[:,0], indices[:,1],:]

        # TODO: VERIFY THIS WORKS AS INTENDED


        grad_angles = np.arctan2(edge_grad[:,1], edge_grad[:,0])
        tmp_idx = grad_angles < 0
        grad_angles[tmp_idx] = 2*np.pi + grad_angles[tmp_idx]

        # each arc is a bounding tuple of angles
        # main idea, do we just look at the bottom of the eye socket or the top as well
        filter_idx = np.zeros(len(grad_angles)).astype('bool')
        for arc in self.edge_angle_arcs:
            filter_idx = filter_idx | ((arc[0] <= grad_angles) & (grad_angles < arc[1]))
        filter_indices = indices[filter_idx, :]

        if filter_indices.shape[0] == 0:
            return None, None

        filter_pts = transform_to_physical(filter_indices, canny_image)

        if filter_pts.shape[0] <= self.ransac_min_samples:
            return None, None

        model, inliers = skimage.measure.ransac(filter_pts, skimage.measure.EllipseModel, self.ransac_min_samples, self.ransac_residual_threshold, is_model_valid=self._good_eye, max_trials=self.ransac_max_trials)
        return model, filter_pts[inliers,:]


    def _display_filter_output_if_debug(self, filter_, text=""):
        if self.debug:
            filter_.Update()
            print(text)
            plt.figure()
            plt.title(text)
            plt.imshow(filter_.GetOutput())
            plt.show()

    # TODO: Split this out into 2 functions? One that runs active contour, another that fits the ellipse?
    def _fit_ellipse_active_contour(self, input_image, ref_point, radius, initial_distance, sigma, output_min,
                                    output_max, alpha, beta, propagation_scaling, curvature_scaling,
                                    advection_scaling, maximum_rms_error, number_of_iterations, speed_constant, edge_tolerance):
        ImageType = type(input_image)
        OutputPixelType = itk.UC
        OutputImageType = itk.Image[OutputPixelType, 2]


        # First apply smoothing
        smoothing = itk.MedianImageFilter[ImageType, ImageType].New(Radius=radius, Input=input_image)

        self._display_filter_output_if_debug(smoothing, "Smoothing")

        # Then magnitude of gradient by convolution with first deriv of gaussian
        gradient = itk.GradientMagnitudeRecursiveGaussianImageFilter[ImageType, ImageType].New(
        Sigma=sigma,
        Input=smoothing.GetOutput())

        self._display_filter_output_if_debug(gradient, "Gradient")

        sigmoid = itk.SigmoidImageFilter[ImageType, ImageType].New(
            OutputMinimum=output_min,
            OutputMaximum=output_max,
            Alpha=alpha,
            Beta=beta,
            Input=gradient.GetOutput()
        )
        self._display_filter_output_if_debug(sigmoid, "Sigmoid")

        sigmoid.Update()
        sigmoid_output = sigmoid.GetOutput()

        index = sigmoid_output.TransformPhysicalPointToIndex(ref_point)
        node = itk.LevelSetNode[InputPixelType, Dimension]()
        node.SetValue(-initial_distance)
        node.SetIndex(index)

        seeds = itk.VectorContainer[itk.UI, itk.LevelSetNode[InputPixelType, Dimension]].New()
        seeds.Initialize()
        seeds.InsertElement(0, node)

        fastmarch = itk.FastMarchingImageFilter[ImageType, ImageType].New()
        fastmarch.SetTrialPoints(seeds)
        fastmarch.SetSpeedConstant(speed_constant)
        fastmarch.SetOutputSize(sigmoid_output.GetBufferedRegion().GetSize())
        fastmarch.Update()
        fastmarch_out = fastmarch.GetOutput()

        fastmarch_out.SetSpacing(sigmoid_output.GetSpacing())

        geodesic = itk.GeodesicActiveContourLevelSetImageFilter[ImageType, ImageType, InputPixelType].New()
        geodesic.SetPropagationScaling(propagation_scaling)
        geodesic.SetCurvatureScaling(curvature_scaling)
        geodesic.SetAdvectionScaling(advection_scaling)
        geodesic.SetMaximumRMSError(maximum_rms_error)
        geodesic.SetNumberOfIterations(number_of_iterations)
        geodesic.SetInput(fastmarch_out)
        geodesic.SetFeatureImage(sigmoid_output)

        self._display_filter_output_if_debug(geodesic, "Active Contour")

        ThresholdingFilterType = itk.BinaryThresholdImageFilter[ImageType, OutputImageType]
        thresholder = ThresholdingFilterType.New()
        thresholder.SetInput(geodesic.GetOutput())
        thresholder.SetUpperThreshold(0.0)
        thresholder.SetOutsideValue(0)
        thresholder.SetInsideValue(1)

        thresholder.Update()
        if self.debug:
            print(geodesic.GetElapsedIterations())
            print(geodesic.GetRMSChange())
        model, pts = find_ellipse_of_best_fit(thresholder.GetOutput(), edge_tolerance)
        if model is not None:
            inliers_mask = model.residuals(pts) < self.ransac_residual_threshold
            inliers_pts = pts[inliers_mask, :]
            # Fix the ellipse
            a, b, theta = model.params[2:]
            if a < b:
                a, b = b, a
                theta -= (np.pi / 2)
                model.params[2:] = [a, b, theta]

            return model, inliers_pts, thresholder.GetOutput()
        return None, None, None

    def _find_seed(self, img, filtered_pts, peak):
        seed_pt = filtered_pts[math.floor(filtered_pts.shape[0]/2), peak].flatten()
        return seed_pt

    def _find_peak(self, npimg, axis=0, peak_sigma=None, prominence=0.1):
        if peak_sigma is None:
            peak_sigma = self.peak_sigma
        npimg1d = np.sum(npimg, axis=axis)
        if len(npimg1d) == 0: # TODO: this check should be put to the calling function
            return None, None
        npimg1d = npimg1d / np.max(npimg1d)
        npimg1d = scipy.ndimage.filters.gaussian_filter(npimg1d, peak_sigma)
        peaks, properties = scipy.signal.find_peaks(1-npimg1d, prominence=prominence)
        if (len(peaks) >= 1):
            return peaks, properties
        else:
            return None, None

    def _get_eye_seed_pt(self, img, eye_seed_sigma=None):
        if eye_seed_sigma is None:
            eye_seed_sigma = self.eye_seed_sigma

        npimg = itk.array_from_image(img)


        peaks, properties = self._find_peak(npimg, axis=1, peak_sigma=eye_seed_sigma)
        if peaks is None:
            return None
        rmid = peaks[np.argmax(properties['prominences'])]

        peaks, _ = self._find_peak(npimg, axis=0, prominence=None, peak_sigma=eye_seed_sigma)
        if peaks is None:
            return None

        _, w = npimg.shape

        # Might be air gaps on either side,
        # find point closest to center
        # TODO: same process for rmid?
        errors = np.abs(peaks - w/2)
        cmid = peaks[np.argmin(errors)]

        return transform_to_physical([(rmid, cmid)], img)[0]




    def _polar_based_bounds(self, x, y, a, b, theta):
        '''
        Very nice.
        '''
        ux = a * np.cos(theta)
        uy = a * np.sin(theta)
        vx = b * np.cos(theta + np.pi/2.0)
        vy = b * np.sin(theta + np.pi/2.0)
        hw = np.sqrt(ux**2 + vx**2)
        hh = np.sqrt(uy**2 + vy**2)
        return np.array([[x - hw, y - hh], [x + hw, y + hh]])

    def _calculate_eye_arc(self, image, model):
        '''
        Return two angles defining an arc on model.  Currently, finds the bounding box of the ellipse and computes
        the greatest (furthest from transducer) y coordinate.  It then calculates the two horizontal edges of the image
        along that y coordinate.  Finally, the two angles on model (ellipse) that are closest to the edge points are
        returned and used to calculate an arc on the ellipse for searching for the nerve.

        '''
        img_size = np.array(image.GetLargestPossibleRegion().GetSize()) # index size
        bounds = self._polar_based_bounds(model.params[0], model.params[1], model.params[2], model.params[3], model.params[4])
        ymax = bounds[1,1]

        index0 = image.GetLargestPossibleRegion().GetIndex()
        index1 = index0 + image.GetLargestPossibleRegion().GetSize()
        point0 = image.TransformIndexToPhysicalPoint(index0)
        point0.SetElement(1, ymax)
        point1 = image.TransformIndexToPhysicalPoint(index1)
        point1.SetElement(1, ymax)
        theta0 = nearest_angle(point0, model)
        theta1 = nearest_angle(point1, model)
        return theta0, theta1

    def _find_nerve_search(self, image, model, inliers):
        '''
        Returns the image strip of where to look for the nerve seed point.

        Parameters
        ----------
        image (itk.Image)
        model (skimage.measure.EllipseModel)
        inliers (ndarray Nx2) : in physical point space
        '''
        # TODO pull out these parameters
        cut = 3
        search_start = 1 # 1mm
        search_thick = 3 # 3mm
        search_spacing = np.array([.1, .1]) # also in mm

        # get an ellipse larger (a little further from the boundary of model)
        e1 = skimage.measure.EllipseModel()
        e1.params = model.params + np.array([0, 0, search_start, search_start, 0])
        e2 = skimage.measure.EllipseModel()
        e2.params = e1.params + np.array([0, 0, search_thick, search_thick, 0])

#         # get rightmost and leftmost points on model's perimeter
#         inliers = inliers[np.argsort(inliers[:,0]),:]
#         pt1 = inliers[cut,:]
#         pt2 = inliers[-(cut+1),:]

#         # find equally spaced arcs along the outer ellipse, because we don't know how the optimizer rotated and stretched the best-fitting ellipse
#         # we don't know what half of the ellipse (defined by theta1 and theta2) is the bottom part.  So, we'll calculate both arcs and pick the
#         # one with the lowest point
#         theta1 = nearest_angle(pt1, model)
#         theta2 = nearest_angle(pt2, model) # these angles still work on the larger ellipses cuz of uniform scaling (i think)

#         # TODO, replace linspace with a very simply approximation THIS IS A HUGE TIME SINK
#         ts1 = linspace_ellipse(e2, theta1, theta2, search_spacing[0]) # .1mm steps along the outer ellipse
#         if len(ts1) < 2:
#             print('Warning: small ts1: {}, {}'.format(theta1, theta2))
#             return None, None
#         miny1 = np.min(e2.predict_xy(ts1)[:,1]) # this should be max because it upper-left origin

#         # TODO: is this flip necessary?
#         #ts2 = np.flip(linspace_ellipse(e2, theta2, theta1, search_spacing[0]))
#         ts2 = np.flip(linspace_ellipse(e2, theta2, theta1, search_spacing[0]))

#         miny2 = np.min(e2.predict_xy(ts2)[:,1])
#         ts = ts1 if miny1 > miny2 else ts2

        theta1, theta2 = self._calculate_eye_arc(image, model)
        ts = linspace_ellipse(e2, theta1, theta2, search_spacing[0])
        ss = np.arange(0, search_thick, search_spacing[1]) / search_thick # no longer in mm, in proportion of search_thick

        search_img = np.zeros([len(ss), len(ts)])
        search_physical_pts = np.zeros([search_img.shape[0], search_img.shape[1], 2])

        ds = []
        x1s = []
        for i in range(len(ts)):
            t = ts[i]
            x1 = e1.predict_xy(t)
            x2 = e2.predict_xy(t)
            d = x2 - x1

            x1s.append(x1)
            ds.append(d) # comment this out

            pts = np.array([x1[0] + d[0] * ss, x1[1] + d[1] * ss]).T
            search_physical_pts[:,i,:] = pts

        return resample_by_grid(search_physical_pts, image)

    def process(self, input_image, eye_seed_pt=None):
        peak = None
        filtered_pts = None
        nerve_seed_pt = None
        model = None
        inliers = None
        mask = None
#         t3 = datetime.now()
        if not self.use_active_contour:
            model, inliers = self._fit_ellipse_ransac(input_image)
            mask = get_mask_from_ellipse(input_image, model)
        else:
            # If the user didn't provide an optional seed point for the eye,
            # find one.
            if eye_seed_pt is None:
                eye_seed_pt = self._get_eye_seed_pt(input_image, self.eye_seed_sigma)

            # eye_seed_point may be none at this point if the above failed
            if eye_seed_pt is not None:
                model, inliers, mask = self._fit_ellipse_active_contour(input_image, eye_seed_pt,
                    self.radius, self.initial_distance, self.ac_sigma, self.output_min,
                    self.output_max, self.alpha, self.beta, self.propagation_scaling, self.curvature_scaling,
                    self.advection_scaling, self.maximum_rms_error, self.number_of_iterations, self.speed_constant, self.edge_tolerance)

        if model is not None and inliers.shape[0] > 8: #TODO - change this to a function of cut in find_nerve_search
            npimg, filtered_pts = self._find_nerve_search(input_image, model, inliers)
            if npimg is not None:
                peaks, properties = self._find_peak(npimg)
                if peaks is not None:
                    peak = peaks[np.argmax(properties['prominences'])]
                    nerve_seed_pt = self._find_seed(input_image, filtered_pts, peak)
#         t4 = datetime.now()
#        frame_times.append(t4-t3)
#         fig = create_eye_figure(input_image, model, inliers, filtered_pts, nerve_seed_pt)
# TODO: remove the mask as a parameter to the eye after making a new class for the AC approach?
        return EyeSegmentationRANSAC.Eye(model, inliers, filtered_pts, nerve_seed_pt, eye_seed_pt, mask)
        #return fig
#         return eye


def resample_by_grid_point(grid, image):
    '''
    grid is a MxNx2
    '''
    img_size = np.asarray(image.GetLargestPossibleRegion().GetSize())
    interp = itk.NearestNeighborInterpolateImageFunction[ImageType, itk.D].New()
    interp.SetInputImage(image)
    ans = np.zeros([grid.shape[0], grid.shape[1]], dtype='float32')
    for c in range(ans.shape[1]):
        for r in range(ans.shape[0]):
            # constructor ought to work, bet this is due to lack of bounds checking
            pt = itk.Point[itk.D, 2](grid[r,c])
            if image.GetLargestPossibleRegion().IsInside(image.TransformPhysicalPointToIndex(pt)):
            #if (0 <= grid[r, c, 0] and grid[r, c, 0] < img_size[0]) and (0 <= grid[r, c, 1] and grid[r, c, 1] < img_size[1]):
                ans[r, c] = interp.Evaluate(pt)
    return ans

class NerveSegmentationSkeleton:
    class Nerve:
        MASK_SUFFIX = '-nerve_mask.mha'
        IMAGE_SUFFIX = '-nerve_image.mha'
        SKELETON_IMAGE_SUFFIX = '-nerve_skeleton.mha'
        OBJECT_SUFFIX = '-nerve.p'

        def __init__(self, skeleton, skeleton_image, nerve_mask, nerve_image, nerve_width, edge_values, onsd, score):
            self.skeleton = skeleton
            self.skeleton_image = skeleton_image
            self.nerve_mask = nerve_mask
            self.nerve_image = nerve_image
            self.nerve_width = nerve_width
            self.edge_values = edge_values
            self.onsd = onsd
            self.score = score
        def save(self, prefix):
            # save images, set to None as they won't pickle
            tmp1 = None
            tmp2 = None
            tmp3 = None
            if self.skeleton_image is not None:
                itk.imwrite(self.skeleton_image, prefix + self.SKELETON_IMAGE_SUFFIX)
            if self.nerve_mask is not None:
                itk.imwrite(self.nerve_mask, prefix + self.MASK_SUFFIX)
            if self.nerve_image is not None:
                itk.imwrite(self.nerve_image, prefix + self.IMAGE_SUFFIX)
            tmp1 = self.nerve_mask
            tmp2 = self.nerve_image
            tmp3 = self.skeleton_image
            self.nerve_mask = None
            self.nerve_image = None
            self.skeleton_image = None

            # pickle object
            with open(prefix + self.OBJECT_SUFFIX, 'wb') as f:
                pickle.dump(self, f)

            # reset images
            self.nerve_mask = tmp1
            self.nerve_image = tmp2
            self.skeleton_image = tmp3

        @classmethod
        def load(cls, prefix):
            if not path.exists(prefix + cls.OBJECT_SUFFIX):
                return None

            with open(prefix + cls.OBJECT_SUFFIX, 'rb') as f:
                ans = pickle.load(f)

            f0 = prefix + cls.SKELETON_IMAGE_SUFFIX
            if path.exists(f0):
                ans.skeleton_image = itk.imread(f0)

            f1 = prefix + cls.MASK_SUFFIX
            if path.exists(f1):
                ans.nerve_mask = itk.imread(f1)

            f2 = prefix + cls.IMAGE_SUFFIX
            if path.exists(f2):
                ans.nerve_image = itk.imread(f2)

            return ans
            # load pickle
            # load nerve_image

    # TODO: Separate each implementation into subclasses to avoid tons of parameters
    # Right now we have separate radius and sigma parameters for active contour
    def __init__(self, level=0, threshold=0.1, radius=1, sigma=1.5, erosion=10,
                 nerve_offset=1, nerve_image_dimension=[6,12],
                 nerve_image_sampling=[50,100], width_sigma=3,
                 onsd_sample_position=[2.5,3.5], use_watershed=True,
                 use_active_contour=False, ac_radius=4, ac_sigma=0.07,
                 initial_distance=8, output_min=0,
                 output_max=1, alpha=-0.147, beta=0, propagation_scaling=7.3,
                 curvature_scaling=1.9, advection_scaling=4.56,
                 maximum_rms_error=0.005, number_of_iterations=1000,
                 speed_constant=1.0, debug=False):
        '''
        level (float) : level parameter rough watershed
        threshold (float) : threshold parameter rough watershed
        radius (float) : radius of median filter (in pixels) before watershed
        sigma (float) : sigma parameter for gradient magnitude filter (on which watershed is run)
        erosion (float) : erosion filter on watershed (to try and remove spurious edges that can affect binary thinning)
        nerve_offset (float) : fixed distance from eye edge to begin straightened nerve image

        '''
        self.level = level
        self.threshold = threshold
        self.radius = radius
        self.sigma = sigma
        self.erosion = erosion
        self.nerve_offset = nerve_offset
        self.nerve_image_dimension = np.asarray(nerve_image_dimension)
        self.nerve_image_sampling = np.asarray(nerve_image_sampling)
        self.width_sigma = width_sigma # supposedly in nerve_image_sampling units
        self.onsd_sample_position = onsd_sample_position

        # active contour based params
        self.ac_radius = ac_radius
        self.ac_sigma = ac_sigma
        self.initial_distance = initial_distance
        self.output_min = output_min
        self.output_max = output_max
        self.alpha = alpha
        self.beta = beta
        self.advection_scaling = advection_scaling
        self.propagation_scaling = propagation_scaling
        self.curvature_scaling = curvature_scaling
        self.maximum_rms_error = maximum_rms_error
        self.number_of_iterations = number_of_iterations
        self.speed_constant = speed_constant

        # If more than one of these are true, we'll use the first one only.
        # See _segment_nerve
        self.use_watershed = use_watershed
        self.use_active_contour = use_active_contour

        # Need to display intermediate results? Set this.
        self.debug = debug

    def _print_if_debug(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def _watershed(self, img, ref_point, level, threshold, radius, sigma):
        self._print_if_debug('_watershed', flush=True)
        smoothing = itk.MedianImageFilter[ImageType, ImageType].New(Radius=radius, Input=img)
        self._print_if_debug('_pre Gradient', flush=True)
        gradient = itk.GradientMagnitudeRecursiveGaussianImageFilter[ImageType, ImageType].New(Sigma=sigma, Input=smoothing.GetOutput())
        self._print_if_debug('pre pre Watershed', flush=True)
        try:
            watershed = itk.MorphologicalWatershedImageFilter[ImageType, itk.Image[itk.UC,2]].New( \
                 Level=level, \
                 Input=gradient.GetOutput())
            if self.debug:
                print('watershed constructed', flush=True)
                print('Writing gradient out', flush=True)
                itk.imwrite(gradient.GetOutput(), 'watershed-test.tif')
                print('Writing level and threshold out', flush=True)
                with open('watershed-out.pickle', 'wb') as f:
                    pickle.dump((level, threshold), f)
                print('Calling watershed.update', flush=True)
            watershed.Update()
            self._print_if_debug('_watershed post watershed.Update()', flush=True)
        except RuntimeError:
            return None, None, None, None # watershed failed
        tmp = watershed.GetOutput()
        self._print_if_debug(type(tmp))
        #itk.cast_image_filter(watershed.GetOutput(), ttype=(itk.Image[itk.ULL,2], itk.Image[itk.UC,2]))
        self._print_if_debug('_watershed pre LabelMap', flush=True)
        LabelMapType = itk.LabelMap[itk.StatisticsLabelObject[itk.UL,2]]
        #self._print_if_debug(type(watershed.GetOutput()))
        labelmap = itk.LabelImageToLabelMapFilter[itk.Image[itk.UC,2],LabelMapType].New(Input=tmp)

        label_value = tmp.GetPixel(tmp.TransformPhysicalPointToIndex(ref_point))
        labelselector = itk.LabelSelectionLabelMapFilter[LabelMapType].New(Input=labelmap, Label=label_value)
        self._print_if_debug('_watershed pre LabelImage', flush=True)
        labelimage = itk.LabelMapToLabelImageFilter[LabelMapType, itk.Image[itk.UC,2]].New(Input=labelselector.GetOutput())
        labelimage.Update()
        return labelimage.GetOutput(), gradient.GetOutput(), labelselector, label_value

    def _display_filter_output_if_debug(self, filter_, text=""):
        if self.debug:
            filter_.Update()
            plt.figure()
            plt.title(text)
            plt.imshow(filter_.GetOutput())
            plt.show()

    def _active_contour(self, img, ref_point, radius, initial_distance, sigma, output_min,
                        output_max, alpha, beta, propagation_scaling, curvature_scaling,
                        advection_scaling, maximum_rms_error, number_of_iterations, speed_constant):

        # This is the image type we are ultimately looking to return
        OutputPixelType = itk.UC
        OutputImageType = itk.Image[OutputPixelType, Dimension]

        # First apply smoothing
        smoothing = itk.MedianImageFilter[ImageType, ImageType].New(Radius=radius, Input=img)

        self._display_filter_output_if_debug(smoothing, "Smoothing")

        # Then magnitude of gradient by convolution with first deriv of gaussian
        gradient = itk.GradientMagnitudeRecursiveGaussianImageFilter[ImageType, ImageType].New(
        Sigma=sigma,
        Input=smoothing.GetOutput())

        self._display_filter_output_if_debug(gradient, "Gradient")

        sigmoid = itk.SigmoidImageFilter[ImageType, ImageType].New(
            OutputMinimum=output_min,
            OutputMaximum=output_max,
            Alpha=alpha,
            Beta=beta,
            Input=gradient.GetOutput()
        )
        self._display_filter_output_if_debug(sigmoid, "Sigmoid")

        sigmoid.Update()
        sigmoid_output = sigmoid.GetOutput()

        index = sigmoid_output.TransformPhysicalPointToIndex(ref_point)
        node = itk.LevelSetNode[InputPixelType, Dimension]()
        node.SetValue(-initial_distance)
        node.SetIndex(index)

        seeds = itk.VectorContainer[itk.UI, itk.LevelSetNode[InputPixelType, Dimension]].New()
        seeds.Initialize()
        seeds.InsertElement(0, node)

        fastmarch = itk.FastMarchingImageFilter[ImageType, ImageType].New()
        fastmarch.SetTrialPoints(seeds)
        fastmarch.SetSpeedConstant(speed_constant)
        fastmarch.SetOutputSize(sigmoid_output.GetBufferedRegion().GetSize())
        fastmarch.Update()
        fastmarch_out = fastmarch.GetOutput()

        self._display_filter_output_if_debug(fastmarch, "Fastmarch")

        fastmarch_out.SetSpacing(sigmoid_output.GetSpacing())

        self._display_filter_output_if_debug(fastmarch, "Fastmarch After Spacing")

        geodesic = itk.GeodesicActiveContourLevelSetImageFilter[ImageType, ImageType, InputPixelType].New()
        geodesic.SetAdvectionScaling(advection_scaling)
        geodesic.SetPropagationScaling(propagation_scaling)
        geodesic.SetCurvatureScaling(curvature_scaling)
        geodesic.SetMaximumRMSError(maximum_rms_error)
        geodesic.SetNumberOfIterations(number_of_iterations)
        geodesic.SetInput(fastmarch_out)
        geodesic.SetFeatureImage(sigmoid_output)

        self._display_filter_output_if_debug(geodesic, "Active Contour")

        ThresholdingFilterType = itk.BinaryThresholdImageFilter[ImageType, OutputImageType]
        thresholder = ThresholdingFilterType.New()
        thresholder.SetUpperThreshold(0.0)
        thresholder.SetOutsideValue(0)
        thresholder.SetInsideValue(1)
        thresholder.SetInput(geodesic.GetOutput())

        thresholder.Update()
        thresholder_out = thresholder.GetOutput()
        ref_point_as_index = thresholder_out.TransformPhysicalPointToIndex(ref_point)
        label_value = thresholder_out.GetPixel(ref_point_as_index)
        return thresholder_out, label_value

    def _map_nerve(self, input_image, eye):
        self._print_if_debug('_map_nerve', flush=True)
        if self.use_watershed:
            label, gradient, labelselector, label_value = self._watershed(input_image, eye.nerve_seed_point, self.level, self.threshold, self.radius, self.sigma)

        elif self.use_active_contour:
            label, label_value = self._active_contour(input_image, eye.nerve_seed_point, self.ac_radius, self.initial_distance,
                                self.ac_sigma, self.output_min, self.output_max, self.alpha, self.beta, self.propagation_scaling,
                                self.curvature_scaling, self.advection_scaling,
                                self.maximum_rms_error, self.number_of_iterations, self.speed_constant)

        else:
            raise RuntimeError("No nerve segmentation algorithm specified.")

        if label is None:
            return None, None, None, None
#         nerve_labels[j] = itk.array_from_image(label)
#TODO: speedup by limiting convolution to bounding box around label
        erosion = itk.BinaryErodeImageFilter[itk.Image[itk.UC,2], itk.Image[itk.UC,2], itk.FlatStructuringElement[2]].New( \
            Input=label, \
            ForegroundValue=label_value, \
            BoundaryToForeground=False, \
            Kernel=itk.FlatStructuringElement[2].Ball(self.erosion))
        erosion.Update()


        nerve_image, skeleton, skeleton_image = _map_nerve(erosion.GetOutput(),
                                                            input_image,
                                                           eye.ellipse_model,
                                                           self.nerve_offset,
                                                           self.nerve_image_dimension,
                                                           self.nerve_image_sampling)

        return nerve_image, erosion.GetOutput(), skeleton, skeleton_image

    def _nerve_width(self, nerve_image):
        self._print_if_debug('_nerve_width', flush=True)
        return _nerve_width_and_grad_vals_from_straight_image(nerve_image, width_sigma=self.width_sigma)[0]

    def calcuate_onsd(self, nerve_width, edge_values, x0, x1):
        return _calculate_onsd(nerve_width, edge_values, x0, x1)


    def process(self, input_image, eye):
        self._print_if_debug('process', flush=True)
        nerve_image, nerve_mask, skeleton, skeleton_image = self._map_nerve(input_image, eye)
        if nerve_image is None:
            nerve_width = None
            edge_values = None
        else:
            nerve_width, edge_values = self._nerve_width(nerve_image)

        onsd, score = self.calcuate_onsd(nerve_width, edge_values, self.onsd_sample_position[0], self.onsd_sample_position[1])

        return NerveSegmentationSkeleton.Nerve(skeleton, skeleton_image, nerve_mask, nerve_image, nerve_width, edge_values, onsd, score)

    def load_nerve(self, prefix):
        return NerveSegmentationSkeleton.Nerve.load(prefix)


#['4mm-capture_1', '5mm-capture_1', '6mm-capture_1', '7mm-capture_1']
#['3mm-capture_1']#,


# ids = ['3mm-capture_1', '4mm-capture_1', '5mm-capture_1', '6mm-capture_1', '7mm-capture_1']
# for myid in ids:
#     with open(myid + '.p', 'rb') as f:
#         r = pickle.load(f)
#     outv = r[0]
#     writer = skvideo.io.FFmpegWriter(myid + "-output.mp4", outputdict={'-pix_fmt': 'yuv420p'})
#     for i in range(outv.shape[0]):
#         writer.writeFrame(outv[i,:,:,:])
#     writer.close()


# def image_point_to_annotation(pt, img, crop_transform):
#     '''
#     Returns:
#     annotation coordinates (original image index in x,y order)
#     '''
#     return np.asarray(img.TransformPhysicalPointToIndex(pt)) - np.asarray(np.flip(crop_transform.translation))

# def annotation_index_to_image(idx, img, crop_transform):
#     x = np.asarray(idx) + np.asarray(np.flip(crop_transform.translation))
#     idx2 = itk.Index[2]()
#     idx2.SetElement(0, int(x[0]))
#     idx2.SetElement(1, int(x[1]))
#     return np.asarray(img.TransformIndexToPhysicalPoint(idx2))

class VideoOutput:
    def __init__(self):
        # probably bad, but pickle doesn't handle the CurveGraph well
        sys.setrecursionlimit(30000)

        self.imgraws = []
        self.imgs = []
        self.eyes = []
        self.nerves = []

    def __len__(self):
        return len(self.imgraws)

    def append(self, imgraw, img, eye, nerve):
        self.imgraws.append(imgraw)
        self.imgs.append(img)
        self.eyes.append(eye)
        self.nerves.append(nerve)

    def calculate_onsd(self, nerves=None, lower_perc=.9, upper_perc=1):
        nerves = self.nerves if nerves is None else nerves
        if nerves is None:
            return None

        onsds = np.asarray([n.onsd for n in nerves if n is not None])
        scores = np.asarray([n.score for n in nerves if n is not None])

        return aggregate_onsds(onsds, scores,
                lower_perc=lower_perc, upper_perc=upper_perc)

    def get(self, i):
        return self.imgraws[i], self.imgs[i], self.eyes[i], self.nerves[i]

    def save(self, path):
        '''
        path (str) : full path to zipfile to save
        '''

        # save everything to a temp directory then zip it up (with-statement handles temp dir destruction)
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            with tempfile.TemporaryDirectory() as tmpdirname:
                for i in range(len(self)):
                    prefix = tmpdirname + os.sep + str(i)
                    itk.imwrite(self.imgraws[i], prefix + '-imgraw.mha')
                    itk.imwrite(self.imgs[i], prefix + '-img.mha')
                    if self.eyes[i] is not None:
                        self.eyes[i].save(prefix)
                    if self.nerves[i] is not None:
                        self.nerves[i].save(prefix)
                for f in os.listdir(tmpdirname):
                    z.write(tmpdirname + os.sep + f, arcname=f)

    def load(self, path):
        '''
        path (str) : full path to zipfile to load
        '''
        with zipfile.ZipFile(path, 'r', zipfile.ZIP_DEFLATED) as z:
            with tempfile.TemporaryDirectory() as tmpdirname:
                z.extractall(tmpdirname)

                i = 0
                prefix = tmpdirname + os.sep + str(i)
                imgrawpath = prefix + '-imgraw.mha'
                while os.path.exists(imgrawpath):
                    imgraw = itk.imread(imgrawpath)

                    img = None
                    imgpath = prefix + '-img.mha'
                    if os.path.exists(imgpath):
                        img = itk.imread(imgpath)

                    eye = EyeSegmentationRANSAC.Eye.load(prefix)
                    nerve = NerveSegmentationSkeleton.Nerve.load(prefix)

                    self.imgraws.append(imgraw)
                    self.imgs.append(img)
                    self.eyes.append(eye)
                    self.nerves.append(nerve)

                    i += 1
                    prefix = tmpdirname + os.sep + str(i)
                    imgrawpath = prefix + '-imgraw.mha'

