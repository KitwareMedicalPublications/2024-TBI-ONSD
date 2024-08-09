import re
import os
import zipfile
import xml.etree.ElementTree as ET
#from tbitk.data_manager import ImageAnnotation, VideoAnnotation
import numpy as np
from glob import glob
import os.path
import itk
import tbitk.util as util

def affine2D(x, transform):
    '''
    Returns affine transformation of Nx2 points
    
    Parameters
    ==========
    x (nparray): Nx2 or None
    transform (nparray): 3x3
    
    Returns:
    nparray : Nx2 or None if x is None
    '''
    if x is None:
        return None
    else:
        # append ones row, multiply by affine matrix, return Nx2 result
        return (transform @ np.vstack((x.T, np.ones(x.shape[0]))))[0:2,:].T

def extract_and_rename(fp, outdir, on_exists='a'):
    '''
    Extracts annotation.xml file from CVAT zip and renames it to the base name of the file (removing the CVAT part of the filename).
    Recommended naming scheme:
      1.  For video, name the CVAT task as the basename of the corresponding video file
      2.  For images, name the CVAT task after the experiment/dataset as all images will be in a single XML

    Parameters
    ==========
    fp - full filepath to .zip file from cvat
    outdir - directory to place output xml
    on_exits - ('f')orce, ('a')bort, ('s')kip
    '''
    assert on_exists == 'a' or on_exists == 'f' or on_exists == 's'
    
    f = os.path.basename(fp)
    m = re.match(r'task_(?P<name>[^.]+).*-\d\d\d\d_\d\d_\d\d_\d\d_\d\d_\d\d-(?P<format>.*)\.zip', f)
    name = m.group('name') + '.xml'
    outname = outdir + os.sep + name
    
    with zipfile.ZipFile(fp, mode='r') as zf:
        zf.extract('annotations.xml', path=outdir)
        if os.path.exists(outname):
            if on_exists == 'f':
                os.remove(outname)
            elif on_exists == 'a':
                raise RuntimeError('{} already exists and on_exists=\'a\''.format(outname))
                
        os.rename(outdir + os.sep + 'annotations.xml', outname)
    
    return outname

def parse_points_elem(txt):
    '''
    Returns list of floats from txt CSV
    '''
    return [float(x) for x in txt.split(',')]

def parse_points(txt):
    '''
    Parse the comma and semi-colon delimited polygon points string from CVAT
    
    Returns
    =======
    nparray
    '''
    return np.array([parse_points_elem(x) for x in txt.split(';')])
    
def get_xml_type(fp=None, tree=None):
    '''
    Parameters
    ==========
    fp - .xml filepath to parse
    tree - result of ElementTree.parse(fp)
    
    Returns
    'image' 'video' or None
    '''
    assert fp is not None or tree is not None
    
    if fp is not None:
        tree = ET.parse(fp)
    
    if tree.find('./image') is not None:
        return 'image'
    elif tree.find('./track') is not None:
        return 'video'
    else:
        return None
    
def parse_image_annotation(fp=None, tree=None):
    '''
    Parses XML file in CVAT for images 1.1 format.  Assumes optional 'eye' and 'nerve' shapes (polygons).
    
    Parameters
    ==========
    fp (str) : filepath to XML file
    
    Returns
    =======
    [ImageAnnotation]
    '''
    assert fp is not None or tree is not None
    
    if fp is not None:
        tree = ET.parse(fp)
        
    images = tree.findall('./image')
    result = []
    for img in images:
        name = os.path.splitext(img.attrib['name'])[0]
        eye_poly = img.find("./polygon[@label='eye']")
        eye = None if eye_poly is None else parse_points(eye_poly.attrib['points'])
        nerve_poly = img.find("./polygon[@label='nerve']")
        nerve = None if nerve_poly is None else parse_points(nerve_poly.attrib['points'])
        result.append(ImageAnnotation(name, eye, nerve))
        
    return result

class SingeInstanceImageAnnotation:
    '''
    Represents a CVAT image annotation where each label is represented by at most one track.
    '''
    @classmethod
    def parse(cls, fp=None, tree=None):
        '''
        Parameters
        ----------
        fp : str
            filepath to XML file
        tree : xml.etree.ElementTree
            already parsed XML element tree
            
        Returns
        -------
        list of SingleInstanceImageAnnotation
        '''
        assert fp is not None or tree is not None
    
        if fp is not None:
            tree = ET.parse(fp)

        images = tree.findall('./image')
        result = []
        for img in images:
            name = os.path.splitext(img.attrib['name'])[0]
            instances = {}
            
            polys = img.find('./polygon')
            for p in polys:
                label = p.attrib['label']
                instances[label] = parse_points(p.attrib['points'])
            
            result.append(SingleInstanceImageAnnotation(name, instances))

        return result
    
    def __init__(self, instances):
        self.instances = instances

    def transform(self, transform):
        for k in self.instances.keys():
            self.instances[k] = affine2D(self.instances[k], transform)
        
class SingleInstanceVideoAnnotation:
    '''
    Represents a CVAT video annotation where each label is represented by at most one track.
    '''
    @classmethod
    def parse(cls, fp=None, tree=None):
        assert fp is not None or tree is not None
    
        if fp is not None:
            tree = ET.parse(fp)
    
        name = os.path.splitext(tree.find('./meta/source').text)[0]
        frm_cnt = int(tree.find('./meta/task/size').text)
        instances = {}
    
        tracks = tree.findall('./track')
        for t in tracks:
            label = t.attrib['label']
            instances[label] = [None] * frm_cnt
            polys = t.findall('./polygon[@occluded="0"]')
            for p in polys:
                frm = int(p.attrib['frame'])
                instances[label][frm] = parse_points(p.attrib['points'])
        
        return SingleInstanceVideoAnnotation(name, frm_cnt, instances)
    
    def __init__(self, name, frame_count, instances):
        self.name = name
        self.frame_count = frame_count
        self.instances = instances
        
    def get_indices(self, key):
        return np.argwhere([x is not None for x in self.instances[key]]).squeeze()
    
    def transform(self, transform):
        for k in self.instances.keys():
            self.instances[k] = [affine2D(x, transform) for x in self.instances[k]]
    

def parse_video_annotation(fp=None, tree=None, keys=None):
    '''
    Parses XML file in CVAT for video 1.1 format.  Assumes optional 'eye' and 'nerve' tracks (polygons).  Eye or nerve tracks
    can be absent, but if they exist the occluded property denotes presence in a frame.
    
    Parameters
    ==========
    fp (str) : filepath to XML file
    
    Returns
    =======
    VideoAnnotation
    '''
    assert fp is not None or tree is not None
    
    if fp is not None:
        tree = ET.parse(fp)
        
    name = os.path.splitext(tree.find('./meta/source').text)[0]
    frm_cnt = int(tree.find('./meta/task/size').text)
            
    eyes = [None] * frm_cnt
    nerves = [None] * frm_cnt
    
    eye_polys = tree.findall("./track[@label='eye']/polygon[@occluded='0']")
    for e in eye_polys:
        frm = int(e.attrib['frame'])
        eyes[frm] = parse_points(e.attrib['points'])
    
    nerve_polys = tree.findall("./track[@label='nerve']/polygon[@occluded='0']")
    for n in nerve_polys:
        frm = int(n.attrib['frame'])
        nerves[frm] = parse_points(n.attrib['points'])
    
    return VideoAnnotation(name, eyes, nerves)
    
def get_image_label_image(img, annotation):
    '''
    Parameters
    ----------
    img : itk.Image[itk.F,2]
        2D image to use a reference for spacing, etc.
    annotation : ImageAnnotation
    
    Returns
    -------
    itk.Image[itk.UC,2]
        0 for bg, 1 for eye, 2 for nerve
    '''
    parent_spatial = itk.GroupSpatialObject[2].New()
    if annotation.has_eye(): # whether it has an eye
        
        # needs the spatial objects to be in index space
        eye_spatial = util.polygon_from_array(annotation.eye)
        
        # ITK filter expects physical space
        tmp = eye_spatial.Clone()
        tmp.SetObjectToWorldTransform(util.get_image_transform(img))
        tmp.SetDefaultInsideValue(1)
        parent_spatial.AddChild(tmp)

    if annotation.has_nerve(): # whether it has a nerve
        nerve_spatial = util.polygon_from_array(annotation.nerve)
        
        tmp = nerve_spatial.Clone()
        tmp.SetObjectToWorldTransform(util.get_image_transform(img))
        tmp.SetDefaultInsideValue(2)
        parent_spatial.AddChild(tmp)
        
    parent_spatial.Update()
    fil = itk.SpatialObjectToImageFilter[itk.SpatialObject[2], itk.Image[itk.UC,2]].New()
    fil.SetInput(parent_spatial)
    fil.SetOutsideValue(0)
    fil.SetUseObjectValue(True)
    fil.SetOrigin(img.GetOrigin())
    fil.SetSpacing(img.GetSpacing())
    fil.SetSize(img.GetLargestPossibleRegion().GetSize())
    fil.Update()
    return fil.GetOutput()
    
def get_video_label_image(img, annotation):
    '''
    Parameters
    ----------
    img : itk.Image[itk.F,3]
        Video image to use a reference for spacing, etc.
    annotation : VideoAnnotation
    
    Returns
    -------
    itk.Image[itk.UC,3]
        0 for bg, 1 for eye, 2 for nerve
    '''
    
    # tons of mess here because the SpatialObjectToImageFilter is too slow on 3D images, so use the 2D filter per frame
    size = np.array(img.GetLargestPossibleRegion().GetSize())
    ans = np.zeros((size[2], size[1], size[0]))
    first_img = util.extract_slice(img, 0)

    fil = itk.SpatialObjectToImageFilter[itk.SpatialObject[2], itk.Image[itk.UC,2]].New()
    fil.SetOutsideValue(0)
    fil.SetUseObjectValue(True)
    fil.SetOrigin(np.array(img.GetOrigin())[0:2])
    fil.SetSpacing(np.array(img.GetSpacing())[0:2])
    fil.SetSize(size[0:2].tolist())

    for i in range(annotation.frame_count):
        parent_spatial = itk.GroupSpatialObject[2].New()
        eye, nerve = annotation.get(i)

        if eye is not None: # whether it has an eye 
            # needs the spatial objects to be in index space
            eye_spatial = util.polygon_from_array(eye)#, world_transform=util.get_image_transform(img))

            # ITK filter expects physical space
            tmp = eye_spatial.Clone()
            tmp.SetObjectToWorldTransform(util.get_image_transform(first_img))
            tmp.SetDefaultInsideValue(1)
            parent_spatial.AddChild(tmp)
            parent_spatial.Update()

        if nerve is not None: # whether it has a nerve        
            nerve_spatial = util.polygon_from_array(nerve)#, world_transform=util.get_image_transform(img))

            tmp = nerve_spatial.Clone()
            tmp.SetObjectToWorldTransform(util.get_image_transform(first_img))
            tmp.SetDefaultInsideValue(2)
            parent_spatial.AddChild(tmp)
            parent_spatial.Update()

        fil.SetInput(parent_spatial)
        fil.Update()
        ans[i] = itk.array_from_image(fil.GetOutput())

    return util.image_from_array(ans, reference_image=img)
    
def load_annotations(inputdir):
    '''
    Returns
    =======
    dictionary {name: annotation}
    '''
    files = glob(os.path.join(inputdir, '*/**.xml'))
    ans = dict()
    for f in files:
        print(f)
        tree = ET.parse(f)
        xmltype = get_xml_type(tree=tree)
        if xmltype == 'image':
            for a in parse_image_annotation(tree=tree): # likely multiple annotations in the single file
                ans[a.name] = a
        elif xmltype == 'video':
            v = parse_video_annotation(tree=tree)
            ans[v.name] = v
            
    return ans

class ImageAnnotation:
    '''
    Polygon labeling of eye and optic nerve in an image.
    '''
    def __init__(self, file_base, eye=None, nerve=None):
        '''
        Attributes
        ----------
        file_base : str
            Name of file without extension
        frame_count : int
            Number of frames
        eye : ndarray or None
            ndarray [Mx2] points representing an open polygon
            NOTE: the points are in [i,j] format (ITK index), NOT [r,c] (ndarray index)
        nerve : ndarray or None
            ndarray [Nx2] points representing an open polygon
            NOTE: the points are in [i,j] format (ITK index), NOT [r,c] (ndarray index)

        '''
        self.file_base = file_base
        self.eye = eye
        self.nerve = nerve
        
    def has_eye(self):
        return self.eye is not None
    
    def has_nerve(self):
        return self.nerve is not None
    
    def transform(self, transform):
        '''
        Affine transform all annotations.  For example, if need to convert to physical space or cropped image.
        
        Parameters
        ----------
        transform : ndarray [3x3]
            Affine transform
        '''
        self.eye = affine2D(self.eye, transform)
        self.nerve = affine2D(self.nerve, transform)

class VideoAnnotation:
    '''
    Polygon labeling of eye and optic nerve in video.
    '''
    def __init__(self, file_base, eyes, nerves):
        '''
        
        Attributes
        ----------
        file_base : str
            Name of file without extension
        frame_count : int
            Number of frames
        eyes : list[ndarray]
            list of len(eyes) = frame count
            eyes[i] = None if no eye present
            eyes[i] = ndarray [Mx2] points representing an open polygon
            NOTE: the points are in [i,j] format (ITK index), NOT [r,c] (ndarray index)

        nerves : list[ndarray]
            list of len(nerves) = frame count
            nerves[i] = None if no nerve present
            nerves[i] = ndarray [Nx2] points representing an open polygon
            NOTE: the points are in [i,j] format (ITK index), NOT [r,c] (ndarray index)

        '''
        self.file_base = file_base
        self.frame_count = len(eyes)
        self.eyes = eyes
        self.nerves = nerves
    
    def get(self, frm):
        return self.eyes[frm], self.nerves[frm]
    
    def __len__(self):
        return self.frame_count
    
    def _not_none(self, xs):
        return np.argwhere([x is not None for x in xs]).squeeze()
    
    def get_eye_indices(self):
        return self._not_none(self.eyes)
    
    def get_nerve_indices(self):
        return self._not_none(self.nerves)
    
    def has_eye(self, frm):
        return self.eyes[frm] is not None
    
    def has_nerve(self, frm):
        return self.nerves[frm] is not None
    
    def transform(self, transform):
        '''
        Affine transform all annotations.  For example, if need to convert to physical space or cropped image.
        
        Parameters
        ----------
        transform : ndarray [3x3]
            Affine transform
        '''
        self.eyes = [affine2D(x, transform) for x in self.eyes]
        self.nerves = [affine2D(x, transform) for x in self.nerves]
    


#need what.????????????
#points with a way to join against them
#dictionary for sure
# main point will be a 
# images:
#  1. name
#  2. eye - pts or none
#  3. nerve - pts or none
# videos:
#  1. name
#  2. frame_rate
#  3. frame_cnt
#  4. transform
#  5. list eye
#  6. list nerve