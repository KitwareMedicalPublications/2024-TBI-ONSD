import numpy as np
import pandas as pd
import itk

from glob import glob
import os
import os.path
from pathlib import Path

import pickle
import tempfile
import time

from tbitk import util
from itkpocus import butterfly
from tbitk import cvat
from itkpocus import sonosite
from itkpocus import clarius
from itkpocus import interson
from itkpocus import sonoque
from itkpocus import sonivate

# Seeing this sort of broken as .mha files can represent both
# TODO move these checks into functions
IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg', '.nrrd']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mha', '.cine', '.mov']

def canonical(fp):
    '''
    Returns fp as an absolute, resolved, POSIX ('/' delimeter), file path
    '''
    return str(Path(fp).absolute().resolve().as_posix())

def canonical_relpath(fp, basedir):
    '''
    Returns the filepath of fp relative to basedir
    '''
    return canonical(fp).replace(canonical(basedir) + '/', '')

def parse_filepath(fp, basedir, filename_sep=None, _splitext=None):
    '''
    Parses a filepath relative to given a basedir and returns subdirectories, filename-encoded information, etc.
    
    Parameters
    ----------
    fp (str) : filepath to parse
    basedir (str) : base or root directory (defines subdirectories to fp)
    filename_sep (str) : separator to parse out of the filename (if the filename itself encodes data)
    _splitext (function) : another _splitext to use (e.g. you need something that will handle .tiff.gz or other combo-extensions)
    
    Returns
    -------
    subdirs : list[str]
        List of the parsed subdirectories (or empty list if none)
    filename : str
        Filename
    file_base : str
        Filename without extension
    file_ext : str
        File extension
    file_parse : list[str] or None
        None (if filename_sep is None) or list of separated fields by sep
        
    Examples
    --------
    >>> parse_filepath('../../blargh/something/me/test.txt', '../..')
    (['blargh', 'something', 'me'],
     'test.txt',
     'test',
     '.txt',
     None)
    '''
    
    # Consider adding option for filename_regex
    
    _split_ext = os.path.splitext if _splitext is None else _splitext
    
    # TODO replace with call to canonical_relpath
    abs_fp = canonical(fp)
    abs_basedir = canonical(basedir)
    sub_fp = abs_fp.replace(abs_basedir + '/', '')
    tmp = sub_fp.split('/')
    subdirs = tmp[:-1]
    filename = tmp[-1]
    tmp2 = _split_ext(filename)
    file_base = tmp2[0]
    file_ext = tmp2[1]
    file_parse = None if filename_sep is None else file_base.split(filename_sep)
    
    return subdirs, filename, file_base, file_ext, file_parse

def get_file_type(fp):
    '''
    Returns 'image' or 'video' depending on file extension.  Fails assert if neither.
    
    Parameters
    ----------
    fp : str
        Filepath
    '''
    ext = os.path.splitext(fp)[1].lower()
    ans = None
    if ext in IMAGE_EXTENSIONS:
        ans = 'image'
    elif ext in VIDEO_EXTENSIONS:
        ans = 'video'
    else:
        assert False
    return ans
    
def get_filepaths(fp):
    '''
    Returns all filepaths (i.e. raw image, annotations, preprocessed image) associated with fp.
    
    This is meant as a convenience function for finding all files associated with another file.
    
    Parameters
    ----------
    fp : str
    
    Returns
    -------
    dict
    '''

    canon_fp = canonical(fp)
    canon_split = canon_fp.split('/')
    base_dir = None
    for i in reversed(range(len(canon_split))):
        if canon_split[i] in ['raw', 'annotation', 'preprocessed']:
            base_dir = '/'.join(canon_split[:i])
            next_dir = canon_split[i]
    assert base_dir is not None
    
    subdirs, filename, file_base, file_ext, file_parse = parse_filepath(fp, base_dir + '/' + next_dir)
    subdirs_join = '/'.join(subdirs)
    if subdirs_join:
        subdirs_join += "/"

    raw = posix_glob(base_dir + '/raw/' + subdirs_join + file_base + '.*')
    assert len(raw) == 1
    raw = raw[0]
    
    preprocessed = base_dir + '/preprocessed/' + subdirs_join + file_base + '.mha'
    preprocessed_meta = base_dir + '/preprocessed/' + subdirs_join + file_base + '.pickle'
    annotation = base_dir + '/annotation/' + subdirs_join + file_base + '.pickle'
    annotation_label_image = base_dir + '/annotation/' + subdirs_join + file_base + '-label.mha'
    return {
        'raw' : raw,
        'preprocessed' : preprocessed,
        'preprocessed_meta' : preprocessed_meta,
        'annotation' : annotation,
        'base_dir' : base_dir,
        'annotation_label_image' : annotation_label_image
    }

def process_raw_directory(mydir, only_new=True, **kwargs):
    '''
    Load raw images and video from a directory and preprocess them.

    Given a dataset directory, preprocess all raw image and videos found.
    
    Parameters
    ----------
    mydir : str
        Path to dataset directory, e.g., with a subdirectory raw/
    only_new : bool
        If True, ignore already preprocessed files
    **kwargs :
        Additional keywords to pass to process_raw

    Returns
    -------
    dict
        str : (bool, None or Exception)
        Returns a dict where each key a file considered and the value is a tuple (bool, None or Exception)
        where (True, None) means the preprocess ran without error, (False, None) means the file already
        exists, and (False, Exception) means an error occurred.
    '''
    ans = dict()

    devices = ['butterfly-iq', 'clarius-l7hd', 'interson-spl01', 'sonoque-l5c', 'sonivate']		
    for d in devices:
        files = posix_glob(f'{mydir}/raw/**/{d}/**/*.*', recursive=True)
        for f in files:
            print(time.asctime(), f)
            ans[f] = (True, None)
            if os.path.exists(get_filepaths(f)['preprocessed']) and only_new:
                print('exists')
                ans[f] = (False, None)
            else:
                try:
                    process_raw(f, device_type=d, **kwargs)
                except Exception as e:
                    ans[f] = (False, e)
    
    return ans


def process_raw(raw_fp, device_type, no_save=False, **kwargs):
    '''
    Load a raw image or video, preprocess it, and save the results in a preprocessed directory.
    
    Crop the image, remove the overlay if possible, rescale intensity to 0.0-1.0, save RGB video to scalar.
    Creates a .mha and .json file, preserving subdirectory structure.
    
    Parameters
    ----------
    raw_fp : str
        Filepath to raw image or video file
    device_type : str
        One of ['buttefly-iq', 'clarius-l7hd', 'sonosite']
    no_save : bool
        Whether to save any of preprocessed files to disk
        
    Returns
    -------
    img : itk.Image[itk.F,2]
        The preprocessed image with correct physical spacing and intensity between 0 and 1.0
    meta : dict
        Meta data dictionary (e.g. spacing and crop)
    preprocessed_fp : str
        Path to preprocessed image file
    preprocessed_meta_fp : str
        Path to meta data dictionary file (.pickle)
        
    Examples
    --------
    >>> img, meta, fp1, fp2 = process_raw('../../data/DATA_DIR/subject1/left/image1.jpg', 'butterfly-iq', no_save=True)
    '''
    file_type = get_file_type(raw_fp)
    file_paths = get_filepaths(raw_fp)
    
    if device_type == 'sonosite':
        if file_type == 'image':
            img, meta = sonosite.load_and_preprocess_image(raw_fp)
        else:
            img, meta = sonosite.load_and_preprocess_video(raw_fp)
    elif device_type == 'clarius-l7hd':
        tick_dist = 2 if 'tick_dist' not in kwargs else kwargs['tick_dist'] # default 2mm per tick, note this goes to 10mm when zoomed out on some settings
            
        if file_type == 'image':
            img, meta = clarius.load_and_preprocess_image(raw_fp, tick_dist=tick_dist)
        else:
            img, meta = clarius.load_and_preprocess_video(raw_fp, tick_dist=tick_dist)
    elif device_type == 'butterfly-iq':
        manual_crop = None if 'manual_crop' not in kwargs else kwargs['manual_crop']
        if file_type == 'image':
            img, meta = butterfly.load_and_preprocess_image(raw_fp, manual_crop=manual_crop)
        else:
            img, meta = butterfly.load_and_preprocess_video(raw_fp, manual_crop=manual_crop)
    elif device_type == 'interson-spl01':
        if file_type == 'image':
            img, meta = interson.load_and_preprocess_image(raw_fp)
        else:
            img, meta = interson.load_and_preprocess_video(raw_fp)
    elif device_type == 'sonoque-l5c':
        if file_type == 'image':
            img, meta = sonoque.load_and_preprocess_image(raw_fp)
        else:
            img, meta = sonoque.load_and_preprocess_video(raw_fp)
    elif device_type == 'sonivate':
        if file_type == 'image':
            img, meta = sonivate.load_and_preprocess_image(raw_fp)
        else:
            img, meta = sonivate.load_and_preprocess_video(raw_fp)
    else:
        assert False, 'Unsupported device_type'
    
    if not no_save:
        os.makedirs(str(Path(file_paths['preprocessed']).parents[0]), exist_ok=True)
        itk.imwrite(img, file_paths['preprocessed'], compression=True)
        with open(file_paths['preprocessed_meta'], 'wb') as f:
            pickle.dump(meta, f)
    
    return img, meta, file_paths['preprocessed'], file_paths['preprocessed_meta']

def _transform_and_write_annotation(annotation, canon_basedir):
    '''
    Finds the preprocessed image that corresponds to annotation.file_base, loads its meta data, and
    transforms the annotation into the preprocessed image's physical space.  Saves the corresponding
    annotation.
    
    Parameters
    ----------
    annotation : ImageAnnotation or VideoAnnotation
        Annotation to transform and write
    canon_basedir : str
        Preprocessed directory in canonical form to search
    '''
    y = annotation.file_base # original file base name defined in annotation file
    z = posix_glob(canon_basedir + '/**/' + y + '.mha', recursive=True) # find the matching preprocessed image
    assert len(z) == 1
    img_file = z[0]
    
    file_paths = get_filepaths(img_file)
    with open(file_paths['preprocessed_meta'], 'rb') as f:
        meta = pickle.load(f)

    # the annotation is in i,j (ITK) index space
    # just apply crop, we want to keep the points in index space
    affine = np.eye(3)
    affine[0,2] = -meta['crop'][1,0]
    affine[1,2] = -meta['crop'][0,0]
    annotation.transform(affine)
    
    os.makedirs(str(Path(file_paths['annotation']).parents[0]), exist_ok=True)
    with open(file_paths['annotation'], 'wb') as f:
        pickle.dump(annotation, f)
        
    img = itk.imread(file_paths['preprocessed'])
    if type(annotation) == cvat.ImageAnnotation:
        label_img = cvat.get_image_label_image(img, annotation)
    else:
        label_img = cvat.get_video_label_image(img, annotation)
    
    itk.imwrite(label_img, file_paths['annotation_label_image'], compression=True)

def process_annotation(annotation_fp, basedir_preprocessed):
    '''
    Load a raw CVAT annotation .zip file, match it with the files being annotated, save it in an annotations directory.
    
    Annotations will be converted to the physical space of the preprocessed image/video.
    
    NOTE: read DATA.md on how to use CVAT and the assumptions about the xml file
    
    Parameters
    ----------
    annotation_fp : str
        Filepath to .zip file
    basedir_preprocessed :
        Directory path to (i.e. DATA_DIR/preprocessed) search for preprocessed images/videos
        
    Examples
    --------
    >>> process_annotation('../DATA_DIR/cvat_raw/task_images-2020_12_01_14_24_31-cvat for images 1.1.zip', '../DATA_DIR/preprocessed/')
    '''
    canon_basedir = canonical(basedir_preprocessed)
    with tempfile.TemporaryDirectory() as td:
        xml_fp = cvat.extract_and_rename(annotation_fp, td)
        xml_type = cvat.get_xml_type(xml_fp)
        if xml_type == 'image':
            # can have multiple files' worth of image annotations in a single .xml
            image_annotations = cvat.parse_image_annotation(xml_fp)
            for x in image_annotations:
                _transform_and_write_annotation(x, canon_basedir)
        else: # video
            video_annotation = cvat.parse_video_annotation(xml_fp)
            _transform_and_write_annotation(video_annotation, canon_basedir)

def posix_glob(pattern, recursive=False):
    '''
    Glob has a weird OS-specific behavior where you'll get mangled Windows and pattern (UNIX) mixed pathnames.  This fixes it to posix-style (i.e. forward-slash).
    
    Parameters
    ==========
    pattern (str) : UNIX-style search string (see glob)
    recursive (bool) : whether to allow ** as a pattern (see glob)
    '''
    return [Path(x).as_posix() for x in glob(pattern, recursive=recursive)]
    
