'''
Handles the busy work of taking raw data and annotations and making it into a Pandas data frame for loading into MONAI.
'''

import pickle
import itk
import itkpocus.clarius as clarius
import tbitk.cvat as cvat
import tbitk.data_manager as dm
import os.path
import uuid

import pandas as pd
import tbitk.util as util
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from skvideo.io import vread, ffprobe
from tbitk.util import extract_slice, window_sample
from random import shuffle
from argparse import ArgumentParser

IMAGE_DATA_COLUMNS=['name', 'image_file', 'seg_file', 'spacing', 'crop', 'annotation', 'has_nerve', 
         'has_eye', 'c1pts', 'c2pts', 'experiment_id', 'device_type', 'subject_id', 'acquisition_id']

VIDEO_DATA_COLUMNS=columns=['name', 'image_file', 'seg_file', 'spacing', 'crop', 'annotation', 'experiment_id', 'device_type', 'subject_id', 'acquisition_id']
# for simplicity, keeping image and mixed data the same

def extract_frame(vid_row, img, seg_img, frame, mixed_outdir):
    '''
    Extracts specified frame from video img and seg_img, returning the extracted frame and its data row
    
    Parameters
    ----------
    vid_row (pandas DataSet): data row corresponding to the video, has VIDEO_DATA_COLUMNS
    img (itk.Image 3D): video image to extract from (X x Y x frames)
    seg_img (itk.Image 3D): video segmentation image to extract from
    frame: index of frame
    mixed_outdir: output directory used to calculate output filenames
    
    Returns
    frame_img (itk.Image), frame_seg_img (itk.Image), frame_row (tuple) 
    '''
    name = vid_row['name'] + '_' + str(frame)
    annotation = vid_row['annotation']
    image_file = os.path.join(mixed_outdir, name + '.mha')
    seg_file = os.path.join(mixed_outdir, name + '-seg.mha')
    img2 = extract_slice(img, frame, 0)
    seg_img2 = extract_slice(seg_img, frame, 0)
    spacing = vid_row['spacing']
    crop = vid_row['crop']
    eye = annotation.eyes[frame]
    nerve = annotation.nerves[frame]
    img_annotation = dm.ImageAnnotation(name, eye, nerve)
    has_nerve = nerve is not None
    has_eye = eye is not None
    c1pts = None
    c2pts = None
    exp_id = vid_row['experiment_id']
    device_type = vid_row['device_type']
    subject_id = vid_row['subject_id']
    acquisition_id = vid_row['acquisition_id']
    return img2, seg_img2, (name, image_file, seg_file, spacing, crop, img_annotation, has_nerve, has_eye, c1pts, c2pts, exp_id, device_type, subject_id, acquisition_id)

def finalize_images(annotations, inputdir, outdir, fn_experiment_id, fn_device_type, fn_subject_id, fn_acquisition_id):
    '''
    Parameters:
    annotations (dict) : name to annotation (VideoAnnotation or ImageAnnotation)
    inputdir : directory to search for images
    outdir : directory to place pre-processed images
    '''
#   IMAGE_DATA_COLUMNS=['name', 'image_file', 'seg_file', 'spacing', 'crop', 'annotation', 'has_nerve', 
#   'has_eye', 'c1pts', 'c2pts', 'experiment_id', 'device_type', 'subject_id', 'acquisition_id']
    img_files = glob(os.path.join(inputdir, '**/*.PNG'), recursive=True) + glob(os.path.join(inputdir, '**/*.JPG'), recursive=True)
    columns=IMAGE_DATA_COLUMNS
    data = []
    for f in img_files:
        device_type = fn_device_type(f)
        if device_type == 'clarius':
            npimg = itk.array_from_image(itk.imread(f))
            img, spacing, crop = clarius.preprocess_image(npimg)
            c1pts = None
            c2pts = None

        name = os.path.splitext(os.path.basename(f))[0]
        prefix = os.path.join(vid_outdir, name)
        image_file = prefix + '.mha'
        itk.imwrite(img, image_file)
        seg_file = prefix + '-seg.mha'
        try:
            annotation = annotations[name]
        except KeyError:
            annotation = None

        if annotation is not None:
            # handle the fact we annotated on uncropped images without spacing information
            translate_transform = np.array([[1, 0, -crop[1,0]], [0, 1, -crop[0,0]], [0, 0, 1]])
            annotation.transform(translate_transform)

            # takes polygons in index space and returns an image with correct physical coordinates
            seg_img = util.image_from_spatial([util.polgyon_from_array(annotation.eye), util.polgyon_from_array(annotation.nerve)], img, inside_value=[1.0,2.0])

            # scale our polygons correctly so we can make physical measurements of the eye/nerve
            # everything should now be the correct physical space as img and seg_img
            scale_transform = np.array([[spacing, 0, 0], [0, spacing, 0], [0, 0, 1]])
            annotation.transform(scale_transform)
        else: # make a blank label image
            seg_img = util.image_from_array(np.zeros(img.GetLargestPossibleRegion().GetSize(), dtype='float32').T, 
                                            spacing=img.GetSpacing(), direction=img.GetDirection(), origin=img.GetOrigin())

        itk.imwrite(seg_img, seg_file)
        has_nerve = annotation is not None and annotation.has_nerve()
        has_eye = annotation is not None and annotation.has_eye()
        experiment_id = fn_experiment_id(f)
        subject_id = fn_experiment_id(f)
        acquisition_id = fn_acquisition_id(f)
        data.append([name, image_file, seg_file, spacing, crop, annotation, has_nerve, has_eye, c1pts, c2pts, experiment_id, device_type, subject_id, acquisition_id])

    data_img = pd.DataFrame(data, columns=columns)
    return data_img

# ok, for video
# don't have 

def finalize_video(annotations, inputdir, outdir, fn_experiment_id, fn_device_type, fn_subject_id, fn_acquisition_id):
    '''
    
    Parameters
    ==========
    annotations (dict)
    inputdir (filepath)
    outdir (filepath)
    '''
    #VIDEO_DATA_COLUMNS=columns=['name', 'image_file', 'seg_file', 'spacing', 'crop', 'annotation', 'experiment_id', 'device_type', 'subject_id', 'acquisition_id']
    vid_files = glob(os.path.join(inputdir, '**/*.MP4'), recursive=True) + glob(os.path.join(inputdir, '**/*.AVI'), recursive=True)
    
    columns=VIDEO_DATA_COLUMNS
    data = []
    for f in vid_files:
        device_type = fn_device_type(f)
        if device_type == 'clarius':
            npimg = vread(f) # itk.array_from_image(itk.imread(f))
            meta_dict = ffprobe(f)
            img, spacing, crop = clarius.preprocess_video(npimg, util.get_framerate(meta_dict))

        name = os.path.splitext(os.path.basename(f))[0]
        prefix = os.path.join(vid_outdir, name)
        image_file = prefix + '.mha'
        itk.imwrite(img, image_file)
        seg_file = prefix + '-seg.mha'
        try:
            annotation = annotations[name]
        except KeyError:
            annotation = None

        if annotation is not None:
            # handle the fact we annotated on uncropped images without spacing information
            translate_transform = np.array([[1, 0, -crop[1,0]], [0, 1, -crop[0,0]], [0, 0, 1]])
            annotation.transform(translate_transform)

            frame_size = np.array(img.GetLargestPossibleRegion().GetSize())[0:2]
            npseg = None
            for i in range(len(annotation)):
                eye, nerve = annotation.get(i) # get the eye/nerve polygons from annotation
                tmp = util.array_from_spatial([util.polgyon_from_array(eye), util.polgyon_from_array(nerve)], frame_size, inside_value=[1.0,2.0])
                npseg = tmp[None,:,:] if npseg is None else np.append(npseg, tmp[None,:,:], axis=0)
            
            # takes polygons in index space and returns an image with correct physical coordinates
            seg_img = util.image_from_array(npseg.astype('float32'), img.GetSpacing(), img.GetDirection(), img.GetOrigin())

            # scale our polygons correctly so we can make physical measurements of the eye/nerve
            # everything should now be the correct physical space as img and seg_img
            scale_transform = np.array([[spacing, 0, 0], [0, spacing, 0], [0, 0, 1]])
            annotation.transform(scale_transform)
        else: # make a blank label image
            seg_img = util.image_from_array(np.zeros(img.GetLargestPossibleRegion().GetSize(), dtype='float32').T, 
                                            spacing=img.GetSpacing(), direction=img.GetDirection(), origin=img.GetOrigin())

        itk.imwrite(seg_img, seg_file)
        experiment_id = fn_experiment_id(f)
        subject_id = fn_subject_id(f)
        acquisition_id = fn_acquisition_id(f)
        data.append([name, image_file, seg_file, spacing, crop, annotation, experiment_id, device_type, subject_id, acquisition_id])

    data_vid = pd.DataFrame(data, columns=columns)
    return data_vid

def random_id(name):
    '''
    Returns a random ID
    '''
    return uuid.uuid4()

def finalize_mixed(vid_data, img_data, frame_spacing=2):
    '''
    Creates a final image-based dataset, extracting spaced images from the video data and combining it with the image data.
    
    Parameters
    ==========
    frame_spacing (float) : minimum spacing between extracted frames in seconds
    '''
#   IMAGE_DATA_COLUMNS=['name', 'image_file', 'seg_file', 'spacing', 'crop', 'annotation', 'has_nerve', 
#   'has_eye', 'c1pts', 'c2pts', 'experiment_id', 'device_type', 'subject_id', 'acquisition_id']
    data = []
    columns = IMAGE_DATA_COLUMNS
    for i in range(len(data_vid)):
        vid_row = data_vid.loc[i]
        annotation = vid_row['annotation']
    #     print(vid_row)
        if annotation is not None:
            nerves = set(annotation.get_nerve_indices())
            eyes = set(annotation.get_eye_indices())
            eyes_not_nerves = eyes - nerves
            blanks = set(range(annotation.frame_count)) - (eyes | nerves)

            # sets are disjoint, subsample the frames so that we aren't just getting adjacent near-identical samples
            img = itk.imread(vid_row['image_file'])
            seg_img = itk.imread(vid_row['seg_file'])
            sampling_window = frame_spacing / img.GetSpacing()[0] # convert frame_spacing in seconds to frame index distance

            idxs = window_sample(list(nerves), sampling_window) + window_sample(list(eyes_not_nerves), sampling_window) + window_sample(list(blanks), sampling_window)

            # convert to image annotation
            for idx in idxs:
                img2, seg_img2, row = extract_frame(vid_row, img, seg_img, idx, mixed_outdir)
                data.append(row)
                itk.imwrite(img2, row[1])
                itk.imwrite(seg_img2, row[2])
        
    data_mixed = pd.DataFrame(data, columns=columns)
    return data_mixed
                
if __name__ == '__main__':
    parser = ArgumentParser(description='Convert image/video files and their optional annotations into a preprocessed dataset.')
    parser.add_argument('input_dir', type=str)
    parser.add_argument('annotation_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('experiment_id', type=str, default=None)
    parser.add_argument('device_type', type=str)
    parser.add_argument('subject_id', type=str, default=None)
    args = parser.parse_args()
    print(args)
    
    
    # i know this is weird, this is a placedholder for a future API
    # basically, either want these values specified or a parser for them (to get out of the filename)
    # acquisition_id is weird, because the default shouldn't be constant but random
    assert args.experiment_id is not None
    assert args.device_type is not None
    assert args.subject_id is not None
    
    
    if args.experiment_id is not None:
        fn_experiment_id = lambda x: args.experiment_id
    
    if args.device_type is not None:
        fn_device_type = lambda x: args.device_type
    
    if args.subject_id is not None:
        fn_subject_id = lambda x: args.subject_id
        
    fn_acquisition_id = random_id
    
    
#     inputdir = r'..\data\clarius-sonosite eye test-20200923\clarius\human'
#     annot_dir = r'..\data\clarius-sonosite eye test-20200923-annotated'
#     device_type = 'clarius'
#     outdir = r'..\data\human-preprocessed'
#     experiment_id = 'duke eye test'
#     subject_id = '1'
#     acquisition_id = '1'

    img_outdir = os.path.join(args.output_dir, 'images')
    vid_outdir = os.path.join(args.output_dir, 'video')
    mixed_outdir = os.path.join(args.output_dir, 'mixed')

    os.makedirs(img_outdir, exist_ok=True)
    os.makedirs(vid_outdir, exist_ok=True)
    os.makedirs(mixed_outdir, exist_ok=True)

    annotations = cvat.load_annotations(args.annotation_dir)
    
    print(annotations)
    data_img = finalize_images(annotations, args.input_dir, img_outdir, fn_experiment_id, fn_device_type, fn_subject_id, fn_acquisition_id)
    data_img.to_pickle(os.path.join(img_outdir, args.experiment_id + '.pickle.gz'))
    data_vid = finalize_video(annotations, args.input_dir, vid_outdir, fn_experiment_id, fn_device_type, fn_subject_id, fn_acquisition_id)
    data_img.to_pickle(os.path.join(vid_outdir, args.experiment_id + '.pickle.gz'))
    data_mixed = finalize_mixed(data_img, data_vid)
    data_mixed.to_pickle(os.path.join(mixed_outdir, args.experiment_id + '.pickle.gz'))
    data_total = data_img.append(pd.DataFrame(data_mixed, columns=IMAGE_DATA_COLUMNS), ignore_index=True)
    data_total.to_pickle(os.path.join(args.output_dir, args.experiment_id + '.pickle.gz'))
    print(data_total)