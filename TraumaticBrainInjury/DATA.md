# Data Policy
Data is mostly managed by the [tbitk.data_manager](python/tbitk/data_manager.py) module and its convenience methods.  As different sets of data and types of data come into the project, we have defined a general set of expectations regarding data and file structure.  For Jupyter notebooks, locally-retrieved data should be stored in the `TraumaticBrainInjury/Data` directory.

## Security
1. Non-human test data may be stored in Git (of reasonable file size).  Otherwise (e.g., internal use-only data, human study data (even if public)), data will be stored locally and shared using other mechanisms (e.g. shared drives like `Proj_MTEC_TBI`).  Large non-human test data may in the future be stored in something like the ITK/data.kitware.com paradigm (Git-committed hash files and a download mechanism).

## Basic Assumptions
1. A dataset (a common set of data with similar structure) has its own directory (refered to as `data_dir`).
2. A dataset's original/unmodified data is stored in `data_dir/raw` with any subdirectory structure.
3. A dataset may encode meta data in its subdirectory structure or a file's name.
4. A file's name in a dataset is unique within a subdirectory (as a matter of course) but may not be unique in the dataset.

These basic assumptions can be exploited using `tbitk.data_manager.parse_filepath()`, a convenience function for parsing and retrieving the data relevant to a file.  These assumptions also imply that derivative data (such as a preprocessed image) need to maintain the subdirectory structure of the original file for the naming to be unique.  Other convenience methods (for example, converting OS-specific file paths to POSIX paths) exist.

## Image and Video Assumptions
1.  Original images and videos typically need to be preprocessed (e.g., cropped, intensity-normalized).
2.  `data_dir/preprocessed` is the location of preprocessed files.
3.  Images and video should be converted to the `.mha` format for convenience.  Any resulting metadata will be store in `.pickle` format alongside the image/video.
4.  The file naming and directory structure are preserved unless there is not a one-to-one correspondence between raw and preprocessed data.
5.  A list of compatible image and video file formats are maintained in `tbitk.data_manager.IMAGE_EXTENSIONS` and `VIDEO_EXTENSIONS`.

## Optic Nerve Ultrasound Data Assumptions
1.  By above, preprocessed data is in `data_dir/preprocessed`
2.  _Region annotations_ for the nerve and eye are stored in `data_dir/annotations` as `.pickle` files contain `tbitk.cvat.ImageAnnotation` or `tbitk.cvat.VideoAnnotation` objects, and label .mha files (0 bg, 1 eye, 2 nerve)
3.  _Region annotations_ are currently done by [CVAT](https://github.com/openvinotoolkit/cvat)
	1.  Multiple images can be in a single CVAT task or .xml file.
	2.  A single video is annotated by a single task .xml file.
	3.  The names of the files within a CVAT task are unique
		1. NOTE: this has a stricter assumption that the names in the CVAT task are unique (not only unique with the subdirectory)
		2. This assumption is codified in `tbitk.data_manager.parse_annotation` and in [tbitk.cvat](python/tbitk/cvat.py)
	4.  Eye and nerve annotations should be named _eye_ and _nerve_ respectively.
	5.  Eye and nerve annotations are created using the _polygon_ tool in CVAT.  
	6.  For video data, the a single eye and nerve track should be created.  Frames missing the nerve or eye are annotated by using the _occluded_ property in CVAT.
	7.  Original CVAT xml files are stored `data_dir/cvat_raw`
4.  There is only at most one _region annotation_ per file.  As opposed to optic nerve sheath diameter measurements, where mutliple measurements are key to the study.

## Data Wrangling and Analysis
Any processed data will be linked back to the original filename when a one-to-one correspondence exists.  Additional metadata, such as human subject demographics, whether the image is of the left or right eye, etc. can be stored in `pandas` dataframes.  Uniqueness of subdirectory/filenames means that the relative path from `data_dir` is a unique identifier for files.

An open question is how to balance encoding information in filenames and directory structure vs encoding information in accompanying metadata files.  The former allows convenient command-line scripting but becomes unwieldy the more information that needs to be recorded.  The other extreme is a flat directory structure with uuid-named files with all metadata being stored elsewhere.
