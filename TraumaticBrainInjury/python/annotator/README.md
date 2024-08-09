# Annotator

Annotator is a tool for running ImageViewer on a dataset.  The dataset is defined by a directory containing a 'preprocessed' subfolder.

## Building
```
# activate your virtual environment for building (a new one)
# we'll assume it is in ROOT/TraumaticBrainInjury/python/venv-annotator
pip install -r requirements.txt
cd /ROOT/TraumaticBrainInjury/python/annotator
pyinstaller --paths=../venv-annotator/Lib/site-packages --noconsole annotator.py
cp -R path/to/ImageViewer ROOT/TraumaticBrainInjury/python/annotator/dist/annotator
cp /ROOT/TraumaticBrainInjury/python/annotator/default_config.json ROOT/TraumaticBrainInjury/python/annotator/dist/annotator

# give the entire directory ROOT/TraumaticBrainInjury/python/annotator/dist/annotator as the deployable app
# note, the ImageViewer version must match the build environment
# pyinstaller can handled multiple OSes, but can't cross-compile, so you must build on the deployment target OS
```

## Usage
1. Double-click or execute annotator.exe (Windows) or annotator (Unix)
2. Click 'Select Folder...' and select the dataset directory (has a 'preprocessed' subfolder)
3. Select the user/annotator or create a new one
4. Select the replicate number (for multiple annotations by the same user)
5. The UI will present how many images/videos are present and how many have already been annotated by this user / replicate.
6. Click 'Start' to begin annotation
7. Annotator will open an ImageViewer window per file.  Use '<' and '>' keys to move through frames, left-click to place ONSD marks (first
ruler is for the 3mm measure, and the second is for the ONSD).  Right-click on a crosshair to move or delete a ruler.  Multiple measurements
can be made in a single video.  To save your measurement, close the ImageViewer window.  If you do not make any measurements, e.g., there
was not a good frame for measuring ONSD, simply close the ImageViewer window without making measurements.  The fact that the file was reviewed
for annotation will be stored.  If you want to stop annotating, or want to cancel annotating a video (e.g., do not record that you viewed
the file nor make a measurement), close the Annotator window.  Any open ImageViewer windows will be closed and their results not saved.  If you
had already annotated (by viewing and then closing ImageViewer) any files in that session, those results will be saved for the next time you
open Annotator.
