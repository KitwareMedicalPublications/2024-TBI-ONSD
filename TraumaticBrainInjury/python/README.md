# Setting up Python Environment

1. Create and activate a virtual environment
```
python -m venv venv-my-virtual # or whatever you want to name it
# source venv-my-virtual/bin/activate # for linux
source venv-my-virtual/Scripts/activate # windows
```
2. Install pip prerequisites
```
pip install -r requirements.txt
```
3. Install local build of itk-pocus to virtual environment
```
# make sure your venv-my-virtual is activated
pip install flit
cd /path/to/ITKPOCUS/itkpocus
pip install -r requirements.txt
flit install --pth-file # will allow changes to itkpocus to "sync"
```

Note, MONAI is listed in the `requirements.txt`.  If you have issues with PyTorch, try installing PyTorch using the command-line from their website (using the same virtual environment) before executing Step 2. 
