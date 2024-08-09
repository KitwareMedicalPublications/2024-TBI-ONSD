# TBITK
## "Installing" tbitk

From the root directory of the repository, run:

1. Create a new virtual environment:
```
$ virtualenv {env_name} -p python3.9
```

and activate:

```
$ source {env_name}/bin/activate
```

2. Install the python dependencies:
```
$ pushd TraumaticBrainInjury/python/tbitk
$ pip install -r requirements.txt
```

3. Create tbitk.pth containing the absolute path to the TraumaticBrainInjury/python/tbitk directory:
```
$ echo $(pwd) > tbitk.pth
```

4. Move ``tbitk.pth`` to the site-packages directory of you virtual environment:
```
$ mv tbitk.pth $VIRTUAL_ENV/lib/python3.9/site-packages/.
$ popd
```
