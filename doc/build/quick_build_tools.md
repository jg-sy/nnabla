# Quick build tools

We provide easiest way to build NNabla with GNU make and docker.
On windows there is helper BATCH files.

## Linux

### Prepare
You need to install just [Docker](https://docs.docker.com/install/) and `make`. 

### Build
```
cd nnabla
$ make all
```

`make all` is as same as `make bwd-cpplib bwd-wheel`
After this you can find .whl file in `nnabla/build_wheel/dist/`


### Build cpplib only

If you want only cpp libraries.
```
cd nnabla
$ make bwd-cpplib
```
After this you can find executable file and shared library in `nnabla/build/lib/` and `nnabla/build/bin/`

### Specify python version

Prepare to specify python version.
```
$ export PYTHON_VERSION_MAJOR=3
$ export PYTHON_VERSION_MINOR=8
```

Then you can get with,
```
$ make all
```

Or you can specify every time.
```
$ make PYTHON_VERSION_MAJOR=3 PYTHON_VERSION_MINOR=8 all
```

## Windows

### Prepare

Please see [Official site](https://chocolatey.org/install)
After installing Chocolatey do following command on Administrator cmd.exe.
```
choco feature enable -n allowGlobalConfirmation
choco install cmake git visualstudio2019-workload-vctools visualstudio2019buildtools
pip install pywin32 Cython boto3 protobuf h5py ipython numpy pip pytest scikit-image scipy wheel pyyaml mako tqdm
```

### Build cpplib
```
> call build-tools\msvc\build_cpplib.bat
```

### Build wheel
```
> call build-tools\msvc\build_wheel.bat PYTHON_VERSION
```
The PYTHON_VERSION we tested is 3.7, 3.8 3.9 and 3.10.

