REM Copyright 2018,2019,2020,2021 Sony Corporation.
REM Copyright 2021 Sony Group Corporation.
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.
REM 

REM 
REM Usage:
REM   build_cpplib.bat [PYTHON_VERSION] [VISUAL_STUDIO_EDITION]
REM     (optional) PYTHON_VERSION: 3.7, 3.8, 3.9 or 3.10
REM     (optional) VISUAL_STUDIO_EDITION: 2015 or 2019(experimental)
REM
@ECHO ON
SETLOCAL

REM Environment
IF [%1] == [] (
    CALL %~dp0tools\env.bat 3.8 %1 || GOTO :error
) ELSE (
    CALL %~dp0tools\env.bat %1 %2 || GOTO :error
)
SET third_party_folder=%nnabla_root%\third_party

REM Build third party libraries.
CALL %~dp0tools\build_zlib.bat       || GOTO :error
CALL %~dp0tools\build_libarchive.bat || GOTO :error
CALL %~dp0tools\build_protobuf.bat   || GOTO :error
CALL %~dp0tools\build_hdf5.bat       || GOTO :error


REM Get pre-built lz4 and zstd libraries
CALL %~dp0tools\get_liblz4.bat || GOTO :error
CALL %~dp0tools\get_libzstd.bat || GOTO :error

@ECHO ON

REM Build CPP library.
ECHO "--- CMake options for C++ build ---"
set nnabla_debug_options=
IF [%build_type%] == [Debug] SET nnabla_debug_options=-DCMAKE_CXX_FLAGS="/bigobj"

ECHO -----------------------------------------
ECHO nnabla_build_folder=%nnabla_build_folder%
ECHO -----------------------------------------

IF NOT EXIST %nnabla_build_folder% MD %nnabla_build_folder%
IF NOT EXIST %nnabla_build_folder%\bin MD %nnabla_build_folder%\bin
IF NOT EXIST %nnabla_build_folder%\bin\%build_type% MD %nnabla_build_folder%\bin\%build_type%

powershell ^"Get-ChildItem %third_party_folder% -Filter *.dll -Recurse ^| ForEach-Object {Copy-Item $_.FullName %nnabla_build_folder%\bin\%build_type%}^"
powershell ^"Get-ChildItem %third_party_folder% -Filter *.lib -Recurse ^| ForEach-Object {Copy-Item $_.FullName %nnabla_build_folder%\bin\%build_type%}^"
powershell ^"Get-ChildItem %third_party_folder% -Filter *.exp -Recurse ^| ForEach-Object {Copy-Item $_.FullName %nnabla_build_folder%\bin\%build_type%}^"

CD %nnabla_build_folder%

cmake -G "%generate_target%" ^
      -DBUILD_CPP_UTILS=ON ^
      -DBUILD_TEST=ON ^
      -DBUILD_PYTHON_PACKAGE=OFF ^
      -Dgtest_force_shared_crt=TRUE ^
      -DLIB_NAME_SUFFIX=%lib_name_suffix% ^
      -DLibArchive_INCLUDE_DIR=%libarchive_include_dir% ^
      -DLibArchive_LIBRARY=%libarchive_library% ^
      -DPROTOC_COMMAND=%protobuf_protoc_executable% ^
      -DPYTHON_COMMAND_NAME=python ^
      -DProtobuf_INCLUDE_DIR=%protobuf_include_dir% ^
      -DProtobuf_LIBRARY=%protobuf_library% ^
      -DProtobuf_LITE_LIBRARY=%protobuf_lite_library% ^
      -DProtobuf_PROTOC_EXECUTABLE=%protobuf_protoc_executable% ^
      -DZLIB_INCLUDE_DIR=%zlib_include_dir% ^
      -DZLIB_LIBRARY_RELEASE=%zlib_library% ^
      -DNNABLA_UTILS_WITH_HDF5=ON ^
      %nnabla_debug_options% ^
      %nnabla_root% || GOTO :error

cmake --build . --config %build_type% || GOTO :error
cmake --build . --config %build_type% --target test_nbla_utils || GOTO :error
cpack -G ZIP -C %build_type%

GOTO :end
:error
ECHO failed with error code %errorlevel%.
exit /b %errorlevel%

:end
ENDLOCAL
