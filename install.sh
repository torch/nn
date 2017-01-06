#!/usr/bin/env bash

SKIP_RC=0
BATCH_INSTALL=0

THIS_DIR=$(cd $(dirname $0); pwd)
PREFIX=${PREFIX:-"~/torch/install"}
TORCH_LUA_VERSION=${TORCH_LUA_VERSION:-"LUAJIT21"} # by default install LUAJIT21

while getopts 'bsh:' x; do
    case "$x" in
        h)
            echo "usage: $0
This script will install nn packages into $PREFIX.

    -b      Run without requesting any user input (will automatically add PATH to shell profile)
    -s      Skip adding the PATH to shell profile
"
            exit 2
            ;;
        b)
            BATCH_INSTALL=1
            ;;
        s)
            SKIP_RC=1
            ;;
    esac
done


# Scrub an anaconda/conda install, if exists, from the PATH.
# It has a malformed MKL library (as of 1/17/2015)
OLDPATH=$PATH
if [[ $(echo $PATH | grep conda) ]]; then
    export PATH=$(echo $PATH | tr ':' '\n' | grep -v "conda[2-9]\?/bin" | grep -v "conda[2-9]\?/lib" | grep -v "conda[2-9]\?/include" | uniq | tr '\n' ':')
fi

echo "Prefix set to $PREFIX"

if [[ `uname` == 'Linux' ]]; then
    export CMAKE_LIBRARY_PATH=/opt/OpenBLAS/include:/opt/OpenBLAS/lib:$CMAKE_LIBRARY_PATH
fi
export CMAKE_PREFIX_PATH=$PREFIX

git submodule update --init --recursive

# If we're on OS X, use clang
if [[ `uname` == "Darwin" ]]; then
    # make sure that we build with Clang. CUDA's compiler nvcc
    # does not play nice with any recent GCC version.
    export CC=clang
    export CXX=clang++
fi
# If we're on Arch linux, use gcc v5
if [[ `uname -a` == *"ARCH"* ]]; then
    path_to_gcc5=$(which gcc-5)
    if [ -x "$path_to_gcc5" ]; then
      export CC="$path_to_gcc5"
    else
      echo "Warning: GCC v5 not found. CUDA v8 is incompatible with GCC v6, if installation fails, consider running \$ pacman -S gcc5"
    fi
fi


echo "Installing core Torch packages"
cd ${THIS_DIR}      && $PREFIX/bin/luarocks make rocks/nn-scm-1.rockspec      || exit 1


