#!/usr/bin/env bash


function setup_build() {
  export HOME_DIR="$PWD"
  export PACKAGE_HOME_DIR="$HOME_DIR/packages"
  export INSTALL_PREFIX_DIR="$HOME_DIR/local"

  if [[ -d $INSTALL_PREFIX_DIR ]]; then
    rm -rf $INSTALL_PREFIX_DIR
  fi

  mkdir $INSTALL_PREFIX_DIR

  if [[ -d $PACKAGE_HOME_DIR ]]; then
    rm -rf $PACKAGE_HOME_DIR
  fi

  mkdir $PACKAGE_HOME_DIR
}


function install_capstone() {
  git clone -n --single-branch --branch next https://github.com/aquynh/capstone
  cd capstone
  git checkout -b bcov_artifact c3b4ce1901
  echo "building capstone ... "
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX_DIR" .. && make -j 4
  echo "installing capstone ... "
  make install
}


function install_unicorn() {
  git clone -n --single-branch --branch master https://github.com/unicorn-engine/unicorn
  cd unicorn
  git checkout -b bcov_artifact 536c4e77c4
  echo "building unicorn ... "
  make -j 4
  echo "installing unicorn ... "
  make install DESTDIR="$INSTALL_PREFIX_DIR"
}


function install_bcov() {
  git clone -n --single-branch --branch master https://github.com/abenkhadra/bcov
  cd bcov
  git checkout -b bcov_artifact 4389af5ad4
  echo "building bcov ... "
  mkdir build
  cd build
  # Set the build type to "Debug" for a more verbose output
  cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX_DIR" \
  -DCMAKE_INCLUDE_PATH="$INSTALL_PREFIX_DIR/include/;$INSTALL_PREFIX_DIR/usr/include/" \
  -DCMAKE_LIBRARY_PATH="$INSTALL_PREFIX_DIR/lib/;$INSTALL_PREFIX_DIR/usr/lib/"  ..
  make -j 4
  echo "installing bcov ... "
  make install
}


setup_build
cd $PACKAGE_HOME_DIR && install_capstone
cd $PACKAGE_HOME_DIR && install_unicorn
cd $PACKAGE_HOME_DIR && install_bcov
