#!/usr/bin/sh

echo "Beginning Installation"
#installing build essentials (GCC4.4.x) and package config and cmake
echo "Installing build essentials,cmake,pkg-config and checkinstall"
sudo apt-get install build-essential cmake pkg-config checkinstall

#installing python dev libraries and numpy library
echo "Installing python dev and  numpy dev libraries"
sudo apt-get install python-dev python-numpy

#installing GTK2.0+ and QT4+
echo "Installing GTK2.0, QT4, TBB and other libraries"
sudo apt-get install libgtk2.0-dev libqt4-dev libtbb-dev libv4l

#installing audio and video codecs
echo "Installing audoio and video codecs"
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev ffmpeg

#installing optional libraries
echo "Installing optional libraries for image processing"
sudo apt-get install libtiff-dev libjpeg-dev libpng-dev libjasper-dev

#creating a directory for build
echo "Creating build directory"
mkdir build
cd build

#running cmake command
echo "Running CMake"
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON ..

#running make command
echo "Running make command"
make -j2

#checking install and creating and installing a debain package
echo "Checking installation and creating debian package"
sudo checkinstall

sudo ldconfig

echo "Installation complete, you can access the library for development"

