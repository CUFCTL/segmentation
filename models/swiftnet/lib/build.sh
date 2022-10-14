#!/usr/bin/env bash
rm cylib.so

cython -a cylib.pyx -o cylib.cc

#g++ -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing \
#-I/usr/lib/python3.7/site-packages/numpy/core/include -I/usr/include/python3.7m -o cylib.so cylib.cc
g++ -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing \
-I/usr/lib/python3.8/site-packages/numpy/core/include -I/usr/include/python3.8 -o cylib.so cylib.cc

# to get the above gcc command to work i had to make a symbolic link to usr/include/numpy
# I found the orignal numpy header location by using np.get_include()
# The final terminal command was: 
# sudo ln -s /home/eceftl7/.local/lib/python3.8/site-packages/numpy/core/include/numpy/ /usr/include/numpy