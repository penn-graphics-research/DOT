import subprocess

# install xorg-dev libglu1-mesa
runCommand = "sudo apt-get install xorg-dev libglu1-mesa-dev"
subprocess.call([runCommand], shell=True)

# rebuild SuiteSparse
runCommand = "cd SuiteSparse\nmake clean -j 12\nmake library BLAS='-lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -lmkl_blas95_lp64 -liomp5' LAPACK='-lmkl_lapack95_lp64' -j 12"
subprocess.call([runCommand], shell=True)

# build ADD
runCommand = 'mkdir build\ncd build\ncmake -DCMAKE_BUILD_TYPE=Release ..\nmake -j 12'
subprocess.call([runCommand], shell=True)
