import subprocess

# rebuild SuiteSparse
runCommand = "cd SuiteSparse\nmake clean -j 12\nmake library -j 12"
subprocess.call([runCommand], shell=True)

# build ADD
runCommand = 'mkdir build\ncd build\ncmake -DCMAKE_BUILD_TYPE=Release ..\nmake -j 12'
subprocess.call([runCommand], shell=True)
