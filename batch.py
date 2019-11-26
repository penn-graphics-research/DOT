import subprocess
import os

from os import listdir
from os.path import isfile, join

inputFolderPath = os.path.realpath('.') + '/input/'
progPath = os.path.realpath('.') + '/build/DOT_bin'

envSetStr = 'export LD_LIBRARY_PATH=/usr/local/lib\n'
NTSetStr0 = 'export MKL_NUM_THREADS=' # for Ubuntu or Mac when CHOLMOD is compiled with MKL LAPACK and BLAS
NTSetStr1 = 'export OMP_NUM_THREADS=' # for Ubuntu when CHOLMOD is compiled with libopenblas
NTSetStr2 = 'export VECLIB_MAXIMUM_THREADS=' # for Mac when CHOLMOD is compiled with default LAPACK and BLAS

for numOfThreads in ['1', '8', '12', '32']:
	inputFolderPath = os.path.realpath('.') + '/input/' + numOfThreads + '/'
	onlyfiles = [f for f in listdir(inputFolderPath) if isfile(join(inputFolderPath, f))]
	for inputModelNameI in onlyfiles:
		runCommand = envSetStr + NTSetStr0 + numOfThreads + '\n'
		runCommand += NTSetStr1 + numOfThreads + '\n'
		runCommand += NTSetStr2 + numOfThreads + '\n'
		runCommand += progPath + ' 10 ' + inputFolderPath + inputModelNameI + ' t' + numOfThreads 
		if subprocess.call([runCommand], shell=True):
			continue
