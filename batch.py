import subprocess
import os

from os import listdir
from os.path import isfile, join

inputFolderPath = os.path.realpath('.') + '/input/'
progPath = os.path.realpath('.') + '/build/OptCuts_dynamic_bin'

priority = 'nice -n -10 '

onlyfiles = [f for f in listdir(inputFolderPath) if isfile(join(inputFolderPath, f))]
for inputModelNameI in onlyfiles:
	runCommand = priority + progPath + ' 10 ' + inputFolderPath + inputModelNameI + ' 0.999 666 4 test'
	if subprocess.call([runCommand], shell=True):
		continue
