import subprocess
import os
def openSmileCall(wavFile,outFile):

    OpenSmile = '/home/helium-balloons/Desktop/midas/opensmile-2.3.0/bin/linux_x64_standalone_libstdc6/SMILExtract'
    configAddr = '/home/helium-balloons/Desktop/midas/opensmile-2.3.0/config/IS13_ComParE.conf'
    os.system(OpenSmile+ ' -C '+ configAddr+' '+ ' -I' +' '+wavFile +' -O'+' '+outFile)
