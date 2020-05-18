
from io_util import get_file_list,arff_to_data
from openSmileUtil import openSmileCall

flist = get_file_list('./audio_new/','.wav')
for i in range(len(flist)):
    file_path = flist[i] + '.csv'
    openSmileCall(flist[i], file_path)