import os

def get_file_list(src_fol, ext):
    f_list = []
    for subdir, _, files in os.walk(src_fol):
        for filename in files:
            _, f_ext = os.path.splitext(filename)
            if f_ext == ext:
                full_file_path = os.path.join(subdir, filename)
                f_list.append(full_file_path)
    return f_list

def arff_to_data(fpath):
    data,meta = arff.load(fpath)

    return data