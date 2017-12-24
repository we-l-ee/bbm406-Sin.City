# Import the os module, for the os.walk function
import os
import shutil
import sys

folder = sys.argv[1]

root_dir = 'dataset\\' + folder

for dir_name, subdir_list, file_list in os.walk(root_dir):
    for f_name in file_list:
        file_name = os.path.join(dir_name, f_name)
        shutil.move(file_name, "dataset\\new\\"+folder)
        print(file_name)


i = 0
for dir_name, subdir_list, file_list in os.walk(root_dir):
    for f_name in file_list:
        f_type = f_name.split('.')[-1]
        file_name = os.path.join(dir_name, f_name)
        dest_name = os.path.join(dir_name, folder[0] + '_' + str(i) + '.' + f_type)
        os.rename(file_name, dest_name)
        i += 1

