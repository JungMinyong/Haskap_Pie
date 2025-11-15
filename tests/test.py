import os

file_list = os.listdir("src/")

if "Find_Refined.py" not in file_list or "haskap.py" not in file_list or "make_pfs_allsnaps.py" not in file_list:
    print("Required Supporting Modules not present, check installation")

del file_list
