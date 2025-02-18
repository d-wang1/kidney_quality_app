import os
import re
import csv
import random



WUSTL_DIR = "/mnt/RIS/group_root/data/pathology/handoff/trusted_kidney_large_study_2022/raw_data/whole_slide_images"
KPMP_DIR = "/mnt/RIS/group_root/data/pathology/david_slide_data/kpmp_data"
CANCER_DIR = "/mnt/RIS/group_root/data/pathology/david_slide_data/cancer_data"

def get_wustl_dir():
    return WUSTL_DIR

def get_kpmp_dir():
    return KPMP_DIR

def get_cancer_dir():
    return CANCER_DIR

def get_kpmp_files():
    kpmp_files = os.listdir(KPMP_DIR)
    return kpmp_files

def get_cancer_files():
    cancer_subdirs = os.listdir(CANCER_DIR)
    cancer_subdirs = [f for f in cancer_subdirs if ('.txt' not in f)]


    cancer_files = []
    for dir in cancer_subdirs:
        d = os.path.join(CANCER_DIR,dir)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if ".svs" in f and ".partial" not in f:
                    cancer_files.append(os.path.join(d,f))
                    break

        else:
            print(f"{d} is not a cancer atlas directory")
            break
    return cancer_files


def get_wustl_files():
    wustl_files = []
    with open('/raw_data/kidney_image_filepaths.txt','r') as f:
        wustl_files = [os.path.join(line.replace("/mnt/RIS/group_root/data/pathology/handoff/trusted_kidney_large_study_2022/raw_data/whole_slide_images/","").replace("\n","")) for line in f]


    gl_pattern = r"GL\d{2}-"
    gl_matching_strings = [s for s in wustl_files if re.search(gl_pattern, s)]
    gl_files = random.sample(gl_matching_strings, min(120, len(gl_matching_strings)))

    mts_pattern = r"MTS\d{2}-"
    mts_matching_strings = [s for s in wustl_files if re.search(mts_pattern, s)]
    mts_files = random.sample(mts_matching_strings, min(120, len(mts_matching_strings)))

    return gl_files, mts_files


def load_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = [row[0] for row in reader]
    return data