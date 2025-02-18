import os
import numpy as np
import numpy.ma as ma
from PIL import Image, ImageChops
from tiffslide import TiffSlide
import skimage
from skimage import data, io
import matplotlib
import torchstain
import csv
from matplotlib import pyplot as plt
from matplotlib import colors
import math
import skimage.measure
from skimage.color import rgb2lab
import pandas as pd
from tqdm.notebook import tqdm
import patchify
import multiprocessing
import shutil
import xarray as xr
from xarray import open_dataset
from tiffslide_xarray import open_all_levels
import tiffslide_xarray.accessors
import traceback
from scipy import stats
import re
import random


# Greyscale & Chan-Vese mask
def getMask(smallSlide, dt=.5, debug = False):
    smallImage = skimage.color.rgb2gray(smallSlide.image)
    if debug:
        print("RGB to grey done")
    mask = skimage.segmentation.chan_vese(smallImage, dt=dt)
    if debug:
        print("chanvese done")
    return mask


# Xarray helper functions
def getFullPatchFromPreview(fullSlide, previewPatch):
    patch = previewPatch.load()
    x1 = round(patch.coords['x'].data[0])
    x2 = round(patch.coords['x'].data[-1])
    y1 = round(patch.coords['y'].data[0])
    y2 = round(patch.coords['y'].data[-1])
    fullPatch = fullSlide.sel(x=slice(x1,x2),y=slice(y1,y2))
    return fullPatch

def getPatchAt(s, x1, x2, y1, y2):
    return s.sel(x=slice(x1,x2),y=slice(y1,y2))

def getClosestMaskCoord(preview, x=-1, y=-1):
    xCoords = preview.coords['x'].values
    yCoords = preview.coords['y'].values
    if x >= 0:
        xIndex = (np.abs(x - xCoords)).argmin()
        return xIndex
    if y >= 0:
        yIndex = (np.abs(y - yCoords)).argmin()
        return yIndex


# Fetch patches to include
def getFocusedPatches(slide, preview, mask,loadInMemory = False, inclusionThreshold = 0.8):
    regions = []
    regionLocation = []
    segment_length = 512
    # Why is the patch size 513 instead of 512? (Rounding issue)
    focusThreshold = 0.1
    b = False
    for y in range(0,slide.sizes['y'],segment_length):
        for x in range(0,slide.sizes['x'], segment_length):
            # print(f"y: {y} out of {slide.sizes['y']}")
            # print(f"x: {x} out of {slide.sizes['x']}")
            yPatchEnd = round(y+segment_length)
            xPatchEnd = round(x+segment_length)
            maskX = round(getClosestMaskCoord(preview,x=x))
            maskY = round(getClosestMaskCoord(preview,y=y))
            maskXPatchEnd = round(getClosestMaskCoord(preview,x=xPatchEnd))
            maskYPatchEnd = round(getClosestMaskCoord(preview,y=yPatchEnd))
            # maskXPatchEnd = maskX+segment_length
            # maskYPatchEnd = maskY+segment_length
            # print(f"x:{x} - {xPatchEnd} ({maskX}-{maskXPatchEnd}), y:{y} - {yPatchEnd} ({maskY}-{maskYPatchEnd})")
            region = mask[max(0,maskY-1):min(maskYPatchEnd+1,mask.shape[0]), max(0,maskX-1):min(maskXPatchEnd+1,mask.shape[1])]
            mean = np.mean(region)
            if mean >= focusThreshold and mean < 1.0 and mean > inclusionThreshold:
                # print(f"Mean is {mean}")
                patch = slide.sel(x=slice(x,xPatchEnd-1),y=slice(y,yPatchEnd-1))
                regionLocation.append(f"X:{x}-{xPatchEnd} ({maskX}-{maskXPatchEnd}), Y:{y}-{yPatchEnd} ({maskY}-{maskYPatchEnd})")
                if loadInMemory:
                    regions.append(patch.load())
                else:
                    regions.append(patch)
    return regions, regionLocation


# Calculate blur
def calcBlur(image):
    bwimage = skimage.color.rgb2gray(image) 
    bwimage = np.interp(bwimage, (bwimage.min(), bwimage.max()), (0, 255)).astype('uint8')
    laplaceImage = skimage.filters.laplace(bwimage)
    return np.var(laplaceImage)

# Convert optical density to RGB.
# Inverse of the function used in https://github.com/EIDOSLAB/torchstain/blob/main/torchstain/numpy/normalizers/macenko.py
def ODtoRGB(OD, Io=240):
    assert OD.min() >= 0, "Negative optical density."
    OD = np.maximum(OD, 1e-6)
    return (Io*np.exp(-1*OD)-1).astype(np.uint8)

# Calculate avg brightness by converting to Lab space
def avgBrightness(image):
    lab_image = rgb2lab(image)
    L_channel = lab_image[:, :, 0]
    average_brightness = np.mean(L_channel)
    return average_brightness

def RGBStringtoList(rgbStr):
    rgbs = rgbStr.replace('[','').replace(']','').split(' ')
    lst = [int(rgb) for rgb in rgbs if rgb != '']
    return lst

def addToAvgCSV(slide_name, blur, hematoxylin, eosin, brightness, source, results_csv='results.csv'):
    with open(results_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([slide_name, blur, hematoxylin, eosin, brightness, source])

def validRGBIndices(df, columnNames = ['Hematoxylin RGB', 'Eosin RGB']):
    validity = []
    for i in range(len(df)):
        valid = True
        for col in columnNames:
            rgbs = df.iloc[i][col].replace('[','').replace(']','').split(' ')
            for rgb in rgbs:
                if rgb != '' and int(rgb) <= 0:
                    valid = False
                    break
            if not valid:
                break
        validity.append(valid)
    return validity

def forceCreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)




def readProcess(process_file="process.txt"):
    if os.path.exists(process_file):
        # Open the file in read mode
        with open(process_file, 'r') as file:
            # Read the contents of the file
            content = file.read()
            content = int(content)
            return content
    print("process.txt doesn't exist")
    return ""
    
def writeToProcess(increment, process_file="process.txt"):
    process = 0
    if os.path.exists(process_file):
        # Open the file in read mode
        with open(process_file, 'r') as file:
            # Read the contents of the file
            content = file.read()
            if content != "":
                process = int(content)
                process += increment
        with open(process_file, 'w') as file:
            if content != "":
                file.write(str(process))
    else:
        with open(process_file, 'w') as file:
            # Read the contents of the file
            file.write("0")


def clearProcess(process_file="process.txt"):
    with open(process_file, 'w') as file:
            file.write("")