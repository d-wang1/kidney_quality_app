from methods import getMask, getFocusedPatches, calcBlur, ODtoRGB, avgBrightness, addToAvgCSV, RGBStringtoList, validRGBIndices, forceCreateDir, readProcess, writeToProcess

import numpy as np
from tqdm import tqdm
import torchstain
from xarray import open_dataset
import pandas as pd
import traceback
import os

def blurFromPatches(maskedPatches):
    # print(f"Calculating blur for {len(maskedPatches)} patches")
    blurCoeffs = []
    for i in tqdm(range(len(maskedPatches)), desc="Calculating Blur Coefficients"):
        blurCoeffs.append(calcBlur(np.interp(np.array(maskedPatches[i].image), (np.array(maskedPatches[i].image).min(), np.array(maskedPatches[i].image).max()), (0, 255)).astype('uint8')))
    return blurCoeffs

def HEValuesFromPatches(maskedPatches):
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    HEvalues = np.empty((len(maskedPatches),2,3))
    errorElements = 0
    for i in tqdm(range(len(maskedPatches)), desc="Calculating HE colors of patches"):
        img = maskedPatches[i].image
        try:
            normalizer.fit(np.asarray(img))
        except Exception:
            # print(f"Eigenvalues did not converge for patch {i}")
            errorElements += 1
            continue
        HE = normalizer.HERef
        try:
            HE_RGB = ODtoRGB(HE)
        except:
            # print("Error in converting OD to RGB")
            errorElements += 1
            continue
        H = HE_RGB[:,0]
        E = HE_RGB[:,1]
        if np.isnan(H).any() or np.isnan(E).any():
            errorElements += 1
            continue
        HEvalues[i][0] = H.astype(int)
        HEvalues[i][1] = E.astype(int)
    if errorElements > 0:
        print(f"WARNING: {errorElements}/{len(maskedPatches)} patches encountered errors during staining color calculations")
    print("HE values:")
    print(HEvalues[0,:])
    print("HE values dtype: ", HEvalues.dtype)
    return HEvalues.astype(int)

def brightnessFromPatches(maskedPatches):
    brightnesses = []
    for i in tqdm(range(len(maskedPatches)), desc='Calculating Brightnesses of Patches'):
        brightnesses.append(avgBrightness(maskedPatches[i].image))
    return brightnesses

def processBrightness(f, dir, previewDownsample=10):
    slide = open_dataset(os.path.join(dir,f))
    slide = slide.wsi.set_mpp()
    preview = open_dataset(os.path.join(dir,f), level=-1)
    preview = preview.wsi.set_mpp()
    smallerPreview = preview.sel(x=slice(1,slide.sizes['x'],previewDownsample),y=slice(1,slide.sizes['y'],previewDownsample))
    # print(f," slide loaded")
    # print(f"Slide dims: {slide.dims}, preview dims: {preview.dims}")
    mask = getMask(smallerPreview).astype(int)
    # print("Found mask")
    # previewPath = os.path.join(RESULT_DIR, f.replace('.svs',"").replace("/","="),"slidePreview.png")
    # preview.image.wsi.pil.save(previewPath)
    focusedPatches, focusedLocations = getFocusedPatches(slide,smallerPreview,mask)
    brightnesses = brightnessFromPatches(focusedPatches)
    return np.mean(brightnesses)

def processSaveFile(f,dir, src, result_path="results", previewDownsample=10):
    slide = open_dataset(os.path.join(dir,f))
    slide = slide.wsi.set_mpp()
    preview = open_dataset(os.path.join(dir,f), level=-1)
    preview = preview.wsi.set_mpp()
    smallerPreview = preview.sel(x=slice(1,slide.sizes['x'],previewDownsample),y=slice(1,slide.sizes['y'],previewDownsample))
    # print(f," slide loaded")
    # print(f"Slide dims: {slide.dims}, preview dims: {preview.dims}")
    mask = getMask(smallerPreview).astype(int)
    # print("Found mask")
    # previewPath = os.path.join(result_path, f.replace('.svs',"").replace("/","="),"slidePreview.png")
    # preview.image.wsi.pil.save(previewPath)
    focusedPatches, focusedLocations = getFocusedPatches(slide,smallerPreview,mask)
    # print("Found focused patches")
    blurCoeffs = blurFromPatches(focusedPatches)
    # print("Found blur, finding HE")
    try:
        HEvalues = HEValuesFromPatches(focusedPatches)
    except Exception:
        print("Error while calculating HE")
        print(traceback.format_exc())
    # print("Found HE")
    if HEvalues is None:
        return None
    # print(blurCoeffs)
    # print(HEvalues)
    
    hematoxylinColors = HEvalues[:,0,:]
    eosinColors = HEvalues[:,1,:]
    hematoxylinColorsStr = ["".join(np.array2string(h)) for h in hematoxylinColors]
    eosinColorsStr = ["".join(np.array2string(e)) for e in eosinColors]
    df = pd.DataFrame({"Patch Location":focusedLocations,"Blur Coefficient":blurCoeffs,"Hematoxylin RGB":hematoxylinColorsStr, "Eosin RGB":eosinColorsStr})
    df = df[df['Blur Coefficient'] > 0.0001]
    df = df[validRGBIndices(df)]
    # dfPath = os.path.join(result_path, f.replace('.svs',"").replace("/","="),"df.csv")
    # if not os.path.isdir(os.path.join(result_path, f.replace('.svs',"").replace("/","="))):
    #     # os.mkdir(os.path.join(RESULT_DIR, f.replace("/","-")[-1].replace('.svs',"")))
    #     print(os.path.join(result_path, f.replace('.svs',"").replace("/","=")), " does not exist. Making new directory")
    # df.to_csv(dfPath)
    avgBlur = np.mean(df['Blur Coefficient'])
    hColors = np.array([RGBStringtoList(x) for x in df['Hematoxylin RGB']])
    avgHColors = np.mean(hColors,axis=0)
    eColors = np.array([RGBStringtoList(x) for x in df['Eosin RGB']])
    avgEColors = np.mean(eColors,axis=0)
    # print(f"Blur: {avgBlur}")
    # print(f"Hematoxylin: {avgHColors}")
    # print(avgHColors.shape)
    # print(f"Eosin: {avgEColors}")
    # print(avgEColors.shape)
    avgBrightness = processBrightness(f, dir, previewDownsample=previewDownsample)
    print(f"Blur: {avgBlur}, Hematoxylin: {avgHColors}, Eosin: {avgEColors}, Brightness: {avgBrightness}")
    try:
        addToAvgCSV(str(f), avgBlur, avgHColors, avgEColors, avgBrightness, src)
    except Exception:
        print(traceback.format_exc())
    return df

def processAndSave(f, dir, src, result_path="results"):
    result_path = os.path.join(result_path, f.replace('.svs',"").replace("/","="))
    forceCreateDir(result_path)
    result = processSaveFile(f,dir, src, result_path = result_path)
    return result


def run_all_files(files, dir, src, processNFiles=-1, started=True):
    counter = 1
    processingFile = 0
    for f in files:
        lastRecord = int(readProcess())
        if processNFiles > 0 and counter > processNFiles:
            break
        if processingFile < lastRecord and not started:
            print(f"Skipping {f}")
            processingFile += 1
            counter += 1
            continue
        else:
            print(f"Processing {f} (#{counter})")
            # if processNWustlFiles > 0:
            #     print(f"Processing {f} (#{counter})")
            # else:
            #     print(f"Processing {f}")
            started = True
            try: 
                df = processAndSave(f, dir, src)
            except Exception as error:
                print(f"Error in processing {f}. Skipping.", error)
            writeToProcess(1)
            counter += 1