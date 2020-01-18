#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import os
import nibabel as nib
import numpy as np
from input_parser import input_parser
from data_loader import load_training_matrix, load_test_matrix, data_combine_matrix
from models import mesc_sep_dict
import nibabel.processing

import time
from tqdm import trange
from keras.models import load_model

#%%        
dwinames, masknames, featurenumbers, featurenames, \
testdwinames, testmasknames, patch_size_low, patch_size_high, \
upsample, nDictQ, nDictS, directory = input_parser(sys.argv)

#%%
if os.path.exists(directory) == False:
    os.mkdir(directory)

start = time.time()
print "Loading"    

with open(dwinames) as f:
    allDwiNames = f.readlines()
with open(masknames) as f:
    allMaskNames = f.readlines()
allFeatureNames = []
for feature_index in range(featurenumbers):
    tempFeatureNames = None
    with open(featurenames[feature_index]) as f:
        tempFeatureNames = f.readlines()
    allFeatureNames.append(tempFeatureNames)
allDwiNames = [x.strip('\n') for x in allDwiNames]
allMaskNames = [x.strip('\n') for x in allMaskNames]
for feature_index in range(featurenumbers):
    allFeatureNames[feature_index] = [x.strip('\n') for x in allFeatureNames[feature_index]]
    

#%%
dwiTraining, featurePatchTraining, scales = load_training_matrix(allDwiNames, allMaskNames, allFeatureNames, 
                                                      featurenumbers, patch_size_high, patch_size_low, 
                                                      upsample)
#%%
regressor_model_name = os.path.join(directory,"mesc_sep_dict_regressor.h5")
scales_txt_name = os.path.join(directory,"scales.txt")
np.savetxt(scales_txt_name, scales)

if os.path.exists(regressor_model_name)==False:

    regressor = mesc_sep_dict(dwiTraining.shape[1:], featurePatchTraining.shape[1:], nDictQ, nDictS)
    epoch = 10
    if patch_size_low > 3:
        epoch = 50
    hist = regressor.fit(dwiTraining, featurePatchTraining, batch_size=128, epochs=epoch, verbose=1, validation_split=0.1)
    print(hist.history)
    end = time.time()
    print "Training took ", (end-start)

    regressor.save(regressor_model_name)
else:
    regressor = load_model(regressor_model_name)
#%%###### Test #######
print "Test Phase"    

start = time.time()
with open(testdwinames) as f:
    allTestDwiNames = f.readlines()
with open(testmasknames) as f:
    allTestMaskNames = f.readlines()

allTestDwiNames = [x.strip('\n') for x in allTestDwiNames]
allTestMaskNames = [x.strip('\n') for x in allTestMaskNames]


#for iMask in progressbar.progressbar(range(len(allTestDwiNames))):
for iMask in trange(len(allTestDwiNames)):
    dwi_nii = nib.load(allTestDwiNames[iMask])
    dwi = dwi_nii.get_data()
    mask_nii = nib.load(allTestMaskNames[iMask])
    mask = mask_nii.get_data()
    
    dwiTest, patchCornerList = load_test_matrix(dwi, mask, patch_size_high, patch_size_low, upsample)
                        
    featureList = regressor.predict(dwiTest)
    
    features = data_combine_matrix(featureList, mask.shape, upsample, patch_size_high, patchCornerList, scales)
    
    mask_upsampled_nii = nibabel.processing.resample_to_output(mask_nii, (mask_nii.header.get_zooms()[0]/upsample, mask_nii.header.get_zooms()[1]/upsample, 
                                                                          mask_nii.header.get_zooms()[2]/upsample))
    hdr = dwi_nii.header
    hdr.set_qform(mask_upsampled_nii.header.get_qform()) 
    for feature_index in range(featurenumbers):
        feature_nii = nib.Nifti1Image(features[:,:,:,feature_index], hdr.get_base_affine(), hdr)
        feature_name = os.path.join(directory, "MESC_sep_dict_feature_" + "%02d" % feature_index + "_sub_" + "%02d" % iMask + ".nii.gz")
        feature_nii.to_filename(feature_name)
    
end = time.time()
print "Test took ", (end-start)
    
