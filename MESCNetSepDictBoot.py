#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import os
import nibabel as nib
import numpy as np
from input_parser import input_parser_boot
from data_loader import load_training_matrix_boot, load_test_matrix, data_combine_matrix
from models import dict_computation
import nibabel.processing
from keras.models import load_model

import time
from tqdm import trange

#%%        
dwinames, masknames, testdwinames, testmasknames, patch_size_low, patch_size_high, \
upsample, nDictQ, nDictS, directory, test, repetition, startInd, endInd = input_parser_boot(sys.argv)

print "Test:", bool(test)

#%%
if os.path.exists(directory) == False:
    os.mkdir(directory)

start = time.time()
print "Loading"    

with open(dwinames) as f:
    allDwiNames = f.readlines()
with open(masknames) as f:
    allMaskNames = f.readlines()
allDwiNames = [x.strip('\n') for x in allDwiNames]
allMaskNames = [x.strip('\n') for x in allMaskNames]
    

#%%

regressor_model_name = os.path.join(directory,"mesc_sep_dict_regressor.h5")
regressor = load_model(regressor_model_name)

residual_model_name = os.path.join(directory,"mesc_sep_dict_residual_computataion.h5")
scales_txt_name = os.path.join(directory,"scales.txt")

if os.path.exists(residual_model_name)==False:
    dwiTraining = load_training_matrix_boot(allDwiNames, allMaskNames, patch_size_high, patch_size_low, 
                                                      upsample)
    residual_computation = dict_computation(regressor, dwiTraining.shape[1:], nDictQ, nDictS)
    epoch = 10
    hist = residual_computation.fit(dwiTraining, dwiTraining, batch_size=128, epochs=epoch, verbose=1, validation_split=0.1)
    print(hist.history)
    residual_computation.save(residual_model_name)
else:
    residual_computation = load_model(residual_model_name)
    
end = time.time()
print "Training took ", (end-start)

#%%###### Test #######
if test > 0:
    print "Test Phase"    
    
    start = time.time()
    with open(testdwinames) as f:
        allTestDwiNames = f.readlines()
    with open(testmasknames) as f:
        allTestMaskNames = f.readlines()
    
    allTestDwiNames = [x.strip('\n') for x in allTestDwiNames]
    allTestMaskNames = [x.strip('\n') for x in allTestMaskNames]
    
    scales = np.loadtxt(scales_txt_name)
    
    if endInd == -1:
        endInd = len(allTestDwiNames)
    print "Scales", scales
    print "From", startInd, "to", endInd

    for iMask in trange(startInd, endInd, desc='subject'):
        dwi_nii = nib.load(allTestDwiNames[iMask])
        dwi = dwi_nii.get_data()
        mask_nii = nib.load(allTestMaskNames[iMask])
        mask = mask_nii.get_data()
        
        mask_upsampled_nii = nibabel.processing.resample_to_output(mask_nii, (mask_nii.header.get_zooms()[0]/upsample, mask_nii.header.get_zooms()[1]/upsample, 
                                                                              mask_nii.header.get_zooms()[2]/upsample))
        hdr = dwi_nii.header
        hdr.set_qform(mask_upsampled_nii.header.get_qform()) 
        
        # Load data
        dwiTest, patchCornerList = load_test_matrix(dwi, mask, patch_size_high, patch_size_low, upsample)
                           
        # Compute microstructure and residuals
        YRecList = residual_computation.predict(dwiTest)
        residual = dwiTest - YRecList
        residual = residual - np.mean(residual, axis=(1,2), keepdims=True)
        
        # Save residuals
        residuals_npy_name = os.path.join(directory,"residuals" + "_sub_" + "%02d" % iMask)
        np.save(residuals_npy_name, residual)
        
        # Bootstrap
        featureBootList = None
        for rep in range(1):
            YBoot = np.zeros(dwiTest.shape)
            for sample in range(residual.shape[0]):
                residual_flatten = residual[sample].flatten()
                sampled_residual = np.random.choice(residual_flatten, residual.shape[1]*residual.shape[2])
                YBoot[sample] = YRecList[sample] + np.reshape(sampled_residual, (residual.shape[1],residual.shape[2]))
            featureBootList = regressor.predict(YBoot)
            
        featuresBootAll = np.zeros([repetition, featureBootList.shape[0], featureBootList.shape[1], featureBootList.shape[2]])
        featuresBootAll[0] = featureBootList
        
        for rep in trange(1, repetition, desc='bootstrap'):
            YBoot = np.zeros(dwiTest.shape)
            for sample in range(residual.shape[0]):
                residual_flatten = residual[sample].flatten()
                sampled_residual = np.random.choice(residual_flatten, residual.shape[1]*residual.shape[2])
                YBoot[sample] = YRecList[sample] + np.reshape(sampled_residual, (residual.shape[1],residual.shape[2]))
                
            featureBootList = regressor.predict(YBoot)
            featuresBootAll[rep] = featureBootList
            
        featureBootMean = np.mean(featuresBootAll, axis = 0)
        featureBootStd = np.std(featuresBootAll, axis = 0)
        featuresBootMeanImage = data_combine_matrix(featureBootMean, mask.shape, upsample, patch_size_high, patchCornerList, scales)
        featuresBootStdImage = data_combine_matrix(featureBootStd, mask.shape, upsample, patch_size_high, patchCornerList, scales)
            
        for feature_index in range(featuresBootMeanImage.shape[-1]):
            feature_boot_mean_nii = nib.Nifti1Image(featuresBootMeanImage[:,:,:,feature_index], hdr.get_base_affine(), hdr)
            feature_boot_mean_name = os.path.join(directory, "MESC_sep_dict_reg_feature_boot_mean_" + "%02d" % feature_index + "_sub_" + "%02d" % iMask + ".nii.gz")
            feature_boot_mean_nii.to_filename(feature_boot_mean_name)
            
            feature_boot_std_nii = nib.Nifti1Image(featuresBootStdImage[:,:,:,feature_index], hdr.get_base_affine(), hdr)
            feature_boot_std_name = os.path.join(directory, "MESC_sep_dict_reg_feature_boot_std_" + "%02d" % feature_index + "_sub_" + "%02d" % iMask + ".nii.gz")
            feature_boot_std_nii.to_filename(feature_boot_std_name)
            
    end = time.time()
    print "Test took ", (end-start)
    
