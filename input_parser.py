#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
#%%
def input_parser(argv):
    dwinames = argv[1]
    masknames = argv[2]
    featurenumbers = int(argv[3])
    featurenames = []
    for feature_index in range(featurenumbers):
        featurenames.append(argv[4 + feature_index])
        
    testdwinames = argv[4 + featurenumbers]
    testmasknames = argv[5 + featurenumbers]
    
    patch_size_low = int(argv[6 + featurenumbers])
    patch_size_high = int(argv[7 + featurenumbers])
    upsample = int(argv[8 + featurenumbers])
    nDictQ = int(argv[9 + featurenumbers])
    nDictS = int(argv[10 + featurenumbers])
    directory = argv[11 + featurenumbers]
    
    return dwinames, masknames, featurenumbers, featurenames, testdwinames, testmasknames, \
        patch_size_low, patch_size_high, upsample, nDictQ, nDictS, directory

#%%
def input_parser_boot(argv):
    dwinames = argv[1]
    masknames = argv[2]
        
    testdwinames = argv[3]
    testmasknames = argv[4]
    
    patch_size_low = int(argv[5])
    patch_size_high = int(argv[6])
    upsample = int(argv[7])
    nDictQ = int(argv[8])
    nDictS = int(argv[9])
    directory = argv[10]
        
    test = 0
    repetition = 0
    if len(sys.argv) > 11:
        test = int(argv[11])
        repetition = int(argv[12])
    startInd = 0
    endInd = -1
    if len(sys.argv) > 13:
        startInd = int(argv[13])
        endInd = int(argv[14])
    return dwinames, masknames, testdwinames, testmasknames, \
        patch_size_low, patch_size_high, upsample, nDictQ, nDictS, directory, test, repetition, startInd, endInd  
