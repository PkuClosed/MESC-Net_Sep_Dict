#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import ThresholdedReLU
from keras.layers import Dense, Input, multiply, add, subtract, Permute, Activation
import sys

sys.setrecursionlimit(10000)

#%%
def mesc_sep_dict(input_shape=[60,3*3*3], output_shape=[3,1*1*1], nDictQ = 300, nDictS = 300):

    nLayers1 = 8
    nLayers2 = 3
    ReLUThres = 0.01
    
    Y = Input(shape=input_shape)
    
    Ws_output = Dense(nDictS, activation='linear', use_bias = True, name = "Ws")(Y)
    Ws_output = Permute((2,1))(Ws_output)
    
    #Ws = Dense(filters=nDictS, activation='relu')
    Wq_output = Dense(nDictQ, activation='linear', use_bias = True, name = "Wq")(Ws_output)
    Wq_output = Permute((2,1))(Wq_output)
    
    Wfs_output = Dense(nDictS, activation='linear', use_bias = True, name = "Wfs")(Y)
    Wfs_output = Permute((2,1))(Wfs_output)
    
    Wfq_output = Dense(nDictQ, activation='linear', use_bias = True, name = "Wfq")(Wfs_output)
    Wfq_output = Permute((2,1))(Wfq_output)
    
    Wis_output = Dense(nDictS, activation='linear', use_bias = True, name = "Wis")(Y)
    Wis_output = Permute((2,1))(Wis_output)
    
    #Ws = Dense(filters=nDictS, activation='relu')
    Wiq_output = Dense(nDictQ, activation='linear', use_bias = True, name = "Wiq")(Wis_output)
    Wiq_output = Permute((2,1))(Wiq_output)
    
    W_fx = Sequential(name = "W_fx")
    W_fx.add(Dense(nDictS, activation='linear', use_bias = True, name = "W_fx_1", input_shape=(nDictQ,nDictS)))
    W_fx.add(Permute((2,1)))
    W_fx.add(Dense(nDictQ, activation='linear', use_bias = True, name = "W_fx_2", input_shape=(nDictS,nDictQ)))
    W_fx.add(Permute((2,1)))
    
    W_ix = Sequential(name = "W_ix")
    W_ix.add(Dense(nDictS, activation='linear', use_bias = True, name = "W_ix_1", input_shape=(nDictQ,nDictS)))
    W_ix.add(Permute((2,1)))
    W_ix.add(Dense(nDictQ, activation='linear', use_bias = True, name = "W_ix_2", input_shape=(nDictS,nDictQ)))
    W_ix.add(Permute((2,1)))
    
    S = Sequential(name = "S")
    S.add(Dense(nDictS, activation='linear', use_bias = True, name = "S_1", input_shape=(nDictQ,nDictS)))
    S.add(Permute((2,1)))
    S.add(Dense(nDictQ, activation='linear', use_bias = True, name = "S_2", input_shape=(nDictS,nDictQ)))
    S.add(Permute((2,1)))
    ## initialize \tilde{C}, x, and C for t = 1
    Ctilde = Wq_output 

    I = Activation('sigmoid')(Wiq_output) # x^{0} = 0

    C = multiply([I, Ctilde]) # c^{0} = 0

    X = ThresholdedReLU(theta = ReLUThres)(C)

    for l in range(nLayers1-1):
        S_output = S(X)
        Diff_X = subtract([X,S_output]) #temp variable
        
        Ctilde = add([Wq_output, Diff_X])
        
        Wfx_Wfy = add([W_fx(X), Wfq_output])
        F = Activation('sigmoid')(Wfx_Wfy) 
        Wix_Wiy = add([W_ix(X), Wiq_output])
        I = Activation('sigmoid')(Wix_Wiy) 
        Cf = multiply([F, C])
        Ci = multiply([I, Ctilde])
        C = add([Cf, Ci])
        
        X = ThresholdedReLU(theta = ReLUThres)(C)
    
    nChannelsQ = 75
    nChannelsS = 75
    H = Sequential(name = "H")
    H.add(Dense(nChannelsS, activation='relu', name = "Hs" + str(0), input_shape=(nDictQ,nDictS)))
    H.add(Permute((2,1)))
    H.add(Dense(nChannelsQ, activation='relu', name = "Hq" + str(0)))
    H.add(Permute((2,1)))
    for i in range(nLayers2-1):
        H.add(Dense(nChannelsS, activation='relu', name = "Hs" + str(i+1)))
        H.add(Permute((2,1)))
        H.add(Dense(nChannelsQ, activation='relu', name = "Hq" + str(i+1)))
        H.add(Permute((2,1)))
    H.add(Dense(output_shape[1], activation='relu', name = "Hs" + str(nLayers2)))
    H.add(Permute((2,1)))
    H.add(Dense(output_shape[0], activation='relu', name = "Hq" + str(nLayers2)))
    H.add(Permute((2,1)))

    outputs = H(X)
    
    ### setting the model ###                    
    regressor = Model(inputs=Y,outputs=outputs)
    regressor.compile(optimizer=Adam(lr=0.0001), loss='mse')
    print regressor.summary()
    return regressor

#%%
def dict_computation(regressor_model, input_shape=[60,3*3*3], nDictQ = 300, nDictS = 300):

    # repeat the network: set untrainable weights 
    nLayers1 = 8
#    nLayers2 = 3
    ReLUThres = 0.01
    
    Y = Input(shape=input_shape)
    
    Ws_output = Dense(nDictS, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("Ws").get_weights(), name = "Ws")(Y)
    Ws_output = Permute((2,1))(Ws_output)
    
    #Ws = Dense(filters=nDictS, activation='relu')
    Wq_output = Dense(nDictQ, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("Wq").get_weights(), name = "Wq")(Ws_output)
    Wq_output = Permute((2,1))(Wq_output)
    
    Wfs_output = Dense(nDictS, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("Wfs").get_weights(), name = "Wfs")(Y)
    Wfs_output = Permute((2,1))(Wfs_output)
    
    Wfq_output = Dense(nDictQ, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("Wfq").get_weights(), name = "Wfq")(Wfs_output)
    Wfq_output = Permute((2,1))(Wfq_output)
    
    Wis_output = Dense(nDictS, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("Wis").get_weights(), name = "Wis")(Y)
    Wis_output = Permute((2,1))(Wis_output)
    
    #Ws = Dense(filters=nDictS, activation='relu')
    Wiq_output = Dense(nDictQ, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("Wiq").get_weights(), name = "Wiq")(Wis_output)
    Wiq_output = Permute((2,1))(Wiq_output)
    
    W_fx = Sequential(name = "W_fx")
    W_fx.add(Dense(nDictS, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("W_fx").get_layer("W_fx_1").get_weights(), name = "W_fx_1", input_shape=(nDictQ,nDictS)))
    W_fx.add(Permute((2,1)))
    W_fx.add(Dense(nDictQ, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("W_fx").get_layer("W_fx_2").get_weights(), name = "W_fx_2", input_shape=(nDictS,nDictQ)))
    W_fx.add(Permute((2,1)))
    
    W_ix = Sequential(name = "W_ix")
    W_ix.add(Dense(nDictS, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("W_ix").get_layer("W_ix_1").get_weights(), name = "W_ix_1", input_shape=(nDictQ,nDictS)))
    W_ix.add(Permute((2,1)))
    W_ix.add(Dense(nDictQ, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("W_ix").get_layer("W_ix_2").get_weights(), name = "W_ix_2", input_shape=(nDictS,nDictQ)))
    W_ix.add(Permute((2,1)))
    
    S = Sequential(name = "S")
    S.add(Dense(nDictS, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("S").get_layer("S_1").get_weights(), name = "S_1", input_shape=(nDictQ,nDictS)))
    S.add(Permute((2,1)))
    S.add(Dense(nDictQ, activation='linear', use_bias = True, trainable = False, 
                      weights = regressor_model.get_layer("S").get_layer("S_2").get_weights(), name = "S_2", input_shape=(nDictS,nDictQ)))
    S.add(Permute((2,1)))
    ## initialize \tilde{C}, x, and C for t = 1
    Ctilde = Wq_output 

    I = Activation('sigmoid')(Wiq_output) # x^{0} = 0

    C = multiply([I, Ctilde]) # c^{0} = 0

    X = ThresholdedReLU(theta = ReLUThres)(C)

    for l in range(nLayers1-1):
        S_output = S(X)
        Diff_X = subtract([X,S_output]) #temp variable
        
        Ctilde = add([Wq_output, Diff_X])
        
        Wfx_Wfy = add([W_fx(X), Wfq_output])
        F = Activation('sigmoid')(Wfx_Wfy) 
        Wix_Wiy = add([W_ix(X), Wiq_output])
        I = Activation('sigmoid')(Wix_Wiy) 
        Cf = multiply([F, C])
        Ci = multiply([I, Ctilde])
        C = add([Cf, Ci])
        
        X = ThresholdedReLU(theta = ReLUThres)(C)
    
    Phi = Sequential()
    Phi.add(Dense(input_shape[1], activation='linear', use_bias = False, input_shape=(nDictQ,nDictS)))
    Phi.add(Permute((2,1)))
    Phi.add(Dense(input_shape[0], activation='linear'))
    Phi.add(Permute((2,1)))
    
    Y_rec = Phi(X)

    ### setting the model ###                    
    residual = Model(inputs=Y, outputs=Y_rec)
    residual.compile(optimizer=Adam(lr=0.0001), loss='mse')
    print residual.summary()
    return residual