# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:40:36 2023

@author: ellio
"""

import math
import numpy as np
import csv

# reads an array from a file
def readBitArray(path):
    myArray = []
    
    with open(path, "r") as fid:
        for line in fid.readlines():
            myLine = []
            for char in line:
                if(char != '' or char != '\n'):
                    myLine.append(char)
                    
            myArray.append(myLine[0:-1])
    
    return myArray

def readWordArray(path):
    myArray = []
    
    with open(path, "r") as fid:
        for line in fid.readlines():
            myWord = ''
            for char in line:
                if(char != '' or char != '\n'):
                    myWord = myWord + char
                    
            myArray.append(myWord[0:-1])
    
    return myArray

# converts all the data in an array to float
def convToFloat(array, dims):
    newArray = []
    
    if(dims > 2):
        raise ValueError("Dims must be less than or equal to 2.")
    
    if(dims == 1):
        for element in array:
            newArray.append(float(element))
        
    elif(dims == 2):
        for vector in array:
            vect = []
            
            for element in vector:
                vect.append(float(element))
                
            newArray.append(vect)
            
    return newArray

# generates a .wav file using a generated signal
def writeToWav(path, iSig, qSig, sampleRate):
    with open(path, 'w') as fid:
        w = csv.writer(fid, delimiter='\t', lineterminator='\n')
        w.writerows(zip(iSig, qSig))
    
          
# generates a normal sinusoidal carrier wave
def generateCarrier(carrierFreq, times):
    carrierWave = np.sin(2 * math.pi * carrierFreq * times)
    return carrierWave

# def groupBits(data, bitsPerGroup):
#     newArray = []
    
#     for vector in data:
#         if(len(vector) % bitsPerGroup):
#             raise ValueError("The length of the bit streams must be divisible by the bits per group variable.")
        
#         numBits = 0
#         bitString = ''
#         word = []
        
#         for bit in vector:
#             numBits += 1
            
#             if(numBits % bitsPerGroup == 1):
#                 bitString = str(int(bit))
                
#             else:
#                 bitString = bitString + str(int(bit))
                
#             if(numBits % bitsPerGroup == 0):
#                 word.append(bitString)
                
#         newArray.append(word)
        
#     return newArray

def groupBits(data, bitsPerGroup):
    numBits = 0
    bitString = ''
    word = []
    
    for bit in data:
        numBits += 1
        
        if(bitsPerGroup == 1):
            word.append(str(int(bit)))
            continue
        
        if(numBits % bitsPerGroup == 1):
            bitString = str(int(bit))
            
        else:
            bitString = bitString + str(int(bit))
            
        if(numBits % bitsPerGroup == 0):
            word.append(bitString)
        
    return word