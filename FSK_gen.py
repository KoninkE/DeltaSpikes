# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:45:45 2023

@author: ellio
"""

## https://www.rfwireless-world.com/source-code/Python/FSK-modulation-python-code.html

import numpy as np
#  matplotlib.pyplot as plt
import math
import radio_aid

# def modulateFSK(times, bitData, carrierFreq, symbolRate):
#     sigLen = len(times)
#     samplesPerBit = math.floor(sigLen / len(bitData))
    
#     bitSig = np.array([])
    
#     for i in range(0, len(bitData)):
#         bitSig = np.append(bitSig, np.full(samplesPerBit, bitData[i]))
    
#     sigDif = sigLen - len(bitSig)
#     bitSig = np.append(bitSig, np.ones(sigDif))
    
#     freqs = carrierFreq + (carrierFreq * bitSig / 2)
#     print("Frequencies: ", freqs)
    
#     return np.sin(2 * math.pi * freqs * times)

def mapFSK2(data, carriers):
    bits = ['0', '1']
    mapped = [[bits[i], float(carriers[i])] for i in range(0, len(bits))]
    dictionary = dict(mapped)
    return [dictionary[string] for string in data]

def mapFSK4(data, carriers):
    bits = ['00', '01', '10', '11']
    mapped = [[bits[i], float(carriers[i])] for i in range(0, len(bits))]
    dictionary = dict(mapped)
    return [dictionary[string] for string in data]


def modulateFSK2(bitData, carriers, sampleRate, symbolRate):
    if(len(carriers) != 2):
        raise ValueError("The number of carrier frequencies in carriers must be 2 for 2FSK.")
        
    sampsPerSym = sampleRate * (1 / symbolRate)
    
    shifts = np.array([])
    
    for i in range(0, len(bitData)):
        shifts = np.append(shifts, np.full(int(sampsPerSym) - 1, str(int(bitData[i]))))
        
    freqs = np.array(mapFSK2(shifts, carriers), dtype=np.float64)
    times = np.arange(0, (1 / sampleRate) * len(freqs), 1 / sampleRate)

    if (len(times) > len(freqs)):
        times = times[0:-1 * (len(times) - len(freqs))]

    iSig = np.sin(2 * np.pi * freqs * times)
    qSig = np.zeros(len(iSig))
    
    return iSig, qSig

def modulateFSK4(bitData, carriers, sampleRate, symbolRate):
    if(len(carriers) != 4):
        raise ValueError("The number of carrier frequencies in carriers must be 2 for 2FSK.")
        
    sampsPerSym = sampleRate * (1 / symbolRate)
    
    shifts = np.array([])
    
    for i in range(0, len(bitData)):
        #shifts = np.append(shifts, np.full(int(sampsPerSym) - 1, str(int(bitData[i]))))
        shifts = np.append(shifts, np.full(int(sampsPerSym) - 1, bitData[i]))
        
    freqs = np.array(mapFSK4(shifts, carriers))
    times = np.arange(0, (1 / sampleRate) * len(freqs), 1 / sampleRate)

    if (len(times) > len(freqs)):
        times = times[0:-1 * (len(times) - len(freqs))]

    iSig = np.sin(2 * np.pi * freqs * times)
    qSig = np.zeros(len(iSig))
    
    return iSig, qSig
