# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:54:43 2023

@author: ellio
"""

## https://www.rfwireless-world.com/source-code/Python/BPSK-QPSK-16QAM-64QAM-modulation-python-code.html

import numpy as np
# import matplotlib.pyplot as plt
import math
from cmath import phase
import radio_aid

# maps bits to symbols depending on what modulation is choosen
def mapSymbols(grouped_bits, mod="16QAM"):
    if(mod == "OOK"):
        ook_map = {'0' : 0,
                    '1' : 1}
        
        # return [[ook_map[string] for string in word] for word in grouped_bits]
        return [ook_map[string] for string in grouped_bits]
        
    elif(mod == "2ASK"):
        ask2_map = {'0' : 0.5,
                    '1' : 1}
        
        # return [[ask2_map[string] for string in word] for word in grouped_bits]
        return [ask2_map[string] for string in grouped_bits]
    
    elif(mod == "4ASK"):
        ask4_map = {'00' : 0.25,
                    '01' : 0.5,
                    '10' : 0.75,
                    '11' : 1}
        
        # return [[ask4_map[string] for string in word] for word in grouped_bits]
        return [ask4_map[string] for string in grouped_bits]
        
    elif(mod == "BPSK"): # good
        bpsk_map = {'0' : -1,
                    '1' : 1}
        
        # return [[bpsk_map[string] for string in word] for word in grouped_bits]
        return [bpsk_map[string] for string in grouped_bits]
        
    if(mod == "QPSK"): # good
        qpsk_map = {'00' : (math.sqrt(2)/2) + (math.sqrt(2)/2) * 1j,
                      '01' : -1 * (math.sqrt(2)/2) + (math.sqrt(2)/2) * 1j,
                      '10' : -1 * (math.sqrt(2)/2) - (math.sqrt(2)/2) * 1j,
                      '11' : (math.sqrt(2)/2) - (math.sqrt(2)/2) * 1j}
        
        # return [[qpsk_map[string] for string in word] for word in grouped_bits]
        return [qpsk_map[string] for string in grouped_bits]
    
    elif(mod == "16PSK"): # good
        psk16_map = {'0000' : 1 + 0j,
                      '0001' : (math.sqrt(3)/2) - 0.5j,
                      '0010' : 0.5 - (math.sqrt(3)/2) * 1j,
                      '0011' : (math.sqrt(2)/2) - (math.sqrt(2)/2) * 1j,
                      '0100' : -1 * (math.sqrt(3)/2) - 0.5j,
                      '0101' : -1 * (math.sqrt(2)/2) - (math.sqrt(2)/2) * 1j,
                      '0110' : 0 - 1j,
                      '0111' : -0.5 - (math.sqrt(3)/2) * 1j, 
                      '1000' : (math.sqrt(3)/2) + 0.5j,
                      '1001' : (math.sqrt(2)/2) + (math.sqrt(2)/2) * 1j,
                      '1010' : 0 + 1j,
                      '1011' : 0.5 + (math.sqrt(3)/2) * 1j,
                      '1100' : -1 - 0j,
                      '1101' : -1 * (math.sqrt(3)/2) + 0.5j,
                      '1110' : -0.5 + (math.sqrt(3)/2) * 1j,
                      '1111' : -1 * (math.sqrt(2)/2) + (math.sqrt(2)/2) * 1j}
        
        # return [[psk16_map[string] for string in word] for word in grouped_bits]
        return [psk16_map[string] for string in grouped_bits]
    
    elif(mod == "16QAM"): # good
        qam16_map = {'0000' : (math.sqrt(2)/6) + (math.sqrt(2)/6) * 1j,
                      '0001' : (math.sqrt(2)/2) + (math.sqrt(2)/6) * 1j,
                      '0010' : (math.sqrt(2)/6) + (math.sqrt(2)/2) * 1j,
                      '0011' : (math.sqrt(2)/2) + (math.sqrt(2)/2) * 1j,
                      '0100' : (math.sqrt(2)/6) - (math.sqrt(2)/6) * 1j,
                      '0101' : (math.sqrt(2)/6) - (math.sqrt(2)/2) * 1j,
                      '0110' : (math.sqrt(2)/2) - (math.sqrt(2)/6) * 1j,
                      '0111' : (math.sqrt(2)/2) - (math.sqrt(2)/2) * 1j, 
                      '1000' : -1 * (math.sqrt(2)/6) + (math.sqrt(2)/6) * 1j,
                      '1001' : -1 * (math.sqrt(2)/6) + (math.sqrt(2)/2) * 1j,
                      '1010' : -1 * (math.sqrt(2)/2) + (math.sqrt(2)/6) * 1j,
                      '1011' : -1 * (math.sqrt(2)/2) + (math.sqrt(2)/2) * 1j,
                      '1100' : -1 * (math.sqrt(2)/6) - (math.sqrt(2)/6) * 1j,
                      '1101' : -1 * (math.sqrt(2)/2) - (math.sqrt(2)/6) * 1j,
                      '1110' : -1 * (math.sqrt(2)/6) - (math.sqrt(2)/2) * 1j,
                      '1111' : -1 * (math.sqrt(2)/2) - (math.sqrt(2)/2) * 1j}
        
        # return [[qam16_map[string] for string in word] for word in grouped_bits]
        return [qam16_map[string] for string in grouped_bits]
    
    # elif(mod == "32QAM"): # not good
    #     qam32_map = {'0000' : 0.5 + 0.5j,
    #                   '0001' : 1 + 0.5j,
    #                   '0010' : 0.5 + 1j,
    #                   '0011' : 1 + 1j,
    #                   '0100' : 0.5 - 0.5j,
    #                   '0101' : 0.5 - 1j,
    #                   '0110' : 1 - 0.5j,
    #                   '0111' : 1 - 1j, 
    #                   '1000' : -0.5 + 0.5j,
    #                   '1001' : -0.5 + 1j,
    #                   '1010' : -1 + 0.5j,
    #                   '1011' : -1 + 1j,
    #                   '1100' : -0.5 - 0.5j,
    #                   '1101' : -1 - 0.5j,
    #                   '1110' : -0.5 - 1j,
    #                   '1111' : -1 - 1j}
        
    #     return [[qam32_map[string] for string in word] for word in grouped_bits]
    
    # elif(mod == "64QAM"): # not good
    #     qam64_map = {'0000' : 0.5 + 0.5j,
    #                   '0001' : 1 + 0.5j,
    #                   '0010' : 0.5 + 1j,
    #                   '0011' : 1 + 1j,
    #                   '0100' : 0.5 - 0.5j,
    #                   '0101' : 0.5 - 1j,
    #                   '0110' : 1 - 0.5j,
    #                   '0111' : 1 - 1j, 
    #                   '1000' : -0.5 + 0.5j,
    #                   '1001' : -0.5 + 1j,
    #                   '1010' : -1 + 0.5j,
    #                   '1011' : -1 + 1j,
    #                   '1100' : -0.5 - 0.5j,
    #                   '1101' : -1 - 0.5j,
    #                   '1110' : -0.5 - 1j,
    #                   '1111' : -1 - 1j}
        
        # return [[qam64_map[string] for string in word] for word in grouped_bits]
    
    else:
        
        raise ValueError("Modulation type must be OOK, 2ASK, 4ASK, BPSK, QPSK, 16PSK, 16QAM, 32QAM, or 64QAM.")

# returns the phase and amplitude of each symbol in a list- is called by qamModulation
def getSymbolInfo(symbols):
    phases = []
    amplitudes = []
    
    for symbol in symbols:
        amplitudes.append(abs(symbol))
        phases.append(phase(symbol))
        
    return amplitudes, phases

# returns the baseband
def apskModulation(symbols, sampleRate, symbolRate):
    amps, phs = getSymbolInfo(symbols)
    sampsPerSym = sampleRate * (1 / symbolRate)
        
    amplitudes = np.array([])
    phases = np.array([])
    
    for i in range(0, len(amps)):
        amplitudes = np.append(amplitudes, np.full(int(sampsPerSym) - 1, amps[i]))
        phases = np.append(phases, np.full(int(sampsPerSym) - 1, phs[i]))
    
    # FIXME: Did I do this right???
    inPhase = amplitudes * np.cos(phases)
    quadPhase = amplitudes * np.sin(phases)
    
    return np.array([inPhase, quadPhase])


