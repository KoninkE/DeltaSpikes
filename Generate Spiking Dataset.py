import numpy as np
from matplotlib import pyplot as plt
import math
import radio_aid
import APSK_gen, FSK_gen
import random
import torch
from snntorch import spikegen
import snntorch.spikeplot as splt
import csv
import os
from itertools import cycle, islice

def genPath(basePath, modulation, carrierFreq, word):
    wordIter = 1

    while (os.path.exists(
            basePath + modulation + '\\' + str(carrierFreq) + '\\' + word + str(wordIter) + '.csv') == True):
        wordIter += 1

    # globalPath = basePath + modulation + '\\' + str(carrierFreq) + '\\' + word + str(wordIter) + '.wav'
    globalPath = basePath + modulation + '\\' + str(carrierFreq) + '\\' + word + str(wordIter) + '.csv'

    return globalPath

# creates folders in base directory if they don't already exists
def checkPath(basePath, carriers):
    modulationExtensions = ['\OOK', '\ASK2', '\ASK4', '\FSK2', '\FSK4', '\BPSK', '\QPSK', '\PSK16', '\QAM16']
    carrierExtensions = ['\\' + str(carrier) for carrier in carriers]

    for mod in modulationExtensions:
        if (os.path.exists(basePath + mod) == False):
            os.makedirs(basePath + mod)
        for carrier in carrierExtensions:
            if (os.path.exists(basePath + mod + carrier) == False):
                os.makedirs(basePath + mod + carrier)

def randSubstet(dataSet, wordData, subsetSize):
    subset = []
    words = []

    for i in range(0, subsetSize):
        r = random.randint(0, len(dataSet) - 1)
        subset.append(dataSet[r])
        words.append(wordData[r])

    return subset, words

def extendSig(sig, totLen):
    newSig = list(islice(cycle(sig), totLen))
    return newSig

def gen_spikes(data, spike_thresh = 0.5):
    data = torch.from_numpy(data)
    spike_data = spikegen.delta(data, threshold=spike_thresh, off_spike=False)
    return spike_data

def plot_spikes(spike_data):
    fig = plt.figure(facecolor="w", figsize=(20, 2))
    ax = fig.add_subplot(111)

    splt.raster(spike_data, ax, c="black")

    plt.title("Spike Plot")
    plt.xlabel("Time step")
    plt.yticks([])
    plt.xlim(0, len(spike_data))
    plt.show()

def plot_time_series(sig, title):
    plt.figure()
    plt.plot(sig)
    plt.title(title)
    plt.show()

# generates a .csv file using a generated signal
def writeSig(path, sig, extendLen = -1):
    if (extendLen > len(sig)):
        sig = extendSig(sig, extendLen)

    sig = [str(int(pnt)) for pnt in sig]

    with open(path, 'w') as fid:
        w = csv.writer(fid, lineterminator='\n')
        w.writerows(sig)

def genAPSK(symbols, carrier, sampleRate, symbolRate):
    inPhase, quadPhase = APSK_gen.apskModulation(symbols, sampleRate, symbolRate)
    times = np.arange(0, (1 / sampleRate) * (len(inPhase)), 1 / sampleRate)

    if(len(times) > len(inPhase)):
        times = times[0:-1 * (len(times) - len(inPhase))]

    iSig = inPhase * np.cos(2 * np.pi * carrier * times)
    qSig = quadPhase * np.sin(2 * np.pi * carrier * times)
    return iSig - qSig

def genFSK2(groupedBits, carrier, sampleRate, symbolRate):
    iSig, qSig = FSK_gen.modulateFSK2(groupedBits, carrier, sampleRate, symbolRate)
    return iSig - qSig

def genFSK4(groupedBits, carrier, sampleRate, symbolRate):
    iSig, qSig = FSK_gen.modulateFSK4(groupedBits, carrier, sampleRate, symbolRate)
    return iSig - qSig

carriers = []

def gen_spiking_data(bitData, wordData, carriers, symbolRate,
                     sigsPerCarrier, basePath, fskOffset, sampleRate = -1, spike_thresh=0.5):
    # if a sample rate is not provided, then we use 15x the carrier frequency as the sample rate
    if (sampleRate == -1):
        sampleRate = max(carriers) * 15

    checkPath(basePath, carriers)

    extendLength = math.ceil((sampleRate / symbolRate) * len(max(bitData, key=len)))

    # iterates through all of the carriers that are provided in the carriers list
    for carrier in carriers:

        # user provides a number of signals they want each modulation to have per carrier frequency
        for i in range(0, sigsPerCarrier):
            data, words = randSubstet(bitData, wordData, 9)

            # OOK signal generation
            path = genPath(basePath, '\OOK', carrier, words[0])
            groupedBits = radio_aid.groupBits(data[0], 1)
            symbols = APSK_gen.mapSymbols(groupedBits, mod="OOK")
            sig = genAPSK(symbols, carrier, sampleRate, symbolRate)
            spikes = gen_spikes(np.array(sig), spike_thresh)
            writeSig(path, spikes.tolist(), extendLength)
            # plot_time_series(sig, "OOK")

            # 2-ASK signal generation
            path = genPath(basePath, '\ASK2', carrier, words[1])
            groupedBits = radio_aid.groupBits(data[1], 1)
            symbols = APSK_gen.mapSymbols(groupedBits, mod="2ASK")
            sig = genAPSK(symbols, carrier, sampleRate, symbolRate)
            spikes = gen_spikes(np.array(sig), spike_thresh)
            writeSig(path, spikes.tolist(), extendLength)
            # plot_time_series(sig, "2-ASK")

            # 4-ASK signal generation
            path = genPath(basePath, '\ASK4', carrier, words[2])
            groupedBits = radio_aid.groupBits(data[2], 2)
            symbols = APSK_gen.mapSymbols(groupedBits, mod="4ASK")
            sig = genAPSK(symbols, carrier, sampleRate, symbolRate)
            spikes = gen_spikes(np.array(sig), spike_thresh)
            writeSig(path, spikes.tolist(), extendLength)
            # plot_time_series(sig, "4-ASk")

            # BPSK signal generation
            path = genPath(basePath, '\BPSK', carrier, words[3])
            groupedBits = radio_aid.groupBits(data[3], 1)
            symbols = APSK_gen.mapSymbols(groupedBits, mod="BPSK")
            sig = genAPSK(symbols, carrier, sampleRate, symbolRate)
            spikes = gen_spikes(np.array(sig), spike_thresh)
            writeSig(path, spikes.tolist(), extendLength)
            # plot_time_series(sig, "BPSK")

            # QPSK signal generation
            path = genPath(basePath, '\QPSK', carrier, words[4])
            groupedBits = radio_aid.groupBits(data[4], 2)
            symbols = APSK_gen.mapSymbols(groupedBits, mod="QPSK")
            sig = genAPSK(symbols, carrier, sampleRate, symbolRate)
            spikes = gen_spikes(np.array(sig), spike_thresh)
            writeSig(path, spikes.tolist(), extendLength)
            # plot_time_series(sig, "QPSK")

            # 16-PSK signal generation
            path = genPath(basePath, '\PSK16', carrier, words[5])
            groupedBits = radio_aid.groupBits(data[5], 4)
            symbols = APSK_gen.mapSymbols(groupedBits, mod="16PSK")
            sig = genAPSK(symbols, carrier, sampleRate, symbolRate)
            spikes = gen_spikes(np.array(sig), spike_thresh)
            writeSig(path, spikes.tolist(), extendLength)
            # plot_time_series(sig, "16-PSK")

            # 16QAM signal generation
            path = genPath(basePath, '\QAM16', carrier, words[6])
            groupedBits = radio_aid.groupBits(data[6], 4)
            symbols = APSK_gen.mapSymbols(groupedBits, mod="16QAM")
            sig = genAPSK(symbols, carrier, sampleRate, symbolRate)
            spikes = gen_spikes(np.array(sig), spike_thresh)
            writeSig(path, spikes.tolist(), extendLength)
            # plot_time_series(sig, "16-QAM")

            # 2-FSK signal generation
            path = genPath(basePath, '\FSK2', carrier, words[7])
            fskOffsets = [carrier, carrier + fskOffset]
            groupedBits = radio_aid.groupBits(data[7], 1)
            sig = genFSK2(groupedBits, fskOffsets, sampleRate, symbolRate)
            spikes = gen_spikes(np.array(sig), spike_thresh)
            writeSig(path, spikes.tolist(), extendLength)
            # plot_time_series(sig, "2-FSK")

            # 4-FSK signal generation
            path = genPath(basePath, '\FSK4', carrier, words[8])
            fskOffsets = [carrier + (i * fskOffset) for i in range(0, 4)]
            groupedBits = radio_aid.groupBits(data[8], 2)
            sig = genFSK4(groupedBits, fskOffsets, sampleRate, symbolRate)
            spikes = gen_spikes(np.array(sig), spike_thresh)
            writeSig(path, spikes.tolist(), extendLength)
            # plot_time_series(sig, "4-FSK")

# where the .txt files full of words and their bit representations are stored
bitPath = r"C:\Users\ellio\Programs\Datasets\Radio Dataset\rand_bits.txt"
wordPath = r"C:\Users\ellio\Programs\Datasets\Radio Dataset\rand_words.txt"

# where the data set is being stored (folder location)
basePath = r"C:\Users\ellio\Programs\Datasets\Radio Dataset\spiking_data"

bitData = radio_aid.convToFloat(radio_aid.readBitArray(bitPath), 2)
wordData = radio_aid.readWordArray(wordPath)

carriers = [100, 200, 250]
sampleRate = 15 * max(carriers)

gen_spiking_data(bitData, wordData, carriers, 50, 1, basePath, 10, sampleRate = -1)
