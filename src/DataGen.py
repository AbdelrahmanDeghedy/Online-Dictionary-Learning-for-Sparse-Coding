import numpy as np
from numpy.random import uniform


def generate_wave(frequency, timeInterval, segmentSize):
    startTimeInterval = timeInterval[0]
    endTimeInterval = timeInterval[1]
    step = (endTimeInterval - startTimeInterval) / segmentSize

    return [np.cos(2 * np.pi * frequency * (startTimeInterval + step * t)) for t in range(segmentSize)]

def add_noise(data, std_deviation):
    return [[x + np.random.normal(scale=std_deviation) for x in wave] for wave in data]

def merge_signals(base_signals, coefficients_list):
    return [np.dot(coefficient, base_signals) for coefficient in coefficients_list]


def generate_coefficients(dict_size, sparsity, coefficients_range):
    bernoulli_mask = np.random.binomial(1, 1 - sparsity, dict_size)
    coefficients_list = [uniform(coefficients_range[0], coefficients_range[1])
                         for _ in range(dict_size)]
    return [bernoulli_mask[i] * coefficients_list[i] for i in range(dict_size)]

def generate(frequenciesList, numberOfObservations, timeInterval, segmentSize, std_deviation, coefficients_range, sparsity):
    numberOfSamples = len(frequenciesList)
    
    base_signals = [generate_wave(frequenciesList[i], timeInterval, segmentSize)
                        for i in range(numberOfSamples)]
    
    coefficients_list = [generate_coefficients(dict_size=numberOfSamples,
                                                sparsity=sparsity,
                                                coefficients_range=coefficients_range)
                            for _ in range(numberOfObservations)]
    
    data = merge_signals(base_signals=base_signals, coefficients_list=coefficients_list)
    noisy_data = add_noise(data, std_deviation)

    return noisy_data


