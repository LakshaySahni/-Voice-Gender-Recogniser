"""
@author:
    Lakshay Sahni
"""
import csv
import random
import math


def loadcsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

filename = "voice_edited.csv"
dataset = loadcsv(filename)

# to split the dataset


def splitDataset(datset, splitratio):
    trainsize = int(len(datset) * splitratio)
    trainset = []
    copy = list(dataset)
    while len(trainset) < trainsize:
        index = random.randrange(len(copy))
        trainset.append(copy.pop(index))
    return[trainset, copy]  # copy is testset

print splitDataset(dataset, 0.67)

# to summarize the dataset


def seperateByClass(dataset):
    seperated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in seperated:
            seperated[vector[-1]] = []
        seperated[vector[-1]].append(vector)
    return seperated
# test code
print seperateByClass(dataset)

# to print mean of a 1D list


def mean(numbers):
    sum = 0
    for i in range(len(numbers)):
        sum += numbers[i]
    return sum / float(len(numbers))

# to print standard deviation of a 1D list


def stddev(numbers):
    stdsum = 0
    for i in range(len(numbers)):
        stdsum += pow(mean(numbers) - numbers[i], 2)
    return math.sqrt(stdsum / float((len(numbers) - 1)))


def summarize(dataset):
    summaries = [(mean(attribute), stddev(attribute))
                 for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries
print summarize(dataset)


def summarizeByClass(dataset):
    seperated = seperateByClass(dataset)
    summaries = {}
    for classValue, instances in seperated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries
print summarizeByClass(dataset)


def calculateProbability(x, mean, stddev):
    exponent = math.exp(-((math.pow(x - mean, 2)) / (2 * stddev**2)))
    return (1.0 / (math.sqrt(math.pi * 2) * stddev) * exponent)
print calculateProbability(71.5, 73, 6.2)


def classprob(summaries, input):
    probabilities = {}
    for classvalue, classSum in summaries.iteritems():
        probabilities[classvalue] = 1
        for i in range(len(classSum)):
            mean, stddev = classSum[i]
            x = input[i]
            probabilities[classvalue] *= calculateProbability(x, mean, stddev)
    return probabilities

summaries = {0: [(1, 0.5)], 1: [(20.0, 5.0)]}
input = [1.1, "?"]
probabilities = classprob(summaries, input)
print probabilities


def predict(summaries, inputVector):
    prob = classprob(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in prob.iteritems():
        if bestLabel is None or probability > bestProb:
            bestLabel = classValue
            bestProb = probability
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(predictions))) * 100.0


def main():
    filename = "voice_edited.csv"
    split = 0.67
    dataset = loadcsv(filename)
    train, test = splitDataset(dataset, split)
    summ = summarizeByClass(train)
    predictions = getPredictions(summ, test)
    accuracy = getAccuracy(test, predictions)
    print accuracy
main()
