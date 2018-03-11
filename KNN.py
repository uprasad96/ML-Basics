import csv
import random
import math
import operator

#Load the IRIS dataset csv file and convert the data into a list of lists (2D array)  , make sure the data file is in the current working directory
#Randomly split dataset to training and testing data

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):#coverting string format to float format
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

#this function calculates the euclidian distance between 2 points to give a measure of dissimilarity between the 2
def euclideanDistance(point1, point2, length):
    distance = 0
    for x in range(length):
        distance += pow((point1[x] - point2[x]), 2)
    return math.sqrt(distance)

#a straight forward process of calculating the distance for all instances and selecting a subset with the smallest distance values.
# function that returns k most similar neighbors from the training set for a given test instance (using the already defined euclideanDistance function)
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))#operator is in built module provides a set of convenient operators also, assumes input is an iterable (tuple and fetches the nth object out of it)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#Once we have located the most similar neighbors for a test instance, the next task is to devise a predicted response based on those neighbors.
#We can do this by allowing each neighbor to vote for their class attribute, and take the majority vote as the prediction.
#Below provides a function for getting the majority voted response from a number of neighbors. It assumes the class is the last attribute for each neighbor.
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# the getAccuracy function that sums the total correct predictions and returns the accuracy as a percentage of correct classifications.
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('iris.data.csv', split, trainingSet, testSet)

    'Train set: ' + repr(len(trainingSet))

    'Test set: ' + repr(len(testSet))
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()