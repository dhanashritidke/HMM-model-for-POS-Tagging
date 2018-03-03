import json
import codecs
import sys
from collections import OrderedDict
from readwrite import read


class HMMLearn:

    def __init__(self, inputPath):
        self.trainFilePath = inputPath
        self.trainingData = read(inputPath, True)
        self.modelParameterFile = "hmmmodel.txt"
        self.startState = "start"
        self.transitionMatrix = OrderedDict([(self.startState, {})])
        self.emissionMatrix = OrderedDict([])
        self.tagCount = OrderedDict([])
        self.words = set([])

    def laplace_smoothen_transition_matrix(self):
        tags = self.tagCount.keys()
        noOfTags = len(tags)
        for currentTag in self.transitionMatrix:
            denominator = float(sum(self.transitionMatrix[currentTag].values()) + 4*noOfTags)
            for nextTag in tags:
                if nextTag in self.transitionMatrix[currentTag]:
                    self.transitionMatrix[currentTag][nextTag] += 1
                else:
                    self.transitionMatrix[currentTag][nextTag] = 1

                self.transitionMatrix[currentTag][nextTag] = self.transitionMatrix[currentTag][nextTag] / denominator

    def normalize_emission_matrix(self):
        for tag in self.emissionMatrix:
            totalOccurencesOfTag = self.tagCount[tag]
            for word in self.words:
                if word in self.emissionMatrix[tag]:
                    self.emissionMatrix[tag][word] = self.emissionMatrix[tag][word] / totalOccurencesOfTag
                else:
                    self.emissionMatrix[tag][word] = 0

    def learn_from_training_data(self):
        for line in self.trainingData:
            prevState = self.startState
            for wordAndTag in line:
                word = wordAndTag[0]
                tag = wordAndTag[1]
                self.words.add(word)

                if tag in self.tagCount:
                    self.tagCount[tag] += 1
                else:
                    self.tagCount[tag] = 1

                if prevState in self.transitionMatrix:
                    if tag in self.transitionMatrix[prevState]:
                        self.transitionMatrix[prevState][tag] += 1
                    else:
                        self.transitionMatrix[prevState][tag] = 1
                else:
                    self.transitionMatrix[prevState] = OrderedDict([(tag, 1)])

                if tag in self.emissionMatrix:
                    if word in self.emissionMatrix[tag]:
                        self.emissionMatrix[tag][word] += 1
                    else:
                        self.emissionMatrix[tag][word] = 1
                else:
                    self.emissionMatrix[tag] = OrderedDict([(word, 1)])

                prevState = tag

            if prevState in self.transitionMatrix:
                if tag in self.transitionMatrix[prevState]:
                    self.transitionMatrix[prevState][tag] += 1
                else:
                    self.transitionMatrix[prevState][tag] = 1
            else:
                self.transitionMatrix[prevState] = OrderedDict([(tag, 1)])

        self.laplace_smoothen_transition_matrix()
        self.normalize_emission_matrix()

    def outputModel(self):
        model = {"transition": self.transitionMatrix, "emission": self.emissionMatrix}
        with codecs.open(self.modelParameterFile, mode="w", encoding="utf-8") as outputFile:
            json.dump(model, outputFile)

hmm = HMMLearn(sys.argv[1])
hmm.learn_from_training_data()
hmm.outputModel()