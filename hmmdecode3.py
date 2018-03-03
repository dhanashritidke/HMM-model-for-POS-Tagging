import json
import codecs
import sys
from collections import OrderedDict
from readwrite import writeOutput


class POSTagger:

    def __init__(self, testFilePath):
        self.testFilePath = testFilePath
        self.modelParameterFile = "hmmmodel.txt"
        self.outputFile = "hmmoutput.txt"
        with codecs.open(self.modelParameterFile, mode='r', encoding="utf-8") as modelFile:
            model = json.load(modelFile)
        self.transitionMatrix = OrderedDict(model['transition'])
        self.emissionMatrix = OrderedDict(model['emission'])
        with codecs.open(self.testFilePath, mode='r', encoding="utf-8") as testFile:
            self.testData = testFile.readlines()
        self.possibleTags = self.emissionMatrix.keys()
        self.startState = "start"

    def get_emission_prob(self, state, word):
        if word in self.emissionMatrix[state]:
            return self.emissionMatrix[state][word]
        else:
            return 1

    def get_transition_prob(self, prevState, currentState):
        return self.transitionMatrix[prevState][currentState]

    def get_tag_sequence(self, backpointers, probabilities, line, T):
        finalTag = max(probabilities[T].keys(), key=(lambda key: probabilities[T][key]))
        outputLine=[]
        outputLine.append([line[T-1],finalTag])
        currentTag = finalTag
        for t in range(T, 1, -1):
            outputLine.append([line[t-2],backpointers[t][currentTag]])
            currentTag = backpointers[t][currentTag]
        return outputLine[::-1]

    def viterbi_decoding(self, line):
        line = line.strip().split(" ")
        T = len(line)
        backpointers = OrderedDict([])
        probabilities = OrderedDict([])
        for t in range(1, T + 1):
            backpointers[t] = {}
            probabilities[t] = {}

        for state in self.transitionMatrix[self.startState]:
            backpointers[1][state] = self.startState
            probabilities[1][state] = self.get_emission_prob(state, line[0]) * self.get_transition_prob(self.startState,state)

        for t in range(2, T + 1):
            for prevState in self.possibleTags:
                for currentState in self.possibleTags:
                    transitionProb = self.get_transition_prob(prevState, currentState)
                    emissionProb = self.get_emission_prob(currentState, line[t - 1])
                    currentProb = probabilities[t - 1][prevState] * transitionProb * emissionProb
                    if currentState in probabilities[t]:
                        if currentProb > probabilities[t][currentState]:
                            probabilities[t][currentState] = currentProb
                            backpointers[t][currentState] = prevState
                    else:
                        probabilities[t][currentState] = currentProb
                        backpointers[t][currentState] = prevState

        return self.get_tag_sequence(backpointers, probabilities, line, T)

    def hmm_decode(self):

        with codecs.open(self.outputFile, mode='a', encoding="utf-8") as outputFile:
            outputFile=[]
            for line in self.testData:
                outputLine = self.viterbi_decoding(line)
                outputFile.append(outputLine)
            writeOutput(outputFile)

hmmDecode = POSTagger(sys.argv[1])
hmmDecode.hmm_decode()