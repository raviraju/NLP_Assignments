import argparse, os
from pprint import pprint
from hw3_corpus_tool import get_data_fileName
import operator

evaluateDict = {}

def loadActualLabels(devDirPath):
    all_conversations = list(get_data_fileName(devDirPath))
    for conversation in all_conversations:
        fileName = os.path.basename(conversation["fileName"])
        utterances = conversation["data"]
        yseq = []
        for utterance in utterances:
            yseq.append(utterance.act_tag)
        #print (fileName, yseq)
        if fileName in evaluateDict:
            evaluateDict[fileName]['actual'] = yseq
        else:
            print("{} wasnt found in predictedLabel output file".format(fileName))


def loadPredictedLabels(labelledoutputFileName):
    with open(labelledoutputFileName, 'r') as labelFile:
        allLines = labelFile.readlines()
        startRecording = False
        fileName = ""
        labels = []
        for line_no in range(len(allLines)):
            line = allLines[line_no]
            #print(line_no, line)
            if "Filename=" in line:
                startRecording = True
                fileName = (line.split('"'))[1]
                continue
            if line.strip() == '':
                startRecording=False
                #print(fileName)
                #print(labels)
                evaluateDict[fileName] = {}
                evaluateDict[fileName]['prediction'] = labels
                predictedLabels = evaluateDict[fileName]['prediction']
                #print(len(predictedLabels), predictedLabels)
                labels = []
            if startRecording:
                labels.append(line.strip('\n'))
            #line_no += 1

def main():
    parser = argparse.ArgumentParser(description="evaluate predicted labels by CRF classifier")
    parser.add_argument("devDir",     help="path to development data")
    parser.add_argument("labelledoutputFile", help="output file containing predicted labels")
    args = parser.parse_args()
    
    devDirPath = args.devDir
    labelledoutputFileName = args.labelledoutputFile
    
    loadPredictedLabels(labelledoutputFileName)
    loadActualLabels(devDirPath)
    
    total_no_of_matches = 0
    total_no_of_non_matches = 0
    nonMatchDict = {}
    nonMatch_sortedKeysDict = {}
    with open('resultAnalyze.txt','w') as outFile:
        for fileName in evaluateDict:
            print(fileName, file=outFile)
            predictedLabels = evaluateDict[fileName]['prediction']
            actualLabels = evaluateDict[fileName]['actual']
            print(len(predictedLabels), predictedLabels, file=outFile)
            print(len(actualLabels), actualLabels, file=outFile)
            zipList= list(zip(predictedLabels, actualLabels))
            print(len(zipList), zipList, file=outFile)
            no_of_matches = 0
            no_of_non_matches = 0
            index = 2
            for labelPair in zipList:
                if(labelPair[0] == labelPair[1]):
                    no_of_matches +=1
                else:
                    print(index, labelPair, file=outFile)
                    key = labelPair[0] + '_' + labelPair[1]
                    if key in nonMatchDict:
                        nonMatchDict[key] +=1
                    else:
                        nonMatchDict[key] = 1
                        
                    keysorted = ''.join(sorted(key))
                    if keysorted in nonMatch_sortedKeysDict:
                        nonMatch_sortedKeysDict[keysorted] +=1
                    else:
                        nonMatch_sortedKeysDict[keysorted] = 1
                        
                    no_of_non_matches += 1
                index += 1
            print("no_of_matches : {}".format(no_of_matches), file=outFile)
            print("no_of_non_matches : {}".format(no_of_non_matches), file=outFile)
            total_no_of_matches += no_of_matches
            total_no_of_non_matches += no_of_non_matches
        #print("****************************************nonMatchDict****************************************",file = outFile)
        #pprint(nonMatchDict, outFile)
        
        #sorted_nonMatchDict = sorted(nonMatchDict.items(), key=operator.itemgetter(1), reverse=True)
        #print("****************************************sorted_nonMatchDict****************************************",file = outFile)
        #pprint(sorted_nonMatchDict, outFile)
        
        sorted_nonMatch_sortedKeysDict = sorted(nonMatch_sortedKeysDict.items(), key=operator.itemgetter(1), reverse=True)
        print("****************************************sorted_nonMatch_sortedKeysDict****************************************",file = outFile)
        pprint(sorted_nonMatch_sortedKeysDict, outFile)
    accuracy = total_no_of_matches/(total_no_of_matches + total_no_of_non_matches) * 100
    print("Overall Accuracy : {}".format(accuracy))
    

#evaluate_model.py DEVDIR OUTPUTFILE
if __name__ == "__main__" : main()
