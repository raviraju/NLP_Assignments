import argparse, os
from hw3_corpus_tool import get_data_fileName
from pprint import pprint
import pycrfsuite
import time

trainer = pycrfsuite.Trainer(verbose=False)
iterations = 100

def getFeatures(utterances):
    xseq = []
    yseq = []
    utterance_no = 0
    for utterance in utterances:
        firstUtterance = False
        speakerChange = False
        if utterance_no == 0:
            firstUtterance = True
            #print(utterance.act_tag)
            #print(utterance.speaker)
            #print(utterance.pos)
            #print(utterance.text)
            current_speaker =  utterances[utterance_no].speaker
        else:
            previous_utterance_no = utterance_no-1
            previous_speaker = utterances[previous_utterance_no].speaker
            current_speaker =  utterances[utterance_no].speaker
            if(previous_speaker != current_speaker):
                speakerChange = True
        #print("{} {} {}".format(firstUtterance, speakerChange, current_speaker))
        utterance_features = [str(int(firstUtterance)), str(int(speakerChange))]
        if utterance.pos:
            tokens = []
            posTags = []
            for token_pos in utterance.pos:
                tokens.append('TOKEN_' + token_pos.token)
                posTags.append('POS_' + token_pos.pos)
            utterance_features.extend(tokens)
            utterance_features.extend(posTags)
        #else:
            #print(utterance)
        utterance_label = utterance.act_tag
        #print(utterance_label, end = '\t')
        #print(utterance_features)
        xseq.append(utterance_features)
        yseq.append(utterance_label)
        utterance_no += 1
    #print(len(xseq), len(yseq))
    #print(xseq[0])
    #print(yseq)
    return(xseq, yseq)

def loadTrainingData(trainingDirPath):
    all_conversations = list(get_data_fileName(trainingDirPath))
    for conversation in all_conversations:
        #print(conversation["fileName"])
        utterances = conversation["data"]
        (xseq, yseq) = getFeatures(utterances)
        #print(len(xseq), len(yseq))
        trainer.append(xseq,yseq)
    print("Loaded Training Data")

def trainCRFModel(binaryModelFile):
    #for param in trainer.get_params():
    #    print(param, trainer.help(param))
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': iterations,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(binaryModelFile)
    #print(trainer.logparser.last_iteration)
    print("Trained Baseline_CRF model with {} iterations".format(iterations))

def makePrediction(testDirPath, binaryModelFile, outputFileName):
    tagger = pycrfsuite.Tagger()
    tagger.open(binaryModelFile)
    #print(tagger.info())
    #print(tagger.labels())
    all_conversations = list(get_data_fileName(testDirPath))
    with open(outputFileName, 'w') as outFile:
        for conversation in all_conversations:
            print("Filename=\"{}\"".format(os.path.basename(conversation["fileName"])), file = outFile)
            utterances = conversation["data"]
            (xseq, yseq) = getFeatures(utterances)
            #print(len(xseq), len(yseq))
            #print(xseq[0], yseq[0])
            #print(xseq[1], yseq[1])
            #print(xseq[2], yseq[2])
            
            prediction = tagger.tag(xseq)
            #print(prediction)
            for predicted_tag in prediction:
                print(predicted_tag, file = outFile)
            print("", file = outFile)
    print("Predicted Sequential Labels for Test Data")

def main():
    parser = argparse.ArgumentParser(description="learn a CRF model for baseline set of features")
    parser.add_argument("inputDir",     help="path to training data")
    parser.add_argument("testDir", 	    help="path to test data")
    parser.add_argument("outputFile",   help="output file containing predicted labels")
    args = parser.parse_args()
    
    trainingDirPath = args.inputDir
    testDirPath = args.testDir
    outputFileName = args.outputFile
    binaryModelFile = 'baseline_crfmodel'
    
    start_time = time.time()
    loadTrainingData(trainingDirPath)
    trainCRFModel(binaryModelFile)
    makePrediction(testDirPath, binaryModelFile, outputFileName)
    print("--- %s seconds ---" % (time.time() - start_time))



#baseline_crf.py INPUTDIR TESTDIR OUTPUTFILE
if __name__ == "__main__" : main()
