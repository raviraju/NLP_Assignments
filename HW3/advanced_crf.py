import argparse, os
from hw3_corpus_tool import get_data_fileName
from pprint import pprint
import pycrfsuite
import time

trainer = pycrfsuite.Trainer(verbose=False)
iterations = 50

posPairSet = {('PRP', 'MD'), ('PRP', 'VBP'), ('PRP', 'VBD')}
questionTagSet = {'qy','qw','qy^d','bh','qo', 'qh', '^g', 'qw^d'}

def isImpPair(key, next_key):
    #print(key.token, next_key.token)
    #print(key.pos, next_key.pos)
    posPair = (key.pos, next_key.pos)
    if posPair in posPairSet:
        return True
    return False

def isQuestion(act_tag):
    if act_tag in questionTagSet:
        return True
    return False

def getFeatures(utterances):
    xseq = []
    yseq = []
    utterance_no = 0
    for utterance in utterances:
        firstUtterance = False
        speakerChange = False
        response = False
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
                previous_act_tag = utterances[previous_utterance_no].act_tag
                current_act_tag = utterances[utterance_no].act_tag
                if isQuestion(previous_act_tag):#on speaker change if previous one is a question current should be a response
                    response = True
        #print("{} {} {}".format(firstUtterance, speakerChange, current_speaker))
        current_act_tag = utterances[utterance_no].act_tag
                   
        utterance_features = []
        if isQuestion(current_act_tag):
            utterance_features = ["Question"]
        if response:
            utterance_features = ["Response"]
        utterance_features.extend([str(int(firstUtterance)), str(int(speakerChange))])
        
        if utterance_no > 1:
            previous_act_tag = utterances[utterance_no-1].act_tag
            previous_to_previous_act_tag = utterances[utterance_no-2].act_tag
            utterance_features.extend(["PREV_TO_PREV_" + previous_to_previous_act_tag, "PREV_" + previous_act_tag])

        utterance_label = utterance.act_tag
        if utterance_label == "sv":
            if utterance.pos:
                length = len(utterance.pos)
                for index, key in enumerate(utterance.pos):
                    #print(index, key)
                    next_index = index+1
                    if next_index < length:
                        next_key = utterance.pos[index+1]
                        #print("Next one : ", next_key)
                        if isImpPair(key, next_key):
                            #print(utterance.text)
                            #print(key, next_key)
                            utterance_features.extend(["sv_" + str(True)])
                            #print(utterance_features)
                            #print(utterance_label)
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
    print("Trained Advanced_CRF model with {} iterations".format(iterations))

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
            #parsedDump = tagger.info()
            #print(parsedDump['transitions'])
            #probability = tagger.probability(prediction)
            #zipList= list(zip(prediction, probability))
            #print(len(zipList), zipList, file=prob_outFile)
            #print(prediction)
            for predicted_tag in prediction:
                print(predicted_tag, file = outFile)
            print("", file = outFile)
    print("Predicted Sequential Labels for Test Data")

def main():
    parser = argparse.ArgumentParser(description="learn a CRF model for advanced set of features")
    parser.add_argument("inputDir",     help="path to training data")
    parser.add_argument("testDir", 	    help="path to test data")
    parser.add_argument("outputFile",   help="output file containing predicted labels")
    args = parser.parse_args()
    
    trainingDirPath = args.inputDir
    testDirPath = args.testDir
    outputFileName = args.outputFile
    binaryModelFile = 'advanced_crfmodel'
    
    start_time = time.time()
    loadTrainingData(trainingDirPath)
    trainCRFModel(binaryModelFile)
    makePrediction(testDirPath, binaryModelFile, outputFileName)
    print("--- %s seconds ---" % (time.time() - start_time))



#baseline_crf.py INPUTDIR TESTDIR OUTPUTFILE
if __name__ == "__main__" : main()
