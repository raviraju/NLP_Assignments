import argparse, pprint, os
from pathlib import Path
import json
from collections import defaultdict

SPAM_CATEGORY = "spam"
HAM_CATEGORY = "ham"

def rscandir2(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                yield (root, file)
# def rscandir(path):
#     for entry in os.scandir(path):
#         yield entry
#         if entry.is_dir():
#             yield from rscandir(entry.path)

def loadModel():
    modelDataFile = open("per_model.txt", "r")
    modelData = json.loads(modelDataFile.read())
    #print(modelData, type(modelData))
    weights     = modelData["weights"]
    bias        = modelData["bias"]
    return (weights, bias)

def classifyEmail(filePath, emailCategory, weights, bias, verboseLog):    
    email = open(filePath, "r", encoding="latin1")
    
    testEmailVector = defaultdict(int)
    for line in email:
        tokens = line.split()
        for token in tokens:
            testEmailVector[token] += 1

    
    activation = 0    
    for token in testEmailVector.keys():
        if token in weights.keys():
            activation += weights[token] * testEmailVector[token]
        else:
            if verboseLog:
                print("ignore token : {}, not found in vocabulary".format(token))
            continue    #ignore tokens not found in vocabulary
            
    if(activation + bias > 0):
        return SPAM_CATEGORY
    else:
        return HAM_CATEGORY

def classifyTestData(testPath, outputPath, **kwargs):
    (weights, bias) = loadModel()
    outfile = open(outputPath, 'w')
    evalfile = open('perEvaluation.txt', 'w')
    no_of_emails_processed = 0

    prediction = {}
    prediction[SPAM_CATEGORY] = 0
    prediction[HAM_CATEGORY] = 0
    
    labelled = {}
    labelled[SPAM_CATEGORY] = 0
    labelled[HAM_CATEGORY] = 0
    
    correctClassification = {}
    correctClassification[SPAM_CATEGORY] = 0
    correctClassification[HAM_CATEGORY] = 0
    
    if kwargs["dumpEvaluation"]:
        print("***********************************Discrepancies***********************************", file = evalfile)
    for path,fileName in rscandir2(testPath):
        filePath = os.path.join(path, fileName)
        emailCategory = ""
        if kwargs["evaluate"]:
            emailCategory = Path(path).name
            if emailCategory == SPAM_CATEGORY:
                labelled[SPAM_CATEGORY] += 1
            elif emailCategory == HAM_CATEGORY:
                labelled[HAM_CATEGORY] += 1
            else:
                print("UnRecognized emailCategory : {}. Hence Skipping {}".format(emailCategory, fileName))
                continue
        no_of_emails_processed += 1
        if kwargs["verbose"]:
            print("Testing email file : ", fileName )
        label_predicted = classifyEmail(filePath, emailCategory, weights, bias, kwargs["verbose"])
        if kwargs["evaluate"]:
            prediction[label_predicted] +=1
            if label_predicted == emailCategory:
                correctClassification[emailCategory] += 1
            else:
                if kwargs["dumpEvaluation"]:
                    print("label : {} prediction : {} for file : {}".format(emailCategory, label_predicted, filePath.replace('\\','/')), file = evalfile)
        if kwargs["verbose"]:
            print(label_predicted, filePath.replace('\\','/'))            
        print("{} {}".format(label_predicted, filePath.replace('\\','/')), file = outfile)
    
    if kwargs["evaluate"]:
        if kwargs["verbose"]:
            print("correctClassification statistics:")
            print(correctClassification)
            print("prediction statistics:")
            print(prediction)
            print("labelled statistics:")
            print(labelled)
        precision = {}
        precision[SPAM_CATEGORY] = 0
        precision[HAM_CATEGORY] = 0
        if prediction[SPAM_CATEGORY] != 0 :
            precision[SPAM_CATEGORY] = correctClassification[SPAM_CATEGORY] / prediction[SPAM_CATEGORY]
        if prediction[HAM_CATEGORY] != 0 :
            precision[HAM_CATEGORY] = correctClassification[HAM_CATEGORY] / prediction[HAM_CATEGORY]
        recall = {}
        recall[SPAM_CATEGORY] = 0
        recall[HAM_CATEGORY] = 0
        if labelled[SPAM_CATEGORY] != 0 :
            recall[SPAM_CATEGORY] = correctClassification[SPAM_CATEGORY] / labelled[SPAM_CATEGORY]
        if labelled[HAM_CATEGORY] != 0 :
            recall[HAM_CATEGORY] = correctClassification[HAM_CATEGORY] / labelled[HAM_CATEGORY]
        f1_score = {}
        f1_score[SPAM_CATEGORY] = 0
        f1_score[HAM_CATEGORY] = 0
        if (precision[SPAM_CATEGORY] + recall[SPAM_CATEGORY]) != 0:
            f1_score[SPAM_CATEGORY] = (2*precision[SPAM_CATEGORY]*recall[SPAM_CATEGORY]) / (precision[SPAM_CATEGORY] + recall[SPAM_CATEGORY])
        if (precision[HAM_CATEGORY] + recall[HAM_CATEGORY]) != 0:
            f1_score[HAM_CATEGORY] = (2*precision[HAM_CATEGORY]*recall[HAM_CATEGORY]) / (precision[HAM_CATEGORY] + recall[HAM_CATEGORY])
        if kwargs["dumpEvaluation"]:
            print("***********************************Evaluation***********************************", file = evalfile)
            print("precision statistics:", file = evalfile)
            print(precision, file = evalfile)
            print("recall statistics:", file = evalfile)
            print(recall, file = evalfile)
            print("f1_score statistics:", file = evalfile)
            print(f1_score, file = evalfile)
            #average weighted by number of examples:
            #(Number_of_Spam_examples * Spam_F1_score)/Total_number_of_examples +  (Number_of_Ham_examples * Ham_F1_score)/ Total_number_of_examples
            weighted_avg = ( (labelled[SPAM_CATEGORY] * f1_score[SPAM_CATEGORY]) + (labelled[HAM_CATEGORY] * f1_score[HAM_CATEGORY]) )/no_of_emails_processed
            print("weighted_avg : ", weighted_avg, file = evalfile)    
    
    
def main():
    parser = argparse.ArgumentParser(description="use a stored perceptron model to classify data from test set")
    parser.add_argument("-v", "--verbose",          help="increase output verbosity", action="store_true")
    parser.add_argument("-e", "--evaluate",         help="compute model evaluation scores", action="store_true")
    parser.add_argument("-d", "--dumpEvaluation",   help="dump evaluations statistics", action="store_true")
    parser.add_argument("testDataPath", help="path to test data")
    parser.add_argument("outputPath", help="path to outputfile")
    args = parser.parse_args()
    
    testPath = args.testDataPath
    outputPath = args.outputPath
        
    classifyTestData(testPath, outputPath, verbose = args.verbose, evaluate = args.evaluate, dumpEvaluation = args.dumpEvaluation)
    
if __name__ == "__main__" : main()