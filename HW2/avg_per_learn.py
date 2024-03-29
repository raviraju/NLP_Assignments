import argparse
import os
import pprint
import json
import random
from pathlib import Path
from collections import defaultdict

SPAM_CATEGORY = "spam"
HAM_CATEGORY = "ham"
weightVocabulary = {}
bias = 0
avgWeightVocabulary = {}
avgBias = 0
count = 1
allTrainingEmails = {}
noOfTrainingIterations = 30

def rScanDir(path):
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                yield (root, file)

def parseEmailFile(filePath, emailCategory):
    # print("parseEmailFile : {}".format(filePath))
    email = open(filePath, "r", encoding="latin1")
    allTrainingEmails[filePath] = {}
    allTrainingEmails[filePath]["emailCategory"] = emailCategory
    allTrainingEmails[filePath]["data"] = defaultdict(int)
    for line in email:
        tokens = line.split()
        for token in tokens:
            weightVocabulary[token] = 0
            avgWeightVocabulary[token] = 0
            allTrainingEmails[filePath]["data"][token] += 1

def computeActivation(filePath):
    activation = 0
    tokens = allTrainingEmails[filePath]["data"].keys()
    activation = sum(weightVocabulary[token] * allTrainingEmails[filePath]["data"][token] for token in tokens)
    global bias
    return activation + bias

def getYValue(emailCategory):
    if emailCategory == SPAM_CATEGORY:
        return 1
    else:
        return -1

def trainModel(**kwargs):
    global bias, weightVocabulary, avgBias, avgWeightVocabulary, count
    for i in range(noOfTrainingIterations):
        # print("Avg Perceptron Training Iteration {}".format(i))
        if kwargs["verbose"]:
            print("Iteration {} : bias = {} avgBias = {}".format(i, bias, avgBias))
            # print("weight {}".format(weightVocabulary))
            # print("avg_weight {}".format(avgWeightVocabulary))
            # print("count {}".format(count))
        filePaths = list(allTrainingEmails.keys())  # List of filenames

        # filePaths = sorted(filePaths)
        # print("Sorted files : filePaths\n\n")
        random.shuffle(filePaths)
        #print("Processing emails...")
        for filePath in filePaths:
            # print("Processing {}".format(filePath))
            # if kwargs["verbose"]:
            #     print(filePath, allTrainingEmails[filePath])
            activation = computeActivation(filePath)
            y = getYValue(allTrainingEmails[filePath]["emailCategory"])
            # if kwargs["verbose"]:
            #     print("activation : {}, y : {}, y * activation : {}".format(activation, y, y * activation))
            if (y * activation) <= 0: #y and activation are of different signs
                tokens = allTrainingEmails[filePath]["data"].keys()
                for token in tokens:
                    weightVocabulary[token] = weightVocabulary[token] + y*allTrainingEmails[filePath]["data"][token]
                    avgWeightVocabulary[token] = avgWeightVocabulary[token] + y*count*allTrainingEmails[filePath]["data"][token]
                bias = bias + y
                avgBias = avgBias + y*count
                # if kwargs["verbose"]:
                #     print("\t   ", weightVocabulary, bias)
                #     print("\tAvg", avgWeightVocabulary, avgBias)
            count += 1 # increment counter regardless of update
    for token in weightVocabulary.keys():
        avgWeightVocabulary[token] = weightVocabulary[token] - ((1/count)*avgWeightVocabulary[token])
    avgBias = bias - ((1/count)*avgBias)

    # if kwargs["verbose"]:
    #     print()
    #     print("Final Model Parameters")
    #     print("weights {} \nbias {}\n".format(weightVocabulary, bias))
    #     print()
    #     print("avg_weights {} \navgBias {}\n".format(avgWeightVocabulary, avgBias))

    print("Avg Perceptron Model Trained. Parameters loaded into per_model.txt")
    results = {}
    results["weights"] = avgWeightVocabulary
    results["bias"] = avgBias
    results_str = json.dumps(results)
    outfile = open('per_model.txt', 'w', encoding="latin1")
    print(results_str, file=outfile)

def learnAllTrainingData(trainingPath, **kwargs):
    print("Parsing All Email Files...")
    for path, fileName in rScanDir(trainingPath):
        emailCategory = Path(path).name
        filePath = os.path.join(path, fileName)
        parseEmailFile(filePath, emailCategory)
    trainModel(verbose=kwargs["verbose"])

def learnPartTrainingData(trainingPath, **kwargs):
    percentScan = kwargs["percentScan"]
    print("Parsing {}% of Email Files".format(percentScan))
    total_no_of_emails = 0
    no_of_spam_emails = 0
    no_of_ham_emails = 0
    # count no of emails and their categories
    for path, fileName in rScanDir(trainingPath):
        emailCategory = Path(path).name
        total_no_of_emails += 1
        if emailCategory == HAM_CATEGORY:
            no_of_ham_emails += 1
        elif emailCategory == SPAM_CATEGORY:
            no_of_spam_emails += 1
        else:
            print("UnRecognized emailCategory : {}. Hence Skipping {}".format(
                emailCategory, fileName))
            continue

    part_total_no_of_emails = (total_no_of_emails * percentScan) // 100
    # (no_of_spam_emails * percentScan) // 100
    part_no_of_spam_emails = part_total_no_of_emails // 2
    # (no_of_ham_emails * percentScan) // 100
    part_no_of_ham_emails = part_total_no_of_emails // 2

    if kwargs["verbose"]:
        print("total_no_of_emails : ", total_no_of_emails)
        print("no_of_spam_emails : ", no_of_spam_emails)
        print("no_of_ham_emails : ", no_of_ham_emails)
        print()
        print("part_total_no_of_emails : ", part_total_no_of_emails)
        print("part_no_of_spam_emails : ", part_no_of_spam_emails)
        print("part_no_of_ham_emails : ", part_no_of_ham_emails)

    if part_total_no_of_emails <= 0 or part_no_of_spam_emails <= 0 or part_no_of_ham_emails <= 0:
        print("Training Data is <=0. Hence aborting learning model")
        return

    scanned_no_of_emails = 0
    scanned_no_of_spam_emails = 0
    scanned_no_of_ham_emails = 0
    for path, fileName in rScanDir(trainingPath):
        emailCategory = Path(path).name
        filePath = os.path.join(path, fileName)
        scanned_no_of_emails += 1
        if emailCategory == HAM_CATEGORY and scanned_no_of_ham_emails < part_no_of_ham_emails:
            scanned_no_of_ham_emails += 1
#             if kwargs["verbose"]:
#                 print("Scan HAM {} file : {}".format(scanned_no_of_ham_emails, fileName ))
        elif emailCategory == SPAM_CATEGORY and scanned_no_of_spam_emails < part_no_of_spam_emails:
            scanned_no_of_spam_emails += 1
#             if kwargs["verbose"]:
#                 print("Scan SPAM {} file : {}".format(scanned_no_of_spam_emails, fileName ))
        else:
            #print("UnRecognized emailCategory : {}. Hence Skipping {}".format(emailCategory, fileName))
            continue
        parseEmailFile(filePath, emailCategory)
    trainModel(verbose=kwargs["verbose"])


def main():
    parser = argparse.ArgumentParser(
        description="learn a perceptron model from labeled data using the averaged perceptron training algorithm")
    parser.add_argument("-v", "--verbose",
                        help="increase output verbosity", action="store_true")
    parser.add_argument("-p", "--percentage",
                        help="percentage of labelled data to learn from", type=int)
    parser.add_argument("path", help="path to training data")
    args = parser.parse_args()
    trainingPath = args.path

    if args.percentage:
        learnPartTrainingData(
            trainingPath, verbose=args.verbose, percentScan=args.percentage)
    else:
        learnAllTrainingData(trainingPath, verbose=args.verbose)


if __name__ == "__main__":
    main()
