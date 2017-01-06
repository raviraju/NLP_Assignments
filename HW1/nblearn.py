import argparse, os, pprint, json
from pathlib import Path
from math import log2

#from nltk.corpus import stopwords

SPAM_CATEGORY = "spam"
HAM_CATEGORY = "ham"

#stopWordsSet = set(stopwords.words('english'))
stopWordsSet = {'is', 'under', 'about', 'will', 'other', 'these', 'hers', 'nor', 'did', 'won', 'but', 't', 'my', 'they', 'own', 'by', 'me', 'same', 'hasn', 'just', 'so', 'to', 'how', 'not', 'm', 'wouldn', 'yourself', 'herself', 'being', 'he', 'am', 'again', 'each', 'than', 'against', 'some', 'during', 'over', 'your', 'himself', 'haven', 'because', 'has', 'those', 'any', 'and', 'before', 'don', 'were', 'o', 'above', 'myself', 'this', 'doing', 'its', 've', 'should', 'll', 'then', 'having', 'had', 'here', 'such', 'weren', 'ma', 'shouldn', 'been', 'have', 'does', 'the', 'or', 'why', 'very', 'on', 'him', 'in', 'while', 'd', 'after', 'now', 'no', 'most', 'all', 's', 'shan', 'as', 'their', 'if', 'an', 'be', 'do', 'down', 'her', 'are', 'was', 'more', 'it', 'with', 'there', 'y', 'which', 'once', 'at', 'off', 'wasn', 'theirs', 'below', 'a', 'couldn', 'yours', 'themselves', 're', 'mightn', 'mustn', 'when', 'between', 'doesn', 'didn', 'ours', 'i', 'that', 'what', 'both', 'yourselves', 'through', 'who', 'aren', 'needn', 'into', 'for', 'up', 'until', 'whom', 'itself', 'from', 'ourselves', 'she', 'further', 'too', 'you', 'his', 'we', 'only', 'them', 'ain', 'where', 'out', 'isn', 'our', 'of', 'hadn', 'few', 'can'}

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

def parseEmailFile(filePath, emailCategory, vocabulary, bagOfWords, **kwargs):
    email = open(filePath, "r", encoding="latin1")
    for line in email:
        tokens = line.split()
        for token in tokens:
            if kwargs.get("ignoreCharToken", None):             
                if len(token) == 1:
                    #print(token, ord(token))
                    if token in kwargs["singleTokenVocab"]:
                        kwargs["singleTokenVocab"][token] += 1
                    else:
                        kwargs["singleTokenVocab"][token] = 0
                    continue#skip single char tokens
            if kwargs.get("ignoreStopWords", None):
                if token in stopWordsSet:
                    if token in kwargs["stopWordsVocab"]:
                        kwargs["stopWordsVocab"][token] +=1
                    else:
                        kwargs["stopWordsVocab"][token] = 0
                    continue#skip stop words
            vocabulary.add(token)
            if token in bagOfWords[emailCategory].keys():
                if kwargs.get("binaryNB", None):
                    continue#skip incrementing token count
                bagOfWords[emailCategory][token] += 1
            else:
                bagOfWords[emailCategory][token] = 1

def learnModel(**kwargs):
    vocabulary = kwargs["vocabulary"]
    bagOfWords = kwargs["bagOfWords"]
    total_no_of_emails = kwargs["total_no_of_emails"]
    no_of_spam_emails = kwargs["no_of_spam_emails"]
    no_of_ham_emails = kwargs["no_of_ham_emails"]
    no_of_unique_words_in_vocabulary = len(vocabulary)
    try:
        if kwargs["verbose"]:
            print("vocabulary size : ", no_of_unique_words_in_vocabulary)
            #print(repr(vocabulary))
            print("bagOfWords")
            pprint.pprint(bagOfWords)
            print("total_no_of_emails : ", total_no_of_emails)
            print("no_of_spam_emails : ", no_of_spam_emails)
            print("no_of_ham_emails : ", no_of_ham_emails)
    except UnicodeEncodeError as e:
        print("Unable to dump bagOfWords in windows console due to ", e)
    logPrior = {}
    #WITHOUT_LOG logPrior[SPAM_CATEGORY] = no_of_spam_emails/total_no_of_emails
    #WITHOUT_LOG logPrior[HAM_CATEGORY] = no_of_ham_emails/total_no_of_emails
    logPrior[SPAM_CATEGORY] = log2(no_of_spam_emails) - log2(total_no_of_emails)    #WITH_LOG
    logPrior[HAM_CATEGORY] =  log2(no_of_ham_emails) - log2(total_no_of_emails)     #WITH_LOG
    if kwargs["verbose"]:
        print("logPrior")
        pprint.pprint(logPrior)
    
    total_no_of_words= {}
    total_no_of_words [SPAM_CATEGORY] = 0
    total_no_of_words [HAM_CATEGORY] = 0
    for key in bagOfWords[SPAM_CATEGORY].keys():
        total_no_of_words [SPAM_CATEGORY] += bagOfWords[SPAM_CATEGORY][key]
    for key in bagOfWords[HAM_CATEGORY].keys():
        total_no_of_words [HAM_CATEGORY] += bagOfWords[HAM_CATEGORY][key]
    if kwargs["verbose"]:
        print("total_no_of_words")
        pprint.pprint(total_no_of_words)
    
    loglikelihood = {SPAM_CATEGORY : {},
                     HAM_CATEGORY : {}
                    }
    denomitorSPAM = total_no_of_words[SPAM_CATEGORY] + no_of_unique_words_in_vocabulary #add_one smoothing
    denomitorHAM = total_no_of_words[HAM_CATEGORY] + no_of_unique_words_in_vocabulary   #add_one smoothing
    for word in vocabulary:
        noOfOccurences = bagOfWords[SPAM_CATEGORY].get(word, 0)
        #WITHOUT_LOG loglikelihood[SPAM_CATEGORY][word] = (noOfOccurences + 1)/denomitorSPAM  #add_one smoothing
        loglikelihood[SPAM_CATEGORY][word] = log2(noOfOccurences + 1) - log2(denomitorSPAM) #add_one smoothing    #WITH_LOG
        
        noOfOccurences = bagOfWords[HAM_CATEGORY].get(word, 0)
        #WITHOUT_LOG loglikelihood[HAM_CATEGORY][word] = (noOfOccurences + 1)/denomitorHAM   #add_one smoothing
        loglikelihood[HAM_CATEGORY][word] = log2(noOfOccurences + 1) - log2(denomitorHAM)   #add_one smoothing    #WITH_LOG
#     if kwargs["verbose"]:
#         print("loglikelihood")
#         pprint.pprint(loglikelihood)
    
    results = {}
    results["vocabulary"] = list(vocabulary)#converting set to list, as set cannot be written to json
    results["logPrior"] = logPrior
    results["loglikelihood"] = loglikelihood
    results_str = json.dumps(results)
#     if kwargs["verbose"]:
#         print("results_str")
#         pprint.pprint(results)
    outfile = open('nbmodel.txt', 'w')
    print(results_str, file = outfile)    
    
# def learnAllLabelDataScanDir(trainingPath, **kwargs):
#     total_no_of_emails = 0
#     no_of_spam_emails = 0
#     no_of_ham_emails = 0
#     vocabulary = set()
#     bagOfWords = {SPAM_CATEGORY : {},
#                   HAM_CATEGORY : {}
#                  }
#     if kwargs["ignoreCharToken"] or kwargs["binaryNB"]:
#         singleTokenVocab = {}
#     for entry in rscandir(trainingPath):
#         #print(entry)
#         if entry.is_file() and entry.name.endswith('.txt'):
#             emailCategory = Path(entry.path).parent.name
#             fileName = entry.name
#             filePath = entry.path
#             total_no_of_emails += 1
#             if emailCategory == HAM_CATEGORY:
#                 #if kwargs["verbose"]:
#                     #print("Scan HAM  file : ", fileName )
#                 no_of_ham_emails += 1
#             elif emailCategory == SPAM_CATEGORY:
#                 #if kwargs["verbose"]:
#                     #print("Scan SPAM file : ", fileName )
#                 no_of_spam_emails +=1
#             else:
#                 print("UnRecognized emailCategory : {}. Hence Skipping {}".format(emailCategory, fileName))
#                 continue
#             if kwargs["ignoreCharToken"] or kwargs["binaryNB"]:
#                 parseEmailFile(filePath, emailCategory, vocabulary, bagOfWords, ignoreCharToken = kwargs["ignoreCharToken"], binaryNB = kwargs["binaryNB"], singleTokenVocab = singleTokenVocab)
#             else:
#                 parseEmailFile(filePath, emailCategory, vocabulary, bagOfWords)
#     if kwargs["ignoreCharToken"]:
#         print("singleTokenVocab ignored")
#         print(singleTokenVocab)
#     learnModel(vocabulary= vocabulary, 
#                verbose= kwargs["verbose"], 
#                bagOfWords = bagOfWords,
#                total_no_of_emails = total_no_of_emails,
#                no_of_spam_emails = no_of_spam_emails,
#                no_of_ham_emails = no_of_ham_emails )

def learnAllLabelData(trainingPath, **kwargs):
    total_no_of_emails = 0
    no_of_spam_emails = 0
    no_of_ham_emails = 0
    vocabulary = set()
    bagOfWords = {SPAM_CATEGORY : {},
                  HAM_CATEGORY : {}
                 }
    if kwargs["ignoreCharToken"] or kwargs["binaryNB"] or kwargs["ignoreStopWords"]:
        singleTokenVocab = {}#init singleTokenVocab if one of modification rule is enabled, so as to pass data structure
        stopWordsVocab = {}
    for path,fileName in rscandir2(trainingPath):
        emailCategory = Path(path).name
        filePath = os.path.join(path, fileName)
        total_no_of_emails += 1
        if emailCategory == HAM_CATEGORY:
            no_of_ham_emails += 1
        elif emailCategory == SPAM_CATEGORY:
            no_of_spam_emails +=1
        else:
            print("UnRecognized emailCategory : {}. Hence Skipping {}".format(emailCategory, fileName))
            continue
        if kwargs["ignoreCharToken"] or kwargs["binaryNB"] or kwargs["ignoreStopWords"]:
            parseEmailFile(filePath, emailCategory, vocabulary, bagOfWords, 
                           ignoreCharToken  = kwargs["ignoreCharToken"],
                           singleTokenVocab = singleTokenVocab, 
                           binaryNB         = kwargs["binaryNB"], 
                           ignoreStopWords  = kwargs["ignoreStopWords"],
                           stopWordsVocab   = stopWordsVocab)
        else:
            parseEmailFile(filePath, emailCategory, vocabulary, bagOfWords)
    if kwargs["ignoreCharToken"]:
        try:
            print("singleTokenVocab ignored")
            print(singleTokenVocab)
        except UnicodeEncodeError as e:
            print("Unable to dump singleTokenVocab in windows console due to ", e)
    if kwargs["ignoreStopWords"]:
        print("stopWordsVocab ignored")
        print(stopWordsVocab)
    if kwargs["threshold"]:
        try:
            removeFromSpam = False
            removeFromHam = False
            print("Before:")
            print("reset vocabulary and bagofWords to meet threshold limits : {}".format(kwargs["threshold"]))
            print("len(bagOfWords[SPAM_CATEGORY].keys()) : ", len(bagOfWords[SPAM_CATEGORY].keys()))
            print("len(bagOfWords[HAM_CATEGORY].keys()) : ", len(bagOfWords[HAM_CATEGORY].keys()))
            print("len(vocabulary) : ", len(vocabulary))
            wordsTobeDiscarded = set()
            for word in vocabulary:
                noOfOccurences = bagOfWords[SPAM_CATEGORY].get(word, None)
                if noOfOccurences and noOfOccurences < kwargs["threshold"]:
                    #print("remove {} in spam bag with freq {}".format(word,bagOfWords[SPAM_CATEGORY][word]))
                    del bagOfWords[SPAM_CATEGORY][word]
                    removeFromSpam = True
                noOfOccurences = bagOfWords[HAM_CATEGORY].get(word, None)
                if noOfOccurences and noOfOccurences < kwargs["threshold"]:
                    #print("remove {} in ham bag with freq {}".format(word,bagOfWords[HAM_CATEGORY][word]))
                    del bagOfWords[HAM_CATEGORY][word]
                    removeFromHam = True
                if removeFromSpam and removeFromHam:
                    #print("remove from vocabulary : {}".format(word))
                    wordsTobeDiscarded.add(word)
            for word in wordsTobeDiscarded:
                vocabulary.discard(word)
            print("After:")
            print("len(bagOfWords[SPAM_CATEGORY].keys()) : ", len(bagOfWords[SPAM_CATEGORY].keys()))
            print("len(bagOfWords[HAM_CATEGORY].keys()) : ", len(bagOfWords[HAM_CATEGORY].keys()))
            print("len(vocabulary) : ", len(vocabulary))
        except UnicodeEncodeError as e:
            print("Unable to dump in windows console due to ", e)
        #return
    learnModel(vocabulary= vocabulary, 
               verbose= kwargs["verbose"], 
               bagOfWords = bagOfWords,
               total_no_of_emails = total_no_of_emails,
               no_of_spam_emails = no_of_spam_emails,
               no_of_ham_emails = no_of_ham_emails )
    
# def learnPartLabelDataScanDir(trainingPath, **kwargs):
#     total_no_of_emails = 0
#     no_of_spam_emails = 0
#     no_of_ham_emails = 0
#     vocabulary = set()
#     bagOfWords = {SPAM_CATEGORY : {},
#                   HAM_CATEGORY : {}
#                  }
#     
#     for entry in rscandir(trainingPath):
#         if entry.is_file() and entry.name.endswith('.txt'):
#             emailCategory = Path(entry.path).parent.name
#             fileName = entry.name
#             total_no_of_emails += 1
#             if emailCategory == HAM_CATEGORY:
#                 no_of_ham_emails += 1
#             elif emailCategory == SPAM_CATEGORY:
#                 no_of_spam_emails +=1
#             else:
#                 print("UnRecognized emailCategory : {}. Hence Skipping {}".format(emailCategory, fileName))
#                 continue
#     percentScan = kwargs["percentScan"]
#     part_total_no_of_emails = (total_no_of_emails * percentScan) // 100
#     part_no_of_spam_emails = part_total_no_of_emails // 2 #(no_of_spam_emails * percentScan) // 100
#     part_no_of_ham_emails = part_total_no_of_emails // 2 #(no_of_ham_emails * percentScan) // 100
#     
#     if kwargs["verbose"]:
#         print("total_no_of_emails : ", total_no_of_emails)
#         print("no_of_spam_emails : ", no_of_spam_emails)
#         print("no_of_ham_emails : ", no_of_ham_emails)
#         print()
#         print("part_total_no_of_emails : ", part_total_no_of_emails)
#         print("part_no_of_spam_emails : ", part_no_of_spam_emails)
#         print("part_no_of_ham_emails : ", part_no_of_ham_emails)
#     
#     if part_total_no_of_emails<=0 or part_no_of_spam_emails<=0 or part_no_of_ham_emails <=0:
#         print("Training Data is <=0. Hence aborting learning model")
#         return
#     
#     scanned_no_of_emails = 0
#     scanned_no_of_spam_emails = 0
#     scanned_no_of_ham_emails = 0
#     for entry in rscandir(trainingPath):
#         #print(entry)
#         if entry.is_file() and entry.name.endswith('.txt'):#and scanned_no_of_spam_emails<=part_no_of_spam_emails and scanned_no_of_ham_emails<=part_no_of_ham_emails
#             emailCategory = Path(entry.path).parent.name
#             fileName = entry.name
#             filePath = entry.path
#             scanned_no_of_emails += 1
#             if emailCategory == HAM_CATEGORY and scanned_no_of_ham_emails<part_no_of_ham_emails:
#                 scanned_no_of_ham_emails += 1
#                 if kwargs["verbose"]:
#                     print("Scan HAM {} file : {}".format(scanned_no_of_ham_emails, fileName ))
#             elif emailCategory == SPAM_CATEGORY and scanned_no_of_spam_emails<part_no_of_spam_emails:
#                 scanned_no_of_spam_emails +=1
#                 if kwargs["verbose"]:
#                     print("Scan SPAM {} file : {}".format(scanned_no_of_spam_emails, fileName ))
#             else:
#                 #print("UnRecognized emailCategory : {}. Hence Skipping {}".format(emailCategory, fileName))
#                 continue
#             parseEmailFile(filePath, emailCategory, vocabulary, bagOfWords)
#              
#     learnModel(vocabulary= vocabulary, 
#            verbose= kwargs["verbose"], 
#            bagOfWords = bagOfWords,
#            total_no_of_emails = total_no_of_emails,
#            no_of_spam_emails = no_of_spam_emails,
#            no_of_ham_emails = no_of_ham_emails )
    
def learnPartLabelData(trainingPath, **kwargs):
    total_no_of_emails = 0
    no_of_spam_emails = 0
    no_of_ham_emails = 0
    vocabulary = set()
    bagOfWords = {SPAM_CATEGORY : {},
                  HAM_CATEGORY : {}
                 }
    
    for path,fileName in rscandir2(trainingPath):
        emailCategory = Path(path).name
        filePath = os.path.join(path, fileName)
        total_no_of_emails += 1
        if emailCategory == HAM_CATEGORY:
            no_of_ham_emails += 1
        elif emailCategory == SPAM_CATEGORY:
            no_of_spam_emails +=1
        else:
            print("UnRecognized emailCategory : {}. Hence Skipping {}".format(emailCategory, fileName))
            continue
    percentScan = kwargs["percentScan"]
    part_total_no_of_emails = (total_no_of_emails * percentScan) // 100
    part_no_of_spam_emails = part_total_no_of_emails // 2 #(no_of_spam_emails * percentScan) // 100
    part_no_of_ham_emails = part_total_no_of_emails // 2 #(no_of_ham_emails * percentScan) // 100
    
    if kwargs["verbose"]:
        print("total_no_of_emails : ", total_no_of_emails)
        print("no_of_spam_emails : ", no_of_spam_emails)
        print("no_of_ham_emails : ", no_of_ham_emails)
        print()
        print("part_total_no_of_emails : ", part_total_no_of_emails)
        print("part_no_of_spam_emails : ", part_no_of_spam_emails)
        print("part_no_of_ham_emails : ", part_no_of_ham_emails)
    
    if part_total_no_of_emails<=0 or part_no_of_spam_emails<=0 or part_no_of_ham_emails <=0:
        print("Training Data is <=0. Hence aborting learning model")
        return
    
    scanned_no_of_emails = 0
    scanned_no_of_spam_emails = 0
    scanned_no_of_ham_emails = 0
    for path,fileName in rscandir2(trainingPath):
        emailCategory = Path(path).name
        filePath = os.path.join(path, fileName)
        scanned_no_of_emails += 1
        if emailCategory == HAM_CATEGORY and scanned_no_of_ham_emails<part_no_of_ham_emails:
            scanned_no_of_ham_emails += 1
#             if kwargs["verbose"]:
#                 print("Scan HAM {} file : {}".format(scanned_no_of_ham_emails, fileName ))
        elif emailCategory == SPAM_CATEGORY and scanned_no_of_spam_emails<part_no_of_spam_emails:
            scanned_no_of_spam_emails +=1
#             if kwargs["verbose"]:
#                 print("Scan SPAM {} file : {}".format(scanned_no_of_spam_emails, fileName ))
        else:
            #print("UnRecognized emailCategory : {}. Hence Skipping {}".format(emailCategory, fileName))
            continue
        parseEmailFile(filePath, emailCategory, vocabulary, bagOfWords)
             
    learnModel(vocabulary= vocabulary, 
           verbose= kwargs["verbose"], 
           bagOfWords = bagOfWords,
           total_no_of_emails = total_no_of_emails,
           no_of_spam_emails = no_of_spam_emails,
           no_of_ham_emails = no_of_ham_emails )

def main():
    parser = argparse.ArgumentParser(description="learn a Naive Bayes model from labeled data")
    parser.add_argument("-p", "--percentage",   help="percentage of labelled data to learn from", type=int)
    parser.add_argument("-v", "--verbose",      help="increase output verbosity", action="store_true")
    parser.add_argument("-i", "--ignoreChar",   help="ignore single character as tokens", action="store_true")
    parser.add_argument("-s", "--ignoreStopWords",  help="ignore Stop Words from data", action="store_true")
    parser.add_argument("-b", "--binaryNB",     help="Multi-variate Bernoulli model : based on binary data", action="store_true")
    parser.add_argument("-t", "--thresholdWords",     help="set threshold level for frequency words to be processed", type=int)
    parser.add_argument("path", help="path to training data")
    args = parser.parse_args()
    
    trainingPath = args.path
    ignoreCharToken = args.ignoreChar
    binaryNB = args.binaryNB
    ignoreStopWords = args.ignoreStopWords

#     for path,fileName in rscandir2(trainingPath):
#         emailCategory = Path(path).name
#         filePath = os.path.join(path, fileName)
#         print(emailCategory, filePath)
#     return
    threshold = 0
    if args.thresholdWords:
        threshold = args.thresholdWords
        if args.verbose:
            print("Considering words which are frequent than {}".format(threshold))
    if args.percentage:
        if args.verbose:
            print("Learning from {}% of labelled data".format(args.percentage))
        learnPartLabelData(trainingPath, verbose = args.verbose, percentScan = args.percentage)
    else:
        if args.verbose:
            print("Learning from all of labelled data")
        if ignoreCharToken or binaryNB or ignoreStopWords or (threshold>0):
            print("***********************************Modification Enabled***********************************")
        learnAllLabelData(trainingPath, verbose = args.verbose, ignoreCharToken = ignoreCharToken, binaryNB = binaryNB, ignoreStopWords = ignoreStopWords, threshold = threshold)
    

if __name__ == "__main__" : main()