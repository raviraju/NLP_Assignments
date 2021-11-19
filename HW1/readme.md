implementing the simple but effective machine learning technique, Naïve Bayes classification, and applying it to a binary text classification task (i.e., spam detection)

nblearn.py will learn a naive Bayes model from labeled data, and nbclassify.py will use the model to classify new data.

nblearn.py will be invoked in the following way:
>python3 nblearn.py /path/to/input

    nblearn.py will learn a naive Bayes model from the training data, and write the model parameters to a file called nbmodel.txt.

nbclassify.py will be invoked in the following way:
>python3 nbclassify.py /path/to/input

    nbclassify.py should read the parameters of a naive Bayes model from the file nbmodel.txt, and classify each ".txt" file in the data directory as "ham" or "spam", and write the result to a text file called nboutput.txt

### Performance on the development data with 100% of the training data
```python
(base) $ python nblearn.py train/ -v
Learning from all of labelled data
vocabulary size :  108007
total_no_of_emails :  17029
no_of_spam_emails :  7496
no_of_ham_emails :  9533
logPrior
{'ham': -0.836991516093823, 'spam': -1.1838008595021385}

(base) $ python nbclassify.py dev/ -e -d -v
actual :
{'spam': 3675, 'ham': 1500}
prediction :
{'spam': 3617, 'ham': 1558}
correctClassification :
{'spam': 3592, 'ham': 1475}
inCorrectClassification :
{'spam': 83, 'ham': 25}
Discrepancies found in : nbDiscrepancies.txt
****************Evaluation****************
precision :
{'spam': 0.99, 'ham': 0.95}
recall :
{'spam': 0.98, 'ham': 0.98}
f1_score :
{'spam': 0.98, 'ham': 0.96}
weighted_avg :  0.97
(base) $ 
```

### Performance on the development data with 10% of the training data (-p 10)

```python
(base) $ python nblearn.py train/ -v -p 10
Learning from 10% of labelled data
total_no_of_emails :  17029
no_of_spam_emails :  7496
no_of_ham_emails :  9533

part_total_no_of_emails :  1702
part_no_of_spam_emails :  851
part_no_of_ham_emails :  851
vocabulary size :  31376
total_no_of_emails :  17029
no_of_spam_emails :  7496
no_of_ham_emails :  9533
logPrior
{'ham': -0.836991516093823, 'spam': -1.1838008595021385}

(base) $ python nbclassify.py dev/ -e -d -v 
actual :
{'spam': 3675, 'ham': 1500}
prediction :
{'spam': 3660, 'ham': 1515}
correctClassification :
{'spam': 3548, 'ham': 1388}
inCorrectClassification :
{'spam': 127, 'ham': 112}
Discrepancies found in : nbDiscrepancies.txt
****************Evaluation****************
precision :
{'spam': 0.97, 'ham': 0.92}
recall :
{'spam': 0.97, 'ham': 0.93}
f1_score :
{'spam': 0.97, 'ham': 0.92}
weighted_avg :  0.96
(base) $ 
```

### Performance on the development data with 100% of the training data & Ignore Single Char Tokens(-i)

```python

(base) $ python nblearn.py train/ -v -i
Learning from all of labelled data
**************Modification Enabled**************
Following are singleTokenVocab ignored
{',': 147217, '?': 25178, 'a': 44593, "'": 25648, '*': 12639, ';': 12860, '.': 225319, '!': 21735, ':': 93593, '/': 103473, '%': 6028, '_': 25837, '-': 228474, '[': 2457, ']': 2461, '@': 30085, 's': 14894, '$': 14646, '(': 22676, ')': 23927, 'm': 3762, '1': 10491, '7': 3953, 'e': 7013, '+': 3705, '8': 3503, '&': 3882, 'k': 996, 'i': 29339, 'v': 1298, '0': 6373, '6': 3524, '9': 2825, 't': 5131, 'r': 2127, '\\': 4406, 'o': 3089, '2': 9087, '5': 5656, '4': 5423, '3': 9262, 'p': 3289, 'g': 1583, '=': 13708, 'y': 661, 'd': 4270, 'c': 2348, 'q': 355, 'b': 2260, 'f': 1080, 'u': 1941, 'x': 2565, 'n': 1319, 'l': 2363, 'z': 278, '#': 4753, '>': 30456, 'h': 1140, '©': 67, 'w': 1288, '"': 15388, '|': 7247, '`': 1186, '^': 290, 'j': 5225, '{': 217, '}': 256, '·': 476, '±': 578, '½': 471, '¬': 161, '³': 173, '°': 124, '»': 275, 'µ': 67, '«': 24, '\xad': 154, '£': 130, '²': 56, '¦': 68, '§': 69, '~': 991, '\x11': 1, '\x16': 2, '\x14': 13, '\x12': 36, '\x13': 4, '¨': 60, '\x9c': 22, '\x9d': 9, '\x00': 4, '\x05': 11, '\x1b': 41, '\x8e': 72, '\x9e': 2, '\x81': 222, '\x8c': 90, '\x8d': 71, '\x9a': 11, '\x88': 8, '<': 101, '\x01': 1326, '\x07': 11, '\x0f': 26, '\x9f': 6, '\x0e': 0, '\x98': 4, '¶': 21, '\x19': 18, '\x8a': 9, '\x90': 0, '\x10': 6, '\x17': 0, '\x93': 67, '\x94': 72, '\x92': 65, '\x96': 4, '®': 53, 'è': 2, '\x99': 44, 'é': 0, '¡': 26, '¯': 8, '´': 1, '\x91': 1, 'â': 0, '\x95': 30, 'à': 0, '\x80': 0, '\x03': 0, '\x08': 0, '\x02': 13, '\x15': 5, '\x06': 28}
vocabulary size :  107873
total_no_of_emails :  17029
no_of_spam_emails :  7496
no_of_ham_emails :  9533
logPrior
{'ham': -0.836991516093823, 'spam': -1.1838008595021385}
(base) $ python nbclassify.py dev/ -e -d -v 
actual :
{'spam': 3675, 'ham': 1500}
prediction :
{'spam': 3626, 'ham': 1549}
correctClassification :
{'spam': 3615, 'ham': 1489}
inCorrectClassification :
{'spam': 60, 'ham': 11}
Discrepancies found in : nbDiscrepancies.txt
****************Evaluation****************
precision :
{'spam': 1.0, 'ham': 0.96}
recall :
{'spam': 0.98, 'ham': 0.99}
f1_score :
{'spam': 0.99, 'ham': 0.97}
weighted_avg :  0.98
(base) $ 

```

### Performance on the development data with 100% of the training data & Ignore Stop Words(-s)

```python
(base) $ python nblearn.py train/ -v -s
Learning from all of labelled data
**************Modification Enabled**************
Following are stopWordsVocab ignored
{'what': 3814, 'up': 4762, 'your': 20760, 'are': 14464, 'you': 38508, 'for': 37438, 'if': 10955, 'a': 44593, 'or': 13079, 'just': 3197, 'then': 1606, 'our': 10256, 'it': 14138, 'was': 5683, 'and': 58930, 'to': 83674, 'they': 3715, 're': 7002, 'on': 25840, 'the': 105032, 'in': 37423, 'of': 50620, 'no': 6777, 'that': 19536, 'be': 19499, 'out': 4293, 'have': 15763, 'this': 26376, 'below': 1851, 'into': 2826, 'more': 5265, 'should': 3392, 'by': 11325, 'is': 28489, 'with': 18286, 'here': 4840, 'not': 11344, 'now': 3508, 'we': 17791, 'at': 13837, 'any': 7008, 'some': 3243, 'own': 944, 's': 14894, 'from': 15261, 'which': 3796, 'than': 2093, 'after': 1855, 'its': 2761, 'will': 15573, 'm': 3762, 'before': 1768, 'further': 1080, 'as': 14021, 'has': 6301, 'their': 3195, 'itself': 179, 'an': 7321, 'over': 2899, 'won': 336, 'does': 1299, 'all': 8125, 'i': 29339, 'off': 1331, 'about': 4332, 'me': 9242, 'through': 2141, 'where': 1243, 'but': 4501, 'because': 1403, 'how': 2402, 'can': 8671, 'most': 1665, 'don': 1858, 't': 5131, 'so': 3916, 'were': 1956, 'my': 6342, 'o': 3089, 'other': 3215, 'only': 4007, 'am': 8356, 'do': 5545, 'these': 3745, 'above': 1152, 'them': 2181, 'there': 3903, 'him': 1468, 'both': 1283, 'against': 339, 'each': 1413, 'yours': 295, 'he': 3798, 'll': 1480, 'same': 1184, 'y': 661, 'd': 4270, 'why': 706, 'down': 898, 've': 1029, 'being': 1338, 'once': 811, 'had': 1994, 'very': 2416, 'again': 1362, 'did': 1113, 'during': 971, 'yourself': 511, 'doesn': 274, 'who': 2546, 'nor': 240, 'such': 1699, 'she': 1287, 'couldn': 140, 'few': 1132, 'when': 2657, 'between': 937, 'those': 1191, 'his': 2327, 'too': 791, 'ourselves': 73, 'while': 1048, 'under': 1413, 'been': 3938, 'until': 855, 'doing': 498, 'her': 1267, 'ma': 119, 'isn': 110, 'having': 538, 'shouldn': 62, 'ours': 30, 'mustn': 1, 'herself': 19, 'haven': 195, 'themselves': 106, 'whom': 169, 'ain': 25, 'myself': 300, 'didn': 339, 'needn': 5, 'himself': 75, 'yourselves': 16, 'aren': 56, 'shan': 17, 'hadn': 14, 'hasn': 33, 'weren': 18, 'wasn': 101, 'wouldn': 67, 'theirs': 16, 'hers': 5, 'mightn': 2}
vocabulary size :  107854
total_no_of_emails :  17029
no_of_spam_emails :  7496
no_of_ham_emails :  9533
logPrior
{'ham': -0.836991516093823, 'spam': -1.1838008595021385}
(base) $ python nbclassify.py dev/ -e -d -v 
actual :
{'spam': 3675, 'ham': 1500}
prediction :
{'spam': 3609, 'ham': 1566}
correctClassification :
{'spam': 3582, 'ham': 1473}
inCorrectClassification :
{'spam': 93, 'ham': 27}
Discrepancies found in : nbDiscrepancies.txt
****************Evaluation****************
precision :
{'spam': 0.99, 'ham': 0.94}
recall :
{'spam': 0.97, 'ham': 0.98}
f1_score :
{'spam': 0.98, 'ham': 0.96}
weighted_avg :  0.97
(base) $ 
```

### Performance on the development data with 100% of the training data & Multi-variate Bernoulli model

Multi-variate Bernoulli model is based on binary data: (-b)
Every token in the feature vector of a document is associated with the value 1 or 0. 
the value 1 means that the word occurs in the particular document, and 0 means that the word does not occur in this document

```python
(base) $ python nblearn.py train/ -v -b
Learning from all of labelled data
**************Modification Enabled**************
vocabulary size :  108007
total_no_of_emails :  17029
no_of_spam_emails :  7496
no_of_ham_emails :  9533
logPrior
{'ham': -0.836991516093823, 'spam': -1.1838008595021385}
(base) $ python nbclassify.py dev/ -e -d -v 
actual :
{'spam': 3675, 'ham': 1500}
prediction :
{'spam': 29, 'ham': 5146}
correctClassification :
{'spam': 29, 'ham': 1500}
inCorrectClassification :
{'spam': 3646, 'ham': 0}
Discrepancies found in : nbDiscrepancies.txt
****************Evaluation****************
precision :
{'spam': 1.0, 'ham': 0.29}
recall :
{'spam': 0.01, 'ham': 1.0}
f1_score :
{'spam': 0.02, 'ham': 0.45}
weighted_avg :  0.14
(base) $ 
```

### Performance on the development data with 100% of the training data & Consider words as tokens if their frequency count is more than specified threshold (-t)and ignoring single character token(-i)

```python
(base) $ python nblearn.py train/ -v -t 3 -i
Considering words which are frequent than 3
Learning from all of labelled data
**************Modification Enabled**************
Following are singleTokenVocab ignored
{',': 147217, '?': 25178, 'a': 44593, "'": 25648, '*': 12639, ';': 12860, '.': 225319, '!': 21735, ':': 93593, '/': 103473, '%': 6028, '_': 25837, '-': 228474, '[': 2457, ']': 2461, '@': 30085, 's': 14894, '$': 14646, '(': 22676, ')': 23927, 'm': 3762, '1': 10491, '7': 3953, 'e': 7013, '+': 3705, '8': 3503, '&': 3882, 'k': 996, 'i': 29339, 'v': 1298, '0': 6373, '6': 3524, '9': 2825, 't': 5131, 'r': 2127, '\\': 4406, 'o': 3089, '2': 9087, '5': 5656, '4': 5423, '3': 9262, 'p': 3289, 'g': 1583, '=': 13708, 'y': 661, 'd': 4270, 'c': 2348, 'q': 355, 'b': 2260, 'f': 1080, 'u': 1941, 'x': 2565, 'n': 1319, 'l': 2363, 'z': 278, '#': 4753, '>': 30456, 'h': 1140, '©': 67, 'w': 1288, '"': 15388, '|': 7247, '`': 1186, '^': 290, 'j': 5225, '{': 217, '}': 256, '·': 476, '±': 578, '½': 471, '¬': 161, '³': 173, '°': 124, '»': 275, 'µ': 67, '«': 24, '\xad': 154, '£': 130, '²': 56, '¦': 68, '§': 69, '~': 991, '\x11': 1, '\x16': 2, '\x14': 13, '\x12': 36, '\x13': 4, '¨': 60, '\x9c': 22, '\x9d': 9, '\x00': 4, '\x05': 11, '\x1b': 41, '\x8e': 72, '\x9e': 2, '\x81': 222, '\x8c': 90, '\x8d': 71, '\x9a': 11, '\x88': 8, '<': 101, '\x01': 1326, '\x07': 11, '\x0f': 26, '\x9f': 6, '\x0e': 0, '\x98': 4, '¶': 21, '\x19': 18, '\x8a': 9, '\x90': 0, '\x10': 6, '\x17': 0, '\x93': 67, '\x94': 72, '\x92': 65, '\x96': 4, '®': 53, 'è': 2, '\x99': 44, 'é': 0, '¡': 26, '¯': 8, '´': 1, '\x91': 1, 'â': 0, '\x95': 30, 'à': 0, '\x80': 0, '\x03': 0, '\x08': 0, '\x02': 13, '\x15': 5, '\x06': 28}
Before:
reset vocabulary and bagofWords to meet threshold limits : 3
len(bagOfWords[SPAM_CATEGORY].keys()) :  85666
len(bagOfWords[HAM_CATEGORY].keys()) :  40226
len(vocabulary) :  107873
After:
len(bagOfWords[SPAM_CATEGORY].keys()) :  34172
len(bagOfWords[HAM_CATEGORY].keys()) :  19162
len(vocabulary) :  1
vocabulary size :  1
total_no_of_emails :  17029
no_of_spam_emails :  7496
no_of_ham_emails :  9533
logPrior
{'ham': -0.836991516093823, 'spam': -1.1838008595021385}
(base) $ python nbclassify.py dev/ -e -d -v 
actual :
{'spam': 3675, 'ham': 1500}
prediction :
{'spam': 0, 'ham': 5175}
correctClassification :
{'spam': 0, 'ham': 1500}
inCorrectClassification :
{'spam': 3675, 'ham': 0}
Discrepancies found in : nbDiscrepancies.txt
****************Evaluation****************
precision :
{'spam': 0, 'ham': 0.29}
recall :
{'spam': 0.0, 'ham': 1.0}
f1_score :
{'spam': 0, 'ham': 0.45}
weighted_avg :  0.13
(base) $
```