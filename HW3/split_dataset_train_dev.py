#$python split_dataset_train_dev.py labeled\ data 75 25
#total_no_of_files : 1076
#train_no_of_files : 807
#dev_no_of_files : 269
#Copying files into train and dev folders...
#TrainingData Dir : labeled data_807 No Of Files : 807
#DevelopmentData Dir : labeled data_269 No Of Files : 269

#$python split_dataset_train_dev.py labeled\ data 93 7
#total_no_of_files : 1076
#train_no_of_files : 1000
#dev_no_of_files : 76
#Copying files into train and dev folders...
#TrainingData Dir : labeled data_1000 No Of Files : 1000
#DevelopmentData Dir : labeled data_76 No Of Files : 76

import argparse, os
import random
import shutil

def rscandir2(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                yield (root, file)
                
def splitDataSet(dataDirPath, train_percentage, dev_percentage):
    #no_of_training_files = 
    # count no of files
    total_no_of_files = 0
    allFilePaths = []
    for path,fileName in rscandir2(dataDirPath):
        total_no_of_files += 1
        filePath = os.path.join(path, fileName)
        allFilePaths.append(filePath)
    train_no_of_files = int(total_no_of_files*train_percentage/100)
    dev_no_of_files = total_no_of_files - train_no_of_files
    print("total_no_of_files : {}".format(total_no_of_files))
    print("train_no_of_files : {}".format(train_no_of_files))
    print("dev_no_of_files : {}".format(dev_no_of_files))
    dataDirName = os.path.basename(dataDirPath)
    trainDirName = dataDirName + '_' + str(train_no_of_files)
    devDirName = dataDirName + '_' + str(dev_no_of_files)

    if not os.path.exists(trainDirName):
        os.makedirs(trainDirName)
    if not os.path.exists(devDirName):
        os.makedirs(devDirName)
    
    trainFilesSet = set(random.sample(allFilePaths, train_no_of_files))

    print("Copying files into train and dev folders...")
    for filePath in allFilePaths:
        if filePath in trainFilesSet:
            shutil.copy(filePath, trainDirName)
        else:
            shutil.copy(filePath, devDirName)
    print("TrainingData Dir : {} No Of Files : {}".format(trainDirName, len(os.listdir(trainDirName))))
    print("DevelopmentData Dir : {} No Of Files : {}".format(devDirName, len(os.listdir(devDirName))))
    
def main():
    parser = argparse.ArgumentParser(description="partition dataset into train and dev picking files in random")
    parser.add_argument("dataDir", help="path to all labelled data")
    parser.add_argument("train_percentage", help="percentage of training data", type=float)
    parser.add_argument("dev_percentage", help="percentage of development data", type=float)
    args = parser.parse_args()
    
    dataDirPath = args.dataDir
    train_percentage = args.train_percentage
    dev_percentage = args.dev_percentage
    
    if (train_percentage + dev_percentage) != 100:
        print("Invalid split of dataset, make sure to percentages add upto 100")
        exit()
    
    splitDataSet(dataDirPath, train_percentage, dev_percentage)
    
#split_dataset_train_dev.py labelled\ data 75 25
#split_dataset_train_dev.py labelled\ data 93 7
if __name__ == "__main__" : main()
