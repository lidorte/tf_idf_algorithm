import findspark
import math
import re
import nltk

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from nltk.corpus import stopwords

findspark.init()

conf = SparkConf().setAppName("TF-IDF").set("spark.dynamicAllocation.enabled", "true")  # Set Spark configuration
try:
    sc = SparkContext(conf=conf)
except:
    sc.stop()
    sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
sql = SQLContext(sc)

docs_path = "C:/stories/test"
textFiles = sc.wholeTextFiles(docs_path)  # (path_doc_name, content)
num_docs = textFiles.count()
# textFiles.take(3)

try:
    stops = set(stopwords.words('english'))
except:
    nltk.download('popular')
    stops = set(stopwords.words('english'))


def deleteStopWord(word: str):
    global stops
    word = word.replace("\n", "") \
        .replace("\r", "") \
        .lower()
    if word in stops:
        return ""
    else:
        return word


def listWithOutStopWord(line):
    fileName = line[0]
    words = line[1]
    withOutStopWords = []
    for word in words:
        if deleteStopWord(word) != "":
            withOutStopWords.append(word)
    return (fileName, withOutStopWords)


def fixText(text):
    regex = re.compile('[^a-zA-Z ]')
    text = text.replace("\t", " ") \
        .replace("\n", " ") \
        .replace("\r", " ") \
        .lower()
    return regex.sub("", text)


# RDD contains name of file and content (as a list of words) (fileName, list(word)) and filtring stopwords
text_files = textFiles.map(lambda docs: (docs[0].split("/")[-1], fixText(docs[1]).split(" "))).map(
    listWithOutStopWord).cache()  # we can take the path as name

# RDD with all the words (every row is a list of words from that story)(list(word))
bag_of_words = text_files.map(lambda docs: docs[1])

# RDD with all the words (every line is one word)
bag_of_words = bag_of_words.flatMap(lambda x: x).cache().filter(lambda x: x != '').distinct()

numOfWords = bag_of_words.count()  # amount of words in all files
bag_of_words.take(50)

# Inverted index
# every row will be word and the files it appears in
text_files_exploded = text_files.flatMapValues(lambda x: x).map(lambda x: (x[1], x[0]))  # (word, file)
# we apply distinct so if word appears in one file more than once it will count only 1 time
# then we group by word and count the number of files it appears in
inverted_index = text_files_exploded.distinct().groupByKey().map(
    lambda x: (x[0], list(x[1]))).cache()  # (word, list(file))
inverted_index.take(50)


def calculate_TF_For_Word(tf):
    if not tf == 0:
        tf = 1.0 + math.log10(tf)
    return tf


# Convert bagOfWords to broadcast variable to be more efficient
bag_of_words_readOnly = sc.broadcast(bag_of_words.collect())


def countWord(words, text_file_list, fileName: str):
    wordsCount = []
    for word in words:
        wordsCount.append((word, (fileName, calculate_TF_For_Word(text_file_list.count(word)))))
    return wordsCount


def calculateTF(textFile, words):
    return countWord(words, textFile[1], textFile[0])


# We could make it a table but this way it was easier to create the tfidf table later on
TFVectors = text_files.map(lambda x: calculateTF(x, bag_of_words_readOnly.value)).flatMap(
    lambda x: x)  # (word, (file, TF))
TFVectors.take(50)


def count_files(line):
    return line[0], len(line[1])


def calculate_DF(line):
    global num_docs
    count = (float(num_docs) / float(line[1]))
    df = math.log10(count)
    return (line[0], df)


# Counting the files that each word appers in and calculate DF fo this value
DF = inverted_index.map(count_files).map(calculate_DF).cache()
DF.take(100)

DFjoinTF = DF.join(TFVectors)  # (word,(df,(fileName, tf)) )
DFjoinTF.take(50)


def calculateTFIDF(line):
    df = (float)(line[1][0])
    tf = (float)(line[1][1][1])
    tfIdf = df * tf
    return (line[1][1][0], (line[0], tfIdf))


Pre_TFIDF = DFjoinTF.map(calculateTFIDF)  # (file, (word, tfidf))
Pre_TFIDF.take(10)

# We group by the file and get the tfidf table
tf_idf = Pre_TFIDF.map(lambda x: (x[0], x[1][1])).groupByKey().map(lambda x: (x[0], list(x[1])))  # (file list(tfidf))
tf_idf.take(50)


def normalize(line):
    doc_name = line[0]
    vector = line[1]
    sum = 0.0
    for i in range(len(vector)):
        sum = sum + vector[i] * vector[i]
    normal = math.sqrt(sum)
    for i in range(len(vector)):
        vector[i] = vector[i] / normal
    return (doc_name, vector)


# Normalize the vector
tf_idf_normalized = tf_idf.map(normalize).cache()
tf_idf_normalized.take(100)

broadcat_tf_idf = sc.broadcast(tf_idf_normalized.collect())


def calculate_distance(vector1, vector2):
    distance = 0.0
    for i in range(len(vector2)):
        distance += vector1[i] * vector2[i]

    return distance


def calculate_distance_best(line):
    res = []

    for (key, value) in broadcat_tf_idf.value:
        if key != line[0]:
            res.append((key, calculate_distance(line[1], value)))

    res = sorted(res, key=lambda x: -x[1])

    return line[0], list(res)[:5]


best5ForEachFile = tf_idf_normalized.map(calculate_distance_best)
best5ForEachFile.take(23)


# As expected same brances of sport will provide the same similarity

def cleanDup(line):
    name1 = line[0]
    name2 = line[1][0]
    return name1 < name2


best5 = best5ForEachFile.flatMapValues(lambda x: x).filter(cleanDup).sortBy(lambda x: -x[1][1])
best5.take(5)


def fixQuery(query):
    words = fixText(query).split(" ")
    listWithOutStopWord = []
    for word in words:
        if deleteStopWord(word) != "":
            listWithOutStopWord.append(word)
    return listWithOutStopWord


def calculate_distance(vector1, vector2):
    distance = 0.0
    for i in range(len(vector2)):
        distance += vector1[i] * vector2[i]

    return distance


def calculate_distance_best_query(line):
    global tf_idf_normalized
    res = []

    for (key, value) in broadcat_tf_idf.value:
        if key != line[0]:
            res.append((key, calculate_distance(line[1], value)))

    res = sorted(res, key=lambda x: -x[1])

    return line[0], list(res)[:10]


def search(query):
    global closest_files
    global tf_idf_normalized
    global bag_of_words_readOnly
    global DF
    QueryWords = fixQuery(query)  # Delete all chars that are not a-z OR A-Z and stopwords
    tfRDD = sc.parallelize(countWord(bag_of_words_readOnly.value, QueryWords, "query"))  # RDD - (word, ('query' , TF)))
    DFjoinTFQuery = DF.join(tfRDD)  # (word, (DF,('query' , TF)))
    Pre_TFIDFQuery = DFjoinTFQuery.map(calculateTFIDF)  # ('query', (word, tfidf))
    tf_idfQuery = Pre_TFIDFQuery.map(lambda x: (x[0], x[1][1])).groupByKey().map(lambda x: (x[0], list(x[1])))
    # ('query', list(tfidf))
    tf_idfQuery_normalized = tf_idfQuery.map(normalize)  # ('query', normalize list(tfidf))
    closest_files = tf_idfQuery_normalized.map(calculate_distance_best_query).sortByKey(
        False)  # ((file,vec),('query',vec))
    return closest_files


def getSearch(query, tries):
    try:
        if tries < 10:
            return search(query)
    except:
        tries = tries + 1
        return getSearch(query, tries)


def getRDDRes(rdd, tries):
    try:
        if tries < 10:
            return rdd.take(10)
    except:
        tries = tries + 1
        return getRDDRes(rdd, tries)


test = getSearch("who is the best football player", 0)
print(getRDDRes(test, 0))

### KNN
vectors = tf_idf_normalized.map(lambda x: x[1])  # (list(normalized vector))
vectors.take(10)

numOfCenters = 4
converge = 0.0001
tempDist = 1.0


def normalizevec(vector):
    sum = 0.0
    for i in range(len(vector)):
        sum = sum + vector[i] * vector[i]
    normal = math.sqrt(sum)
    for i in range(len(vector)):
        vector[i] = vector[i] / normal
    return (vector)


import random

# Generating centers
centers = []
for i in range(numOfCenters):
    centers.append(normalizevec([random.random() for _ in range(numOfWords)]))
print(centers)


def calculateVectorsDistance(p, center):
    sumVector = 0.0
    for i in range(len(p)):
        sumVector = sumVector + (p[i] - center[i]) ** 2
    return sumVector


def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist1 = calculateVectorsDistance(p, centers[i])
        if tempDist1 < closest:
            closest = tempDist1
            bestIndex = i
    return bestIndex


def sumVectorsAndCount(x, y):
    return ([m + n for m, n in zip(x[0], y[0])], x[1] + y[1])


def calculateNewCenter(s):
    return (s[0], [(float)(m / s[1][1]) for m in s[1][0]])


def calculateDistanceBetweenCenters(newPoints, centers):
    res = 0
    for (centerNumber, centerVector) in newPoints:
        res = res + calculateVectorsDistance(centers[centerNumber], centerVector)
    return res


while tempDist > converge:
    closest = vectors.map(lambda x: (closestPoint(x, centers), (x, 1)))  # (Index of closeset center, (vector, 1))
    pointStats = closest.reduceByKey(sumVectorsAndCount)
    # (Index of center, (sum of all vector related to center, num of vectors we sum))
    newPoints = pointStats.map(calculateNewCenter).collect()
    # [(Index of center, vector center)]
    tempDist = calculateDistanceBetweenCenters(newPoints, centers)
    # num - The Distance Between new and old Centers

    # Replace Old centers
    for (centerNumber, centerVector) in newPoints:
        centers[centerNumber] = centerVector


def findClosestFile(line, centers):
    bestIndex = 0
    p = line[1]
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist1 = calculateVectorsDistance(p, centers[i])
        if tempDist1 < closest:
            closest = tempDist1
            bestIndex = i
    return bestIndex


# tf_idf_normalized #(File, Vector)
# tf_idf_normalized.take(10)
# NOT OF THE ALGO JUST TO SEE THE RESULT
closestFile = tf_idf_normalized.map(
    lambda x: (findClosestFile(x, centers), x[0]))  # (Index of closeset center, fileName)
closestFileByIndex = closestFile.groupByKey().map(lambda x: (x[0], list(x[1])))
closestFileByIndex.take(4)
