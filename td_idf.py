import math
import random
import re

import findspark
import nltk
from nltk.corpus import stopwords
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

findspark.init()

conf = SparkConf().setAppName("TF-IDF").set("spark.dynamicAllocation.enabled", "true")  # Set Spark configuration
try:
    sc = SparkContext(conf=conf)
except:
    sc.stop()
    sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
sql = SQLContext(sc)

docs_path = "C:/stories/test"  # Path to the data
textFiles = sc.wholeTextFiles(docs_path)  # (path_doc_name, content)
num_docs = textFiles.count()

# Get the list of stop word.
try:
    stops = set(stopwords.words('english'))
except:
    nltk.download('popular')
    stops = set(stopwords.words('english'))


def delete_stop_word(word: str):
    global stops
    word = word.replace("\n", "") \
        .replace("\r", "") \
        .lower()
    if word in stops:
        return ""
    else:
        return word


def list_with_out_stop_word(line):
    file_name = line[0]
    words = line[1]
    with_out_stop_words = []
    for word in words:
        if delete_stop_word(word) != "":
            with_out_stop_words.append(word)
    return file_name, with_out_stop_words


def fix_text(text):
    regex = re.compile('[^a-zA-Z ]')
    text = text.replace("\t", " ") \
        .replace("\n", " ") \
        .replace("\r", " ") \
        .lower()
    return regex.sub("", text)


def calculate_tf_word(tf):
    if not tf == 0:
        tf = 1.0 + math.log10(tf)
    return tf


def count_word(words, text_file_list, file_name: str):
    words_count = []
    for word in words:
        words_count.append((word, (file_name, calculate_tf_word(text_file_list.count(word)))))
    return words_count


def calculate_tf(textFile, words):
    return count_word(words, textFile[1], textFile[0])


def count_files(line):
    return line[0], len(line[1])


def calculate_df(line):
    global num_docs
    count = (float(num_docs) / float(line[1]))
    df = math.log10(count)
    return line[0], df


def calculate_tf_idf(line):
    df = float(line[1][0])
    tf = float(line[1][1][1])
    tf_idf_calculate = df * tf
    return line[1][1][0], (line[0], tf_idf_calculate)


def normalize_vector(line):
    doc_name = line[0]
    vector = line[1]
    calculated_sum = 0.0
    for cell in range(len(vector)):
        calculated_sum = calculated_sum + vector[cell] * vector[cell]
    normal = math.sqrt(calculated_sum)
    for cell in range(len(vector)):
        vector[cell] = vector[cell] / normal
    return doc_name, vector


def calculate_distance(vector1, vector2):
    distance = 0.0
    for cell in range(len(vector2)):
        distance += vector1[cell] * vector2[cell]
    return distance


def calculate_distance_best(line):
    res = []
    for (key, value) in broadcast_tf_idf.value:
        if key != line[0]:
            res.append((key, calculate_distance(line[1], value)))
    res = sorted(res, key=lambda x: -x[1])
    return line[0], list(res)[:5]


def clean_dup_tuple(line):
    name1 = line[0]
    name2 = line[1][0]
    return name1 < name2


# RDD contains name of file and content (as a list of words) (fileName, list(word)) and filtering stopwords
text_files = textFiles.map(lambda docs: (docs[0].split("/")[-1], fix_text(docs[1]).split(" "))).map(
    list_with_out_stop_word).cache()  # we can take the path as name

# RDD with all the words (every row is a list of words from that story)(list(word))
bag_of_words = text_files.map(lambda docs: docs[1])

# RDD with all the words (every line is one word)
bag_of_words = bag_of_words.flatMap(lambda x: x).cache().filter(lambda x: x != '').distinct()

numOfWords = bag_of_words.count()  # amount of words in all files

# Inverted index
# Every row will be word and the files it appears in
text_files_exploded = text_files.flatMapValues(lambda x: x).map(lambda x: (x[1], x[0]))  # (word, file)
# We apply distinct so if word appears in one file more than once it will count only 1 time
# Then we group by word and count the number of files it appears in
inverted_index = text_files_exploded.distinct().groupByKey().map(
    lambda x: (x[0], list(x[1]))).cache()  # (word, list(file))
inverted_index.take(50)

# Convert bagOfWords to broadcast variable to be more efficient
bag_of_words_readOnly = sc.broadcast(bag_of_words.collect())

TFVectors = text_files.map(lambda x: calculate_tf(x, bag_of_words_readOnly.value)).flatMap(
    lambda x: x)  # (word, (file, TF))

# Counting the files that each word appears in and calculate DF fo this value
DF = inverted_index.map(count_files).map(calculate_df).cache()
DF.take(100)

DF_join_TF = DF.join(TFVectors)  # (word,(df,(fileName, tf)) )
DF_join_TF.take(50)

Pre_TF_IDF = DF_join_TF.map(calculate_tf_idf)  # (file, (word, tf-idf))
Pre_TF_IDF.take(10)

# We group by the file and get the tf-idf table
tf_idf = Pre_TF_IDF.map(lambda x: (x[0], x[1][1])).groupByKey().map(lambda x: (x[0], list(x[1])))  # (file list(tfidf))
tf_idf.take(50)

# Normalize the vector
tf_idf_normalized = tf_idf.map(normalize_vector).cache()
tf_idf_normalized.take(100)

broadcast_tf_idf = sc.broadcast(tf_idf_normalized.collect())

best5ForEachFile = tf_idf_normalized.map(calculate_distance_best)
best5ForEachFile.take(23)

best5 = best5ForEachFile.flatMapValues(lambda x: x).filter(clean_dup_tuple).sortBy(lambda x: -x[1][1])
best5.take(5)


# find best much by query


def fix_query(query):
    words = fix_text(query).split(" ")
    list_with_out_stop_word_res = []
    for word in words:
        if delete_stop_word(word) != "":
            list_with_out_stop_word_res.append(word)
    return list_with_out_stop_word_res


def calculate_distance(vector1, vector2):
    distance = 0.0
    for cell in range(len(vector2)):
        distance += vector1[cell] * vector2[cell]
    return distance


def calculate_distance_best_query(line):
    global tf_idf_normalized
    res = []

    for (key, value) in broadcast_tf_idf.value:
        if key != line[0]:
            res.append((key, calculate_distance(line[1], value)))

    res = sorted(res, key=lambda x: -x[1])

    return line[0], list(res)[:10]


def search(query):
    global tf_idf_normalized
    global bag_of_words_readOnly
    global DF
    query_words = fix_query(query)  # Delete all chars that are not a-z OR A-Z and stopwords
    tf_rdd = sc.parallelize(
        count_word(bag_of_words_readOnly.value, query_words, "query"))  # RDD - (word, ('query' , TF)))
    df_join_tf_query = DF.join(tf_rdd)  # (word, (DF,('query' , TF)))
    pre_tf_idf_query = df_join_tf_query.map(calculate_tf_idf)  # ('query', (word, tf-df))
    tf_idf_query = pre_tf_idf_query.map(lambda x: (x[0], x[1][1])).groupByKey().map(lambda x: (x[0], list(x[1])))
    # ('query', list(tf-idf))
    tf_idf_query_normalized = tf_idf_query.map(normalize_vector)  # ('query', normalize list(tf-idf))
    closest_files = tf_idf_query_normalized.map(calculate_distance_best_query).sortByKey(
        False)  # ((file,vec),('query',vec))
    return closest_files


def get_search(query, tries):
    try:
        if tries < 10:
            return search(query)
    except:
        tries = tries + 1
        return get_search(query, tries)


def get_rdd_res(rdd, tries):
    try:
        if tries < 10:
            return rdd.take(10)
    except:
        tries = tries + 1
        return get_rdd_res(rdd, tries)


test = get_search("write your query", 0)
print(get_rdd_res(test, 0))

# KNN
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


# Generating centers
centers = []
for i in range(numOfCenters):
    centers.append(normalizevec([random.random() for _ in range(numOfWords)]))
print(centers)


def calculate_vectors_distance(p, center):
    sumVector = 0.0
    for i in range(len(p)):
        sumVector = sumVector + (p[i] - center[i]) ** 2
    return sumVector


def find_closest_point(p, i_centers):
    best_index = 0
    closest_vector = float("+inf")
    for cell in range(len(i_centers)):
        temp_dist1 = calculate_vectors_distance(p, i_centers[cell])
        if temp_dist1 < closest_vector:
            closest_vector = temp_dist1
            best_index = cell
    return best_index


def sum_vectors_and_count(x, y):
    return [m + n for m, n in zip(x[0], y[0])], x[1] + y[1]


def calculate_new_center(s):
    return s[0], [float(m / s[1][1]) for m in s[1][0]]


def calculate_distance_between_centers(new_points, i_centers):
    res = 0
    for (center_number, center_vector) in new_points:
        res = res + calculate_vectors_distance(i_centers[center_number], center_vector)
    return res


while tempDist > converge:
    closest = vectors.map(lambda x: (find_closest_point(x, centers), (x, 1)))  # (Index of closest center, (vector, 1))
    pointStats = closest.reduceByKey(sum_vectors_and_count)
    # (Index of center, (sum of all vector related to center, num of vectors we sum))
    newPoints = pointStats.map(calculate_new_center).collect()
    # [(Index of center, vector center)]
    tempDist = calculate_distance_between_centers(newPoints, centers)
    # num - The Distance Between new and old Centers

    # Replace Old centers
    for (centerNumber, centerVector) in newPoints:
        centers[centerNumber] = centerVector


def find_closest_file(line, i_centers):
    best_index = 0
    p = line[1]
    closest_vector = float("+inf")
    for i in range(len(i_centers)):
        temp_dist1 = calculate_vectors_distance(p, i_centers[i])
        if temp_dist1 < closest_vector:
            closest_vector = temp_dist1
            best_index = i
    return best_index


# tf_idf_normalized #(File, Vector)
# tf_idf_normalized.take(10)
# NOT OF THE algorithm JUST TO SEE THE RESULT
closestFile = tf_idf_normalized.map(
    lambda x: (find_closest_file(x, centers), x[0]))  # (Index of closest center, fileName)
closestFileByIndex = closestFile.groupByKey().map(lambda x: (x[0], list(x[1])))
closestFileByIndex.take(4)
