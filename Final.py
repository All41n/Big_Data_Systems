import csv
import random as random
import math
import numpy as numpy
from numpy import dot
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

lab_1 = []
lab_2 = []
lab_3 = []
easter_test = []
christmas_test = []
exam_grades = []
part_time = []

#reference(https://stackoverflow.com/questions/27416296/how-to-push-a-csv-data-to-mongodb-using-python)
#reference(https://pythonprogramming.net/reading-csv-files-python-3/)
def csv_reader(file):
	reader = csv.DictReader(file, delimiter=";")
	
	for line in reader:
		lab_1.append(int(line["Lab 1"])),
		lab_2.append(int(line["Lab 2"])),
		lab_3.append(int(line["Lab3"])),
		easter_test.append(int(line["Easter Test"])),
		christmas_test.append(int(line["Christmas Test"])),
		exam_grades.append(int(line["Exam Grade"])),
		part_time.append(int(line["parttimejob"]))
		
if __name__ == "__main__":
	with open("C:\\Users\\Allai\\OneDrive\\Documents\\Big_Data_Systems\\Fina_CA\\FinalProjectData1718.csv") as f_obj:
		csv_reader(f_obj)	

zippedLists = [list(i) for i in zip(lab_1,lab_2,lab_3,easter_test,christmas_test,part_time)]

for i in  range(0, len(zippedLists)):
	zippedLists[i].insert(0,1)
print(zippedLists)

zippedLists2 = [list(i) for i in zip(lab_1,lab_2,lab_3,easter_test,christmas_test,part_time,exam_grades)]
unzippedLists = [list(i) for i in zip(*zippedLists2)]
#Functions
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
	
def sum_of_squares1(v):
    return dot(v, v)

def mean(x):
    return sum(x) / len(x)
	
def de_mean(x):
    return [x_i - mean(x) for x_i in x]
	
def variance(x):
    return sum_of_squares1(de_mean(x)) / (len(x) - 1)

def standard_deviation(x):
    return math.sqrt(variance(x))

def covariance(x, y):
    return dot(de_mean(x), de_mean(y)) / (len(x) - 1)

def correlation(x, y):
    if standard_deviation(x) > 0 and standard_deviation(y) > 0:
        return covariance(x, y) / standard_deviation(x) / standard_deviation(y)
    else:
        return 0
print("")
print("________________________________Correlatio of all the data sets__________________________________________")
print("\t\t\tCorrelation(Easter/Exam): ",correlation(easter_test,exam_grades))
print("\t\t\tCorrelation(Easter/Christmas): ",correlation(easter_test,christmas_test))
print("\t\t\tCorrelation(Easter/Lab 1): ",correlation(easter_test,lab_1))
print("\t\t\tCorrelation(Easter/Lab2): ",correlation(easter_test,lab_2))
print("\t\t\tCorrelation(Easter/Lab 3): ",correlation(easter_test,lab_3))
print("\n")
print("\t\t\tCorrelation(Christmas/Exam): ",correlation(christmas_test,exam_grades))
print("\t\t\tCorrelation(Christmas/Easter): ",correlation(christmas_test,easter_test))
print("\t\t\tCorrelation(Christmas/Lab 1): ",correlation(christmas_test,lab_1))
print("\t\t\tCorrelation(Christmas/Lab 2): ",correlation(christmas_test,lab_2))
print("\t\t\tCorrelation(Christmas/Lab 3): ",correlation(christmas_test,lab_3))
print("\n")
print("\t\t\tCorrelation(Exam/Christmas): ",correlation(exam_grades,christmas_test))
print("\t\t\tCorrelation(Exam/Easter): ",correlation(exam_grades,easter_test))
print("\t\t\tCorrelation(Exam/Lab 1): ",correlation(exam_grades,lab_1))
print("\t\t\tCorrelation(Exam/Lab 2): ",correlation(exam_grades,lab_2))
print("\t\t\tCorrelation(Exam/Lab 3): ",correlation(exam_grades,lab_3))
print("\n")