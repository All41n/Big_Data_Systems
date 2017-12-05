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
