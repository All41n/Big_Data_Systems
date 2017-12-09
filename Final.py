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
print("\n")
print("\t\t\tCorrelation(Lab 1/Christmas): ",correlation(lab_1,christmas_test))
print("\t\t\tCorrelation(Lab 1/Easter): ",correlation(lab_1,easter_test))
print("\t\t\tCorrelation(Exam/Lab 1): ",correlation(exam_grades,lab_1))
print("\t\t\tCorrelation(Lab 1/Lab 2): ",correlation(lab_1,lab_2))
print("\t\t\tCorrelation(Lab 1/Lab 3): ",correlation(lab_1,lab_3))
print("\n")
print("\t\t\tCorrelation(Lab 2/Christmas): ",correlation(lab_2,christmas_test))
print("\t\t\tCorrelation(Lab 2/Easter): ",correlation(lab_2,easter_test))
print("\t\t\tCorrelation(Lab 2/Lab 1): ",correlation(lab_2,lab_1))
print("\t\t\tCorrelation(Exam/Lab 2): ",correlation(lab_2,exam_grades))
print("\t\t\tCorrelation(Lab 2/Lab 3): ",correlation(lab_2,lab_3))
print("\n")
print("\t\t\tCorrelation(Lab 3/Christmas): ",correlation(lab_3,christmas_test))
print("\t\t\tCorrelation(Lab 3/Easter): ",correlation(lab_3,easter_test))
print("\t\t\tCorrelation(Lab 3/Lab 1): ",correlation(lab_3,lab_1))
print("\t\t\tCorrelation(Lab 3/Lab 2): ",correlation(lab_3,lab_2))
print("\t\t\tCorrelation(Exam/Lab 3): ",correlation(exam_grades,lab_3))
print("___________________________________________________________________________________________________________")
print("________________________________Covariance of eater and exam sets__________________________________________")
print("\t\t\tCoveriance: ",covariance(easter_test,exam_grades))
print("___________________________________________________________________________________________________________")
print("________________________________Simple Linear Regression of eater and exam sets____________________________")
def predict(alpha, beta, x):
    return beta * x + alpha
	
def error(alpha, beta, x, y):
    return y - predict(alpha, beta, x)
	
def sum_of_squared_errors1(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))
	
def least_squares_fit(x,y):
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta
	
def total_sum_of_squares(y):
    return sum(v ** 2 for v in de_mean(y))
	
def r_squared(alpha, beta, x, y):
    return 1.0 - (sum_of_squared_errors1(alpha, beta, x, y) / total_sum_of_squares(y))
	
def squared_error1(x_i, y_i, theta):
    alpha, beta = theta
    return error(alpha, beta, x_i, y_i) ** 2
	
def squared_error_gradient(x_i, y_i, theta):
    alpha, beta = theta
    return [-2 * error(alpha, beta, x_i, y_i),-2 * error(alpha, beta, x_i, y_i) * x_i] 
	
def coefficient(x,y):
	b1 = covariance(x,y)/ variance(x)
	b0 = mean(y) - b1 * mean(x)
	return b1,b0
    
easter_x, exam_x = least_squares_fit(easter_test,exam_grades)
print("\t\t\tEaster_X: ",easter_x)	
print("\t\t\tExam_X: ",exam_x)
print("\t\t\tr-squared:", r_squared(easter_x, exam_x, easter_test, exam_grades))
print("\t\t\tBest Line Fit(y = m* x + c):", coefficient(easter_test,exam_grades))

plt.figure
plt.scatter(easter_test,exam_grades, color = 'k', alpha = 0.5)
plt.plot([easter_test,exam_grades],[easter_test,exam_grades],'k-', color = 'r')
plt.xlabel("Easter Test")
plt.ylabel("Exam Grades")
plt.title("Simple Linear")
plt.show()

plt.figure
plt.scatter(christmas_test,easter_test, color = 'k', alpha = 0.5)
plt.plot([christmas_test,easter_test],[christmas_test,easter_test],'k-', color = 'r')
plt.xlabel("Christmas Test")
plt.ylabel("Easter Grades")
plt.title("Simple Linear")
plt.show()
print("___________________________________________________________________________________________________________")
print("________________________________Multiple Linear Regression of eater and exam sets__________________________")
def sum_of_squares(y):
	return sum([y_i**2 for y_i in y])

def partial_difference_quotient_s(f,v,i,x,y,h):
    w = [v_j +(h if j == i else 0) for j,v_j in enumerate(v)]
    return (f(w,x,y) - f(v,x,y))/h
	
def estimate_gradient_s(f,v,x,y,h =0.001):
    return [partial_difference_quotient_s(f,v,i,x,y,h) for i, _ in enumerate(v)]
	
def step(v,direction,step_size):
    return [v_i - step_size * direction_i for v_i,direction_i in zip(v,direction)]

def predict_y(x_i, beta):
	return numpy.dot(x_i, beta)
	
def error(v,x,y):
	error = y-predict_y(v,x)
	return error
    
def squared_error(v,x,y):
	return error(v,x,y)**2

def multiple_r_squared(v, x, y):
    sum_of_squared_errors = sum(squared_error(v, x_i, y_i) for x_i, y_i in zip(x, y))
    return 1.0 - sum_of_squared_errors / sum_of_squares(de_mean(y))	
	
def normal_cdf(x, mu=0, sigma=1):
	return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)