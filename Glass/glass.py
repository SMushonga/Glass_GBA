from pylab import *
from urllib import request
import matplotlib.pyplot as plot
import numpy
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
target_url = ("http://archive.ics.uci.edu/ml/machine-learning-"
"databases/abalone/abalone.data")
#Learning and evaluating Gradient Boost
data = request.urlopen(target_url)
xList = []
labels = []
for line in data:
    line = line.decode()
    #split on semi-colon
    row = line.strip().split(",")
    #put labels in separate array and remove label from row
    labels.append(float(row.pop()))
    #form list of list of attributes (all strings)
    xList.append(row)
#code three-valued sex attribute as numeric
xCoded = []
for row in xList:
    #first code the three-valued sex variable
    codedSex = [0.0, 0.0]
    if row[0] == 'M': codedSex[0] = 1.0
    if row[0] == 'F': codedSex[1] = 1.0
    numRow = [float(row[i]) for i in range(1,len(row))]
    rowCoded = list(codedSex) + numRow
    xCoded.append(rowCoded)
#list of names for
abaloneNames = numpy.array(['Sex1', 'Sex2', 'Length', 'Diameter',
    'Height', 'Whole weight', 'Shucked weight',
    'Viscera weight', 'Shell weight', 'Rings'])
#number of rows and columns in x matrix
nrows = len(xCoded)
ncols = len(xCoded[1])
#form x and y into numpy arrays and make up column names
X = numpy.array(xCoded)
y = numpy.array(labels)
#break into training and test sets.
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30,
    random_state=531)
#instantiate model
nEst = 2000
depth = 5
learnRate = 0.005
maxFeatures = 3
subsamp = 0.5
abaloneGBMModel = ensemble.GradientBoostingRegressor(n_estimators=nEst,max_depth=depth, learning_rate=learnRate,max_features=maxFeatures,subsample=subsamp,loss='squared_error')
#train
abaloneGBMModel.fit(xTrain, yTrain)
# compute mse on test set
msError = []
predictions = abaloneGBMModel.staged_predict(xTest)
for p in predictions:
    msError.append(mean_squared_error(yTest, p))
print("MSE" )
print(min(msError))
print(msError.index(min(msError)))
#plot training and test errors vs number of trees in ensemble
plot.figure()
plot.plot(range(1, nEst + 1), abaloneGBMModel.train_score_,
    label='Training Set MSE', linestyle=":")
plot.plot(range(1, nEst + 1), msError, label='Test Set MSE')
plot.legend(loc='upper right')
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Mean Squared Error')
plot.show()
