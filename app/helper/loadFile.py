import numpy as np
import xlrd

# Load xls sheet with data
doc = xlrd.open_workbook('app/Data/SAheart.xls').sheet_by_index(0)

# Extract attribute names (1st row, column 4 to 12)
attributeNames = doc.row_values(0, 1, 11)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(10, 1, 463)
classLabelsInt = [int(num) for num in classLabels]
classNames = sorted(set([0, 1]))#[str(int(num)) for num in classLabels] sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))

binaryDict = dict(zip(['Absent', 'Present'], range(2)))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabelsInt])

# Preallocate memory, then extract excel data to matrix X
X = np.empty((462, 10))
for i, col_id in enumerate(range(1, 11)):
    # If col is famhist (binary categorical), encode as 0,1
    if col_id == 5:
        X[:, i] = np.asarray([binaryDict[value] for value in doc.col_values(col_id, 1, 463)])
    # otherwise (real valued attributes) keep as is
    else:
        X[:, i] = np.asarray(doc.col_values(col_id, 1, 463))

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

print(classNames)

print("loaded file correctly")