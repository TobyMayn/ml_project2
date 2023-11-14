import numpy as np
import xlrd

# Load xls sheet with data
doc = xlrd.open_workbook('data/SAheart.xls').sheet_by_index(0)

# Extract attribute names (1st row, column 4 to 12)
old_attributeNames = doc.row_values(0, 1, 11)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(10, 1, 463)
classLabelsInt = [int(num) for num in classLabels]
classNames = sorted(set([0, 1]))#[str(int(num)) for num in classLabels] sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))

binaryDict = dict(zip(['Absent', 'Present'], range(2)))

# Preallocate memory, then extract excel data to matrix X
old_X = np.empty((462, 10))
for i, col_id in enumerate(range(1, 11)):
    # If col is famhist (binary categorical), encode as 0,1
    if col_id == 5:
        old_X[:, i] = np.asarray([binaryDict[value] for value in doc.col_values(col_id, 1, 463)])
    # otherwise (real valued attributes) keep as is
    else:
        old_X[:, i] = np.asarray(doc.col_values(col_id, 1, 463))

X = old_X[:, [1,6]]
attributeNames = [doc.cell_value(0, idx) for idx in [2,7]]

# Compute values of N, M and C.

M = len(attributeNames)
C = len(classNames)

print(classNames)

print("loaded file correctly")