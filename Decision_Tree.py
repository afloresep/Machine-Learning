## The usual libraries, loading the dataset and performing the train-test split
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep']=='unacc') #1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:,0:6])
y = df['accep']

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)

## Functions to calculate gini impurity and information gain

def gini(data):
    """calculate the Gini Impurity
    """
    data = pd.Series(data)
    return 1 - sum(data.value_counts(normalize=True)**2)
   
def info_gain(left, right, current_impurity):
    """Information Gain associated with creating a node/split data.
    Input: left, right are data in left branch, right banch, respectively
    current_impurity is the data impurity before splitting into left, right branches
    """
    # weight for gini score of the left branch
    w = float(len(left)) / (len(left) + len(right))
    return current_impurity - w * gini(left) - (1 - w) * gini(right)

#### -----------------------------------
## 1. Calculate sample sizes for a split on `persons_2`
left = y_train[x_train['persons_2']== 0]
right = y_train[x_train['persons_2'] == 1]
len_left = len(left)
len_right = len(right)
print ('No. of cars with persons_2 == 0:', len_left)
print ('No. of cars with persons_2 == 1:', len_right)
# Output == 0 (917); == 1 (465)

"""
Aquí queremos seleccionar las filas de y_train que tengan en la columna ['persons_2] de x_train un valor de 0 o de 1. Para ello usamos x_train['persons_2]==0. Esto devuelve una lista boolean (True si es 0 o False si fuera 1) y así cuando pasamos la lista boolean (True, False, False, True) por y_train[] selecciona sólo las que son True, o en otras palabras, las que tienen 0 en el x_train['persons_2']   
"""

## 2. Gini impurity calculations
gi = gini(y_train)
print(gi)
gini_left = gini(left)
gini_right = gini(right)
print('Original gini impurity (without splitting!):', gi)
print('Left split gini impurity:', gini_left)
print('Right split gini impurity:', gini_right)

## 3.Information gain when using feature `persons_2`
info_gain_persons_2 = info_gain(right, left, gi)
print(f'Information gain for persons_2:', info_gain_persons_2)

## 4. Which feature split maximizes information gain?
info_gain_list = []
for i in x_train.columns:
    left = y_train[x_train[i]==0]
    right = y_train[x_train[i]==1]
    info_gain_list.append([i, info_gain(left, right, gi)])

info_gain_table = pd.DataFrame(info_gain_list).sort_values(1,ascending=False)
print(f'Greatest impurity gain at:{info_gain_table.iloc[0,:]}')
print(info_gain_table)

# Output 
#              0         1
# 19      safety_low  0.091603
# 12       persons_2  0.090135
# 18     safety_high  0.045116
# 14    persons_more  0.025261
# 13       persons_4  0.020254

"""
Information gain is a measure used in decision trees to evaluate the usefulness of a feature for splitting data. It quantifies the reduction in entropy achieved by splitting the data based on that feature. Higher information gain indicates a more informative feature, helping to create more accurate and discriminative decision trees
"""