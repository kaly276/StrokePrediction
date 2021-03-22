from etl_job import extract
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

train, test = train_test_split(data, test_size=0.25, random_state=42)

display(train)
X_train = ...
y_train = train['stroke']

X_test = ...
y_test = test['stroke']