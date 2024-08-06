import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data
data_dict = pickle.load(open("leaf_data_vgg16.pickle", "rb"))
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

# Feature Scaling
# scaler = StandardScaler()
# data = scaler.fit_transform(data)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# Define the parameter distribution for Randomized Search
# param_dist = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_depth': [10, 20, 30, 40, 50, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }

# Perform randomized search with cross-validation
# random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', verbose=1, n_jobs=-1, random_state=42)
# random_search.fit(x_train, y_train)
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
# Best parameters and model from Randomized Search
# print("Best parameters found: ", random_search.best_params_)
# best_model = random_search.best_estimator_

# # Make predictions with the best model
# y_pred = best_model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest accuracy :", accuracy)

f=open("model.pickle","wb")
pickle.dump({"model":model},f)
f.close()
print("dumped successfully")