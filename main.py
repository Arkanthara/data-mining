import numpy as np
import naive_bayes as bayes


data =np.genfromtxt("iris.csv", delimiter=',', dtype=str)
header = data[0]
features = data[1:, :-1]
labels = data[1:, -1]
model = bayes.Bayes()

model.fit(features, labels)
print(model.prior_y)
print(model.predict(features) - model.string_to_num(labels))
