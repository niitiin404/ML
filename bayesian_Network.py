
import pandas as ps
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart (1).csv")
#data.info()

x = data.drop(columns="target")
y = data["target"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=21)

model = GaussianNB().fit(x_train,y_train)

y_pred = model.predict(x_test)

acc = model.score(x_test,y_test)*100
print(acc)


sample = [[43,1,0,150,247,0,1,171,0,1.5,2,2,1]]


pred = model.predict(sample)
print(f"Predicted heart Desicies : { 'yes' if pred[0] == 1 else 'No'}")
