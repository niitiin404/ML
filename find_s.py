import pandas as pd 

data = pd.read_csv("training_data.csv")

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

def hypo(x,y):
    sh = None

    for i in range(len(y)):
        if y[i].lower() == "yes":
            if sh is None:
                sh = x[i].copy()
                print("Intial Hypo :",sh)
            else:
                print(f" step {i} : Before Hypo :",sh)
                sh = [ h if h == x else "?" for h,x in zip(sh,x[i])]
                print(f" step {i+1} : After Hypo :",sh)
    print("Result Hypo :",sh)
hypo(x,y)
