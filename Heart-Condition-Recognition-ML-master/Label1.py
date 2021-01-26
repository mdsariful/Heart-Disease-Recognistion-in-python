import tkinter as tk
from tkinter.ttk import *
import tkinter.ttk as ttk
import pandas as ps 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


ch = int()

def click():
	i = list()
	a = age.get()
	a = int(a)
	i.append(a)
	if(sex.get()=="Male"):
		s=0
	else:
		s=1
	i.append(s)
	i.append(cpt.get())
	b = bp.get()
	b = int(b)
	i.append(b)
	c = chol.get()
	c = int(c)
	i.append(c)
	if(fbs.get()=="True"):
		f=1
	else:
		f=0
	i.append(f)
	if(recg.get()=="Normal"):
		r=0
	elif(recg.get()=="Abnormal"):
		r=1
	else:
		r=2
	i.append(r)
	m = mhr.get()
	m = int(m)
	i.append(m)
	if(ia.get()=="True"):
		ag = 1
	else:
		ag = 0
	i.append(ag)
	o = op.get()
	o = float(o)
	i.append(o)
	if(slop.get()=="Up"):
		sl = 0
	elif(slop.get()=="Flat"):
		sl = 1
	else:
		sl = 2
	i.append(sl)
	i.append(ncv.get())
	i.append(thal.get())
	inp = np.array(i, ndmin=2, dtype=float)
	Label(frame2, text=i).grid()
	if(model.get()=="SVM"):
		pred = svm.predict(inp)
	elif(model.get()=="LogisticRegression"):
		pred = lr.predict(inp)
	elif(model.get()=="KNeighborsClassifier"):
		pred = knn.predict(inp)
	elif(model.get()=="DecisionTreeClassifier"):
		pred = tree.predict(inp)
	elif(model.get()=="Naive Bayes"):
		pred = nb.predict(inp)
	else:
		from keras.models import load_model
		m = load_model('model.h5')
		pred = m.predict(inp)
		if(pred>0.5):
			pred=1
		else:
			pred=0
	if(pred==1):
		Label(frame2, text="Your Heart is not in normal Condition consult doctor").grid()
	else:
		Label(frame2, text="Your Heart is in normal Condition").grid()





####################################
root = tk.Tk()
root.title("Heart Condition Recogonition")
root.geometry('525x450')
main = Frame(root)
main.grid()

Label(main, text="Heart Condition Recogonition",  font=("times",25)).grid()



frame = Frame(root)
frame1 = Frame(root)
frame2 = Frame(root)
frame.grid()
frame1.grid()
frame2.grid()

Label(frame, text="Age").grid(row=3, column=1)
age= Entry(frame, width=23)
age.grid(row=3, column=2)

Label(frame, text="Sex").grid(row=3, column=3)
sex = Combobox(frame)
sex['values'] = ("Male","Female")
sex.current(0)
sex.grid(row=3, column=4)

Label(frame, text="Chest pain type").grid(row=5, column=1)
cpt = Combobox(frame)
cpt['values'] = (0,1,2,3)
cpt.current(0)
cpt.grid(row=5, column=2)

Label(frame, text="Blood Pressure").grid(row=5, column=3)
bp= Entry(frame, width=23)
bp.grid(row=5, column=4)

Label(frame, text="Cholestrol").grid(row=7, column=1)
chol= Entry(frame, width=23)
chol.grid(row=7, column=2)

Label(frame, text="Fasting Blood Sugar").grid(row=7, column=3)
fbs = Combobox(frame)
fbs['values'] = ("True","False")
fbs.current(0)
fbs.grid(row=7, column=4)

Label(frame, text="Resting ECG").grid(row=9, column=1)
recg = Combobox(frame)
recg['values'] = ("Normal", "Abnormal", "Hyper")
recg.current(0)
recg.grid(row=9, column=2)

Label(frame, text="Maximum Heart Rate").grid(row=9, column=3)
mhr = Entry(frame, width=23)
mhr.grid(row=9, column=4)

Label(frame, text="Induced Again").grid(row=10, column=1)
ia = Combobox(frame)
ia['values'] = ("True", "False")
ia.current(0)
ia.grid(row=10, column=2)

Label(frame, text="Old Peak").grid(row=10, column=3)
op = Entry(frame, width=23)
op.grid(row=10, column=4)

Label(frame, text="Slop").grid(row=11, column=1)
slop = Combobox(frame)
slop['values'] = ("Up", "Flat", "Down")
slop.current(0)
slop.grid(row=11, column=2)

Label(frame, text="Number of Colored Vessel").grid(row=11, column=3)
ncv = Combobox(frame)
ncv['values'] = (0,1,2,3,4)
ncv.current(0)
ncv.grid(row=11, column=4)

Label(frame, text="Thal").grid(row=12, column=1)
thal = Combobox(frame)
thal['values'] = (0,1,2,3)
thal.current(0)
thal.grid(row=12,column=2)


data = ps.read_excel("train3.xlsx")
x = data.drop("Class", axis=1)
y = data.drop(["Age","Sex","Chest pain Type","Blood pressure","Cholesterol","Fasting Blood Sugar","Resting ECG","maximum Heart Rate","Induced Angina","Old Peak","Slop","Number Colored Vessel","Thal"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=30)
scoring='accuracy_score'

Label(frame1,text="Model and Accuracy").grid(row=1,column=1)
from sklearn import svm

svm = svm.SVC()
svm.fit(x_train,y_train)
p=svm.predict(x_test)
acc = (accuracy_score(y_test,p))

Label(frame1,text="SVM Accuracy :").grid(row=2, column=0)
Label(frame1,text=acc).grid(row=2, column=1)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train,y_train)
p2=lr.predict(x_test)
acc = accuracy_score(y_test,p2)

Label(frame1, text="LR Accuracy : ").grid(row=15, column=0)
Label(frame1, text=acc).grid(row=15, column=1)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
p = knn.predict(x_test)
acc = accuracy_score(y_test,p)

Label(frame1, text="KNN Accuracy :").grid(row=16, column=0)
Label(frame1, text=acc).grid(row=16, column=1)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(x_train,y_train)
p = tree.predict(x_test)
acc = accuracy_score(y_test,p)

Label(frame1, text="Decision Tree :").grid(row=17, column=0)
Label(frame1, text=acc).grid(row=17, column=1)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)
p = nb.predict(x_test)
acc = accuracy_score(y_test,p)

Label(frame1, text="Naive Bayes :").grid(row=18, column=0)
Label(frame1, text=acc).grid(row=18, column=1)
Label(frame1, text="Keras Model :").grid(row=19, column=0)
Label(frame1, text="0.8647").grid(row=19, column=1)
model = Combobox(frame1)
model['values'] = ("SVM","LogisticRegression","KNeighborsClassifier","DecisionTreeClassifier","Naive Bayes","Keras Model")
model.current(0)
model.grid(row=20,column=1)
Button(frame1, text='Predict', command=click).grid(row=20,column=2)

root.mainloop()
