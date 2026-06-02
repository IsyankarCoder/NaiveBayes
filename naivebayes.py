import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mixed_naive_bayes import MixedNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report

# Veri Okuma
veriSeti = pd.read_csv("bank-full.csv",sep=";")

# Veri Ön İşleme
print(veriSeti.dtypes)
print(veriSeti.head())
print(veriSeti.columns)

# Ardından mixedNB() fonksiyonunu kullanabilmek için kategorik nitelikler LabelEnceder() yardımı ile 
# ayrık sayısal hale dönüştürülmüştür. 
# Dönüşüm sonrasında tüm niteliklerin veri tipi aşağıda sağda verilen ekran görüntüsünden görülebilir.
# Ayrıca döngünün her bir adımında kategorik niteliklerin indeks numaraları 
# kategorikNitelikler listesine atanmıştır. 
# Döngü bittikten sonra hedef nitelik bu listeden çıkarılmıştır.
# kategorikNitelikler listesi mixedNB() fonksiyonunun bir argümanı olarak kullanılacaktır.


label_encoder =  LabelEncoder()

kagetorikNitelikler = []
for nitelik in veriSeti.columns:
    if veriSeti[nitelik].dtype=="object":
     kagetorikNitelikler.append(veriSeti.columns.get_loc(nitelik))
     veriSeti.loc[:,nitelik]= label_encoder.fit_transform(veriSeti.loc[:,nitelik])
     veriSeti[nitelik]=veriSeti[nitelik].astype("int64")
     
     kagetorikNitelikler.pop(len(kagetorikNitelikler)-1)
     
print(veriSeti.describe())
  
# Scikit-learn kütüphanesinin model_selection modülünden train_test_split() fonk. kullanilarak
# egitim ve test veri setleri olusturulmustur. Verinin %70 'i egitim ,%30 ise test veri setinde olacak sekilde
# frac=0.7 ikiye ayirma yontemi ile ayrilmistir.

X = veriSeti.iloc[:, 0:16]
y = veriSeti.iloc[:, 16]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1
)
print("X_train------------------------------")
print(X_train)
print("X_test------------------------------")
print(X_test)
print("y_train------------------------------")
print(y_train)
print("y_test-------------------------------")
print(y_test)