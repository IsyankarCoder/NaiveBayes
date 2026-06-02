import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mixed_naive_bayes import MixedNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report

# Veri Okuma
veriSeti = pd.read_csv("bank-full.csv",sep=";")

# Veri Ön İşleme
print("--- VeriSeri Type----")
print(veriSeti.dtypes)
print(veriSeti.head())

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
    if veriSeti[nitelik].dtype in ['object', 'string']:
     kagetorikNitelikler.append(veriSeti.columns.get_loc(nitelik))
     veriSeti[nitelik] = veriSeti[nitelik].astype(object)
     veriSeti.loc[:,nitelik]= label_encoder.fit_transform(veriSeti.loc[:,nitelik])
     veriSeti[nitelik]=veriSeti[nitelik].astype("int64")
     
     kagetorikNitelikler.pop(len(kagetorikNitelikler)-1)
     
     
print("--- VeriSeri Type----")
print(veriSeti.dtypes)
  
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

# Modelleme
# Naive Bayes sınıflandırıcı modelinin (nb_model) oluşturulabilmesi için MixedNB() kullanılmıştır.
# Bunun için veri setindeki kategorik niteliklerin indeksini içeren kategorikNitelikler listesi verilmiştir (categorical_features). 
# Ardından nb_model modelinin fit() fonksiyonuna eğitim veri seti ve eğitim veri setinin hedef niteliği verilmiştir.
print("--------- Kategorik Nitelikler ---------------")
print(kagetorikNitelikler)
nb_model = MixedNB(categorical_features=kagetorikNitelikler)
nb_model.fit(X_train,y_train)

# Performans Değerlendirme
# Naive Bayes sınıflandırıcı modelinin performansının değerlendirilebilmesi için nb_model.predict() fonksiyonundan faydalanılmıştır.
# Modelin tahminleri ve test veri setinin hedef niteliği görüntülenirse
# LabelEncoder() ile sayısal hale dönüştürüldüğü biçimiyle yer almakta oldukları görülebilir.
# İlgili değerleri kategorik hale (orijinal haline) döndürmek için 
# label_encoder.inverse_transform() kullanılmıştır.
# Bu adım elde edilecek olan kontenjans tablosunun daha iyi yorumlanabilmesini sağlamak amacıyla yapılmıştır.

y_tahmin = nb_model.predict(X_test)
y_tahmin = label_encoder.inverse_transform(y_tahmin)
y_test = label_encoder.inverse_transform(y_test)

print("--- ytahmin----")
print(y_tahmin)
print("--- ytest ----")
print(y_test)

# Kontenjans tablosu sklearn.metrics içindeki confusion_matrix() ile elde edilmiştir.
# “yes”, yani evet sınıfı pozitif sınıf olarak kabul edilmiştir. 
# Ardından, kontenjans tablosunun görsel açıdan daha iyi biçimde oluşturulabilmesi için ConfusionMatrixDisplay() fonksiyonu kullanılmıştır.

my_cm = confusion_matrix(y_true=y_test,y_pred=y_tahmin,labels=["no","yes"])
my_cm_p = ConfusionMatrixDisplay(my_cm,display_labels=["no","yes"])

my_cm_p.plot()
plt.show()
