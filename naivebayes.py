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

# Kontenjans tablosundaki tn, tp, fn, fp değerlerinin sırasıyla aynı isimlerde
# tanımlanan değişkenlere atanabilmesi için my_cm.ravel() kullanılmıştır.
tn,fp,fn,tp = my_cm.ravel()
print("True Negatives",tn)
print("False Positives",fp)
print("False Negatives",fn)
print("True Positives",tp)

# Bu bilgiler ışığında, bir önceki bölümde k-NN model performansı değerlendirilirken
# formüller yardımı ile hesaplanan tüm ölçütler Naive Bayes sınıflandırıcı modeli için de hesaplanabilir.
# Bu bölümde classification_report() ile en temel performans değerlendirme ölçütleri elde edilmiştir. 

my_report = classification_report(y_true=y_test,y_pred=y_tahmin,labels=["no","yes"])
print(my_report)

# yes sınıfının pozitif sınıf seçilmiş olması durumunda;
# gerçekte banka vadeli mevduatına abone olan müşteriler arasında,
# modelin doğru şekilde öngördüğü ve gerçekte de banka vadeli mevduatına abone olan müşterilerin oranı %53’tür (duyarlılık/recall).

# no sınıfının pozitif sınıf seçilmiş olması durumunda; 
# gerçekte banka vadeli mevduatına abone olmayan müşteriler arasında, 
# modelin doğru şekilde öngördüğü ve gerçekte de banka vadeli mevduatına abone olmayan müşterilerin oranı ise %92’tür (duyarlılık/recall).

#Sınıfa bağlı kesinlik ve F-Ölçüsü değerleri de yukarıdakilere benzer şekilde yorumlanabilir.

# Hem gerçek hem de tahmin edilen yes sınıfına ait müşteriler ele alındığında,
# aslında modelin yes sınıfına ait gözlemleri daha az başarı ile tahmin ettiği görülebilir. 
# Bunun bir nedeni olarak veri setinde 39922 adet no, 5289 adet ise yes sınıfına ait örneğin yer alması olabilir. 
# Yani modelin eğitimi ve testinde no sınıfına ait örneklerin sayısı, 
# yes sınıfına ait örneklerin sayısının yaklaşık 8 katıdır. 
# Model no sınıfına ait daha çok örnek görmüştür ve bu nedenle yes sınıfına ait örnekleri
# daha az başarı ile tahmin etmesi çok şaşırtıcı değildir. İşte özellikle bu 
# gibi sınıf dengesizliğinin olduğu durumlarda 
# her ne kadar doğruluk %87 elde edilmişse de F-Ölçüsü model performansını belirlemede daha etkilidir.
# yes sınıfının pozitif seçilmesi durumunda F-Ölçüsü %49 çıkmıştır. 
# Oysa, pozitif sınıfın no seçilmesi durumunda F-Ölçüsü %93’tür. 
# Dolayısıyla sınıflandırıcının nihai performans değerinin
# belirlenebilmesinde makro ortalama (macro averaging) performans değerleri etkili olabilir.
# Yani doğruluk %87, duyarlılık %72, kesinlik %70 ve F-Ölçüsü %71 alınabilir.
# --------------------------------------------------------------------------------------------
# 5-kat Tabakalı Çapraz Geçerleme
# Tabakalı çapraz geçerlemenin gerçekleştirilebilmesi için sklearn.model_selection modülündeki StratifiedKFold() fonksiyonu kullanılmıştır.
# n_splits kaç kat çapraz geçerleme yapılacağını/kaç parçanın oluşturulacağını,
# shuffle gruplara ayırmadan önce her sınıfın örneklerinin karıştırılıp karıştırılmayacağını gösteren parametrelerdir.
# Bu bilgiler yardımı ile cv nesnesi oluşturulmuştur.
# Ardından for döngüsü içinde eğitim ve testte kullanılacak örneklerin indeks değerleri aşağıdaki gibi yazdırılmıştır.
# cv.split() sırasıyla tahmini sağlayan nitelikler (X) ile hedef niteliği (y) almaktadır. 
# Orijinal veri seti birbirine olabildiğince eşit sayıda 5 parçaya ayrılmaktadır. 
# Her parçanın biri test, diğer dördü ise eğitim veri setinin oluşturulması için kullanılmaktadır. 
# for döngüsünde train_index ve test_index her bir iterasyonda sırasıyla eğitim ve test veri setlerinde 
# yer alacak gözlemlerin indeks bilgisini tutmaktadır.

from sklearn.model_selection import StratifiedKFold
k=5
cv= StratifiedKFold(n_splits=k,shuffle=True,random_state=1)
for train_index,test_index in cv.split(X=veriSeti.iloc[:,0:16],y=veriSeti.y):
    print("Egitim Indisleri :",train_index)
    print("Test Indisleri:",test_index,"\n")

#  Örneğin ilk iterasyonda 0, 1, 2 …. 45208, 45209 ve 45210 indeks numaralı örnekler eğitim veri setinde  
#  6, 12, 28, …, 45199, 45204 ve 45205 indeks numaralı örnekler ise test veri setinde yer alacaktır.    

#  5 farklı iterasyon olacağından, 5 defa da performans değerlendirme ölçütleri yeniden hesaplanacaktır (Örneğin 5 farklı doğruluk, 5 farklı hata, 5 farklı duyarlılık gibi).
#  Bu örnek için doğruluk ve F-Ölçüsünün hesaplanmasına karar verilmiştir. 
#  Dolayısıyla doğruluk ve F-Ölçüsü metriklerinin her iterasyonda hesaplanacak değerlerini tutabilmek için dogruluk 
#  ve F1 adında iki liste tanımlanmıştır.
#  Sonrasında cv nesnesi yukarıda detayları verildiği şekilde tanımlanarak for döngüsü yapılandırılmıştır. 
#  for döngüsünün her bir iterasyonunda X_train, X_test, y_train ve y_test değişkenleri yapılandırılmaktadır.
#  Ardından Naive Bayes sınıflandırıcı modeli kurulmakta (nb_model), model tahminleri elde edilmekte (y_tahmin), 
#  sınıflandırma raporundan (my_report) doğruluk ve F-Ölçüsü değerleri çekilerek sırasıyla dogruluk ve F1 listelerine eklenmektedir.

dogruluk = []
F1 =[]
k=5
cv = StratifiedKFold(n_splits=k,shuffle=True,random_state=1)

for train_index,test_index in cv.split(X=veriSeti,y=veriSeti.y):
    X_train,X_test,y_train,y_test = veriSeti.iloc[train_index,0:16],veriSeti.iloc[test_index,0:16],veriSeti.iloc[train_index,16],veriSeti.iloc[test_index,16]
    
# Naive Bayes Siniflandiricisinin Oluşturulması
nb_model = MixedNB(categorical_features=kagetorikNitelikler)
nb_model.fit(X_train,y_train)

# Naive Bayes Siniflandirici Tahminlerinin Elde edilmesi 
y_tahmin =nb_model.predict(X_test)
y_tahmin = label_encoder.inverse_transform(y_tahmin)
y_test =label_encoder.inverse_transform(y_test)

# Performans degerlendirme 
my_report = classification_report(y_true=y_test,y_pred=y_tahmin,labels=["no","yes"],output_dict=True)
dogruluk.append(my_report["accuracy"])
F1.append(my_report["yes"]["f1-score"])

print(my_report)



 


