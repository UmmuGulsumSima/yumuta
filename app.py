from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError


# Flask uygulaması
app = Flask(__name__)

# Eğitimli modeli yükleme
# Modeli yükle
model = tf.keras.models.load_model(
    "C:/Users/User/OneDrive/Masaüstü/yumuta/egg_quality_model.h5",
    custom_objects={"mse": MeanSquaredError()}
) # Eğittiğin modeli .h5 olarak kaydetmiş olmalısın

# Parametreleri ölçeklendirmek için scaler (eğitimde kullandığın MinMaxScaler ile aynı olmalı)
scaler = MinMaxScaler(feature_range=(0, 1))
# Eğitim verilerindeki minimum ve maksimum değerleri kullanarak scaler'ı ayarla
scaler.fit([[0, 0, 0, 0, 0, 0], [100, 100, 100, 100, 100, 100]])
 # Eğitim verilerindeki minimum ve maksimum değerleri ayarla

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Formdan gelen verileri al
            yas = float(request.form["yas"])
            agirlik = float(request.form["agirlik"])
            uzunluk = float(request.form["uzunluk"])
            genislik = float(request.form["genislik"])
            sekil_indeksi = float(request.form["sekil_indeksi"])
            mukavemet = float(request.form["mukavemet"])

            # Girdi verilerini ölçeklendirme
            input_data = np.array([[yas, agirlik, uzunluk, genislik, sekil_indeksi, mukavemet]])

            # Ölçeklendirilmiş veriyi oluştur
            scaled_data = scaler.transform(input_data)

            # Tahmini hesapla
            tahmin = model.predict(scaled_data)
            tahmin = tahmin[0][0]  # Modelin çıktısını alınır

            return render_template("index.html", tahmin=round(tahmin, 2))

        except Exception as e:
            return render_template("index.html", error=str(e), tahmin=None)

    return render_template("index.html", tahmin=None)
if __name__ == "__main__":
    app.run(debug=True, port=5000)
