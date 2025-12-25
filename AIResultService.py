import joblib
import os
import sys

class AIResultService:
    # Singleton icin static degisken
    _instance = None
    
    # Yuklenen modelleri ve vektorlestiriciyi hafizada tutacagimiz sozlukler
    _vectorizer = None
    _models = {}

    def __new__(cls):
        """
        Singleton desenini garanti eder.
        Ilk cagrilista butun modelleri RAM'e yukler (Eager Loading).
        Boylece kullanici butona basinca bekleme yapmaz.
        """
        if cls._instance is None:
            cls._instance = super(AIResultService, cls).__new__(cls)
            cls._instance._load_resources()
        return cls._instance

    def _load_resources(self):
        """
        Tum .pkl dosyalarini (Modeller ve Vektorlestirici) diskten okuyup RAM'e yukler.
        """
        try:
            # Calisilan dizini bul (app.py'nin oldugu yer)
            # Not: Dosyalarin ana dizinde oldugunu varsayiyoruz.
            base_path = os.getcwd()
            
            # 1. Vektorlestiriciyi yukle (Bu olmazsa hicbir model calismaz)
            vec_path = os.path.join(base_path, "tfidf_vectorizer.pkl")
            if os.path.exists(vec_path):
                self._vectorizer = joblib.load(vec_path)
                print("Vektorlestirici (TF-IDF) basariyla yuklendi.")
            else:
                print(f"HATA: {vec_path} bulunamadi!")

            # 2. Modelleri yukle
            # Dosya isimleri ile arayuzdeki isimleri eslestiriyoruz
            model_files = {
                "Logistic Regression": "model_logistic_regression.pkl",
                "Random Forest": "model_random_forest.pkl",
                "Naive Bayes": "model_naive_bayes.pkl"
            }

            for model_name, filename in model_files.items():
                file_path = os.path.join(base_path, filename)
                if os.path.exists(file_path):
                    self._models[model_name] = joblib.load(file_path)
                    print(f"Model yuklendi: {model_name}")
                else:
                    print(f"UYARI: {filename} bulunamadi, bu model kullanilamayacak.")
                    
        except Exception as e:
            print(f"Kritik Hata (Resource Loading): {str(e)}")

    def predict_ai_probability(self, model_name, text_input):
        """
        Verilen metni ve secilen modeli kullanarak AI olma olasiligini hesaplar.
        
        Girdi:
        - model_name: UI'dan gelen model ismi (orn: "Random Forest")
        - text_input: Analiz edilecek metin stringi
        
        Cikti:
        - float: %0.0 ile %1.0 arasinda AI olma olasiligi.
        """
        # 1. Gerekli dosyalar yuklenmis mi kontrol et
        if self._vectorizer is None:
            return 0.0 # Veya hata firlatilabilir
        
        if model_name not in self._models:
            print(f"Hata: {model_name} yuklu degil!")
            return 0.0

        try:
            # 2. Metni matematige cevir (Vektorlestirme)
            input_vector = self._vectorizer.transform([text_input])
            
            # 3. Secilen modeli getir
            selected_model = self._models[model_name]
            
            # 4. Tahmin yap (predict_proba [Human%, AI%] dondurur)
            # Biz index 1'i (AI olasiligini) aliyoruz.
            probabilities = selected_model.predict_proba(input_vector)[0]
            ai_probability = probabilities[1]
            
            return ai_probability
            
        except Exception as e:
            print(f"Tahminleme hatasi: {str(e)}")
            return 0.0

# --- EAGER INITIALIZATION (Onemli Nokta) ---
# Bu dosya import edildigi anda asagidaki satir calisir 
# ve nesne hemen olusturulup RAM'e yuklenir. Bekleme olmaz.
ai_result_service = AIResultService()