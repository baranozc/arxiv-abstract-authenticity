import unittest
import os
import sqlite3
import pandas as pd
from streamlit.testing.v1 import AppTest
from unittest.mock import MagicMock, patch

# Test edilecek siniflari import ediyoruz
from DBConnectorService import DBConnectorService
from AIResultService import AIResultService

class TestProjectWhitebox(unittest.TestCase):

    def setUp(self):
        """
        Her testten once calisir. 
        Singleton yapilari testleri etkilemesin diye instance'lari sifirlariz.
        Bu tam bir Whitebox mudahalesidir (Private degiskeni manupule ediyoruz).
        """
        DBConnectorService._instance = None
        AIResultService._instance = None

    def tearDown(self):
        """
        Her testten sonra temizlik yapar.
        Singleton baglantisini zorla kapatir ve dosyalari siler.
        """
        # 1. ONCE BAGLANTIYI ZORLA KAPAT (CRITICAL STEP)
        # Singleton instance duruyorsa, icindeki baglantiyi oldurmeliyiz.
        # Yoksa dosya 'kullanimda' oldugu icin silinemez.
        if DBConnectorService._instance is not None:
            try:
                DBConnectorService._instance.conn.close()
            except Exception as e:
                print(f"Baglanti kapatma hatasi: {e}")
            
            # Instance'i oldur
            DBConnectorService._instance = None

        # 2. DOSYALARI SIL
        temp_files = ["test_db.db", "test_db_del.db", "test_db_threshold.db"]
        
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except PermissionError:
                    print(f"UYARI: {f} silinemedi cunku hala kullanimda!")
                except Exception as e:
                    print(f"Temizlik hatasi ({f}): {e}")

    # --- TEST 1: Singleton Mimarisi Dogrulama ---
    def test_singleton_behavior(self):
        """
        Amac: DBConnectorService'in gercekten tek bir nesne (instance) urettigini kanitlamak.
        Yontem: Iki kere nesne olusturup hafiza adreslerini (id) kiyaslariz.
        """
        # 1. İlk nesneyi oluştur
        db1 = DBConnectorService()
        
        # 2. İkinci nesneyi oluştur
        db2 = DBConnectorService()
        
        # 3. Whitebox Kontrolü: İkisi de aynı RAM adresini mi gösteriyor?
        self.assertIs(db1, db2, "Singleton hatasi: Nesneler farkli!")
        self.assertEqual(id(db1), id(db2), "Singleton hatasi: ID'ler uyusmuyor!")

    # --- TEST 2: AI Servisi Kaynak Yukleme (Mocking) ---
    @patch('AIResultService.joblib.load')
    @patch('AIResultService.os.path.exists')
    def test_ai_resource_loading(self, mock_exists, mock_joblib_load):
        """
        Amac: Dosyalar diskte varmis gibi davranip servisin onlari yukleyip yuklemedigini kontrol etmek.
        Yontem: os.path.exists ve joblib.load fonksiyonlarini 'mock'luyoruz (taklit ediyoruz).
        """
        # Senaryo: Dosyalar var densin
        mock_exists.return_value = True
        mock_joblib_load.return_value = "MockModel"

        # Servisi baslat
        service = AIResultService()
        
        # Whitebox Kontrolü: İçerideki _models sözlüğü doldu mu?
        self.assertIn("Random Forest", service._models)
        self.assertIsNotNone(service._vectorizer)
        print("Test 2 (AI Loading): Basarili")

    # --- TEST 3: DB Normalizasyon ve Insert Mantigi ---
    def test_db_insert_logic(self):
        """
        Amac: insert_log fonksiyonunun arka planda Model ID ve Result ID'yi 
        dogru sekilde bulup iliskisel tabloya yazdigini test etmek.
        """
        # Test icin gecici bir DB ismi ayarlayalim (Kodun icine mudahale)
        DBConnectorService.DB_FILE = "test_db.db"
        DBConnectorService.DB_FOLDER = "." # Ana dizine koysun
        
        db = DBConnectorService()
        
        # Veri ekle: Random Forest (ID:2 olmali), AI (ID:2 olmali)
        # Not: Seed data ile Random Forest ID'si 2 olarak geliyor.
        db.insert_log("Test Metni", "Random Forest", 0.95, True)
        
        # Whitebox Kontrolü: SQL ile ham veriyi cekip ID'leri kontrol edelim
        cursor = db.conn.cursor()
        cursor.execute("SELECT model_id, result_id, ai_probability FROM history WHERE input_text='Test Metni'")
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 2, "Model ID yanlis eslesmis (Random Forest=2 olmali)")
        self.assertEqual(row[1], 2, "Result ID yanlis eslesmis (AI=2 olmali)")
        self.assertEqual(row[2], 0.95, "Olasilik degeri yanlis kaydedilmis")
        
        db.close_connection()
        DBConnectorService._instance = None

        # TEMIZLIK:
        if os.path.exists("test_db.db"):
            os.remove("test_db.db")

    # --- TEST 4: AI Tahmin Algoritmasi (Probability Flow) ---
    def test_ai_prediction_flow(self):
        """
        Amac: predict_ai_probability fonksiyonunun modelden gelen [Human, AI] dizisinden
        dogru indeksi (1. indeks) aldigini dogrulamak.
        """
        service = AIResultService()
        
        # Mocking: Vektorizer ve Model nesnelerini elimizle olusturuyoruz
        mock_vectorizer = MagicMock()
        mock_model = MagicMock()
        
        # Modelin predict_proba ciktisini simule et: [Human=0.2, AI=0.8]
        # Scikit-learn formatinda sonuc [[0.2, 0.8]] doner
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        
        # Servisin icine bu sahte nesneleri enjekte ediyoruz (Internal State Injection)
        service._vectorizer = mock_vectorizer
        service._models["TestModel"] = mock_model
        
        # Fonksiyonu calistir
        result = service.predict_ai_probability("TestModel", "deneme metni")
        
        # Whitebox Kontrolü: Sonuc 0.8 mi? (Yani listenin 1. elemanini mi aldi?)
        self.assertEqual(result, 0.8)
        
        # Vektorizer calisti mi?
        mock_vectorizer.transform.assert_called_once()

    # --- TEST 5: DB Silme Islemi ---
    def test_db_delete_operation(self):
        """
        Amac: delete_log fonksiyonunun gercekten o ID'li satiri silip silmedigini kontrol etmek.
        """
        DBConnectorService.DB_FILE = "test_db_del.db"
        DBConnectorService.DB_FOLDER = "."
        
        db = DBConnectorService()
        
        try:
            # --- TEST MANTIGI ---
            # Once bir veri ekleyelim
            db.insert_log("Silinecek Metin", "Naive Bayes", 0.1, False)
            
            # Eklenen verinin ID'sini alalim
            df = db.get_logs_dataframe()
            
            # Guvenlik kontrolu: Tablo bos mu?
            if df.empty:
                self.fail("Veri eklenemedigi icin test yapilamiyor.")

            log_id = df.iloc[0]['id'] 
            
            # Silelim
            db.delete_log(log_id)
            
            # Whitebox Kontrolü
            cursor = db.conn.cursor()
            cursor.execute("SELECT * FROM history WHERE id=?", (int(log_id),))
            result = cursor.fetchone()
            
            self.assertIsNone(result, "Silme islemi basarisiz, kayit hala duruyor!")

        finally:
            # --- ZORUNLU TEMIZLIK ---
            # Test fail etse de, hata verse de burasi CALISIR.
            db.close_connection()
            DBConnectorService._instance = None # Singleton'i sifirla
            
            if os.path.exists("test_db_del.db"):
                try:
                    os.remove("test_db_del.db")
                except Exception as e:
                    print(f"Manuel silme hatasi: {e}")

        # --- YENİ TEST 8: Arayüz (UI) Uyarı Testi ---
    def test_ui_empty_input_warning(self):
        """
        Amac: Kullanici hicbir sey yazmadan 'Analiz Et' butonuna bastiginda,
        arayuzde (main.py) o sari uyari kutusunun cikip cikmadigini test eder.
        Referans: Attigin ekran goruntusundeki 'Lutfen analiz icin...' uyarisi.
        """
        # 1. Uygulamayi sanal ortamda baslat ("main.py" senin dosya adin olmali)
        at = AppTest.from_file("main.py")
        at.run()

        # 2. Metin kutusunu bilerek BOS birakiyoruz (default zaten bos)
        
        # 3. 'Analiz Et' butonuna sanal olarak tikla
        # (at.button[0], sayfadaki ilk butonu temsil eder)
        at.button[0].click().run()

        # 4. KONTROL: Ekranda warning (sari kutu) var mi?
        # at.warning listesi ekrandaki tum sari kutulari tutar.
        if len(at.warning) > 0:
            mesaj = at.warning[0].value
            self.assertEqual(mesaj, "Lutfen analiz icin bir metin girin.", 
                             "Uyari mesaji ekrandaki goruntuden farkli!")
        else:
            self.fail("Butona basildi ama sari uyari kutusu cikmadi!")
    # --- YENİ TEST 7: %50 Eşik Değeri (Threshold) ve Etiketleme ---
    def test_threshold_logic_mapping(self):
        """
        Senaryo: 
        1. Olasılık > 0.5 ise veritabanına AI (Result ID: 2) olarak mı geçiyor?
        2. Olasılık <= 0.5 ise veritabanına HUMAN (Result ID: 1) olarak mı geçiyor?
        """
        # Test DB ayarla
        DBConnectorService.DB_FILE = "test_db_threshold.db"
        DBConnectorService.DB_FOLDER = "."
        db = DBConnectorService()

        # DURUM A: %51 Olasılık (AI Olmalı)
        # app.py içindeki mantık: is_ai = prob > 0.5
        is_ai_case = 0.51 > 0.5  # True
        db.insert_log("AI Metni", "Random Forest", 0.51, is_ai_case)

        # DURUM B: %49 Olasılık (Human Olmalı)
        is_human_case = 0.49 > 0.5 # False
        db.insert_log("Insan Metni", "Random Forest", 0.49, is_human_case)

        # KONTROLLER (Veritabanından Sorgulama)
        cursor = db.conn.cursor()
        
        # Kontrol A: AI Metni için Result ID 2 mi?
        cursor.execute("SELECT result_id FROM history WHERE input_text='AI Metni'")
        ai_result_id = cursor.fetchone()[0]
        self.assertEqual(ai_result_id, 2, "%51 olasilik AI (ID:2) olarak kaydedilmeliydi.")

        # Kontrol B: İnsan Metni için Result ID 1 mi?
        cursor.execute("SELECT result_id FROM history WHERE input_text='Insan Metni'")
        human_result_id = cursor.fetchone()[0]
        self.assertEqual(human_result_id, 1, "%49 olasilik HUMAN (ID:1) olarak kaydedilmeliydi.")

        # Temizlik
        db.close_connection()
        if os.path.exists("test_db_threshold.db"):
            os.remove("test_db_threshold.db")

if __name__ == '__main__':
    unittest.main()