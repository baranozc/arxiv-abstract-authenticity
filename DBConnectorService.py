import sqlite3
import pandas as pd
import os
from datetime import datetime

class DBConnectorService:
    # Singleton yapisi icin sinif degiskeni
    _instance = None
    
    # --- AYARLAR ---
    DB_FOLDER = "db"             # Alt klasor adi
    DB_FILE = "app_database.db"  # Dosya adi

    def __new__(cls):
        """
        Singleton desenini garanti eder.
        """
        if cls._instance is None:
            cls._instance = super(DBConnectorService, cls).__new__(cls)
            cls._instance._initialize_database()
        return cls._instance

    def _initialize_database(self):
        """
        Veritabani klasorunu ve baglantisini yonetir.
        Oncelikle 'db' klasoru yoksa olusturur, sonra baglanir.
        """
        # 1. Klasor yolunu ve tam dosya yolunu hazirla
        # Kodun calistigi dizini baz alir
        current_dir = os.getcwd()
        folder_path = os.path.join(current_dir, self.DB_FOLDER)
        db_path = os.path.join(folder_path, self.DB_FILE)

        # 2. Klasor yoksa olustur (Bu adim cok onemlidir, yoksa hata verir)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"'{self.DB_FOLDER}' klasoru olusturuldu.")

        # 3. Dosya var mi kontrol et
        file_exists = os.path.exists(db_path)
        
        # 4. Baglantiyi kur
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        # 5. Eger dosya yeni olustuysa tablolari kur
        if not file_exists:
            print(f"Veritabani ({db_path}) bulunamadi. Sifirdan kuruluyor...")
            self._create_tables()
            self._seed_initial_data()
        else:
            print(f"Veritabani baglantisi basarili: {db_path}")

    def _create_tables(self):
        """
        Normalizasyon kurallarina uygun 3 tabloyu olusturur.
        """
        # 1. Tablo: Modeller
        self.cursor.execute("""
            CREATE TABLE models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE
            )
        """)

        # 2. Tablo: Sonuclar
        self.cursor.execute("""
            CREATE TABLE results (
                id INTEGER PRIMARY KEY,
                label_name TEXT UNIQUE
            )
        """)

        # 3. Tablo: Gecmis
        self.cursor.execute("""
            CREATE TABLE history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                model_id INTEGER,
                ai_probability REAL,
                result_id INTEGER,
                created_at TEXT,
                FOREIGN KEY(model_id) REFERENCES models(id),
                FOREIGN KEY(result_id) REFERENCES results(id)
            )
        """)
        self.conn.commit()

    def _seed_initial_data(self):
        """
        Varsayilan verileri (Seed Data) girer.
        """
        models = [
            (1, 'Logistic Regression'),
            (2, 'Random Forest'),
            (3, 'Naive Bayes')
        ]
        self.cursor.executemany("INSERT INTO models (id, model_name) VALUES (?, ?)", models)

        labels = [
            (1, 'HUMAN'),
            (2, 'AI')
        ]
        self.cursor.executemany("INSERT INTO results (id, label_name) VALUES (?, ?)", labels)
        
        self.conn.commit()

    def get_model_id(self, model_name):
        self.cursor.execute("SELECT id FROM models WHERE model_name = ?", (model_name,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def insert_log(self, text, model_name, ai_prob, is_ai):
        """
        Yeni kayit ekler.
        """
        model_id = self.get_model_id(model_name)
        result_id = 2 if is_ai else 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        query = """
            INSERT INTO history (input_text, model_id, ai_probability, result_id, created_at)
            VALUES (?, ?, ?, ?, ?)
        """
        self.cursor.execute(query, (text, model_id, ai_prob, result_id, timestamp))
        self.conn.commit()

    def get_logs_dataframe(self):
        """
        Gecmis verileri okur ve DataFrame doner.
        """
        query = """
            SELECT 
                h.id,
                h.created_at as Tarih,
                m.model_name as Model,
                r.label_name as Tahmin,
                h.ai_probability as Guven_Orani,
                h.input_text as Girilen_Metin
            FROM history h
            JOIN models m ON h.model_id = m.id
            JOIN results r ON h.result_id = r.id
            ORDER BY h.id DESC
        """
        return pd.read_sql_query(query, self.conn)

    def delete_log(self, log_id):
        self.cursor.execute("DELETE FROM history WHERE id = ?", (log_id,))
        self.conn.commit()

    def close_connection(self):
        self.conn.close()