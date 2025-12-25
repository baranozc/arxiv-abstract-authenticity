import streamlit as st
import pandas as pd
import time  # Toast mesaji ve bekleme icin eklendi
from AIResultService import AIResultService
from DBConnectorService import DBConnectorService

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Human vs AI Detector",
    page_icon="🕵️",
    layout="centered"
)

# --- SERVISLERIN BASLATILMASI (SINGLETON) ---
# Burada siniflardan nesne uretiyoruz. 
# Ancak Singleton yapisi sayesinde arka planda 
# zaten olusturulmus olan TEK nesneler bize geliyor.
ai_service = AIResultService()
db_service = DBConnectorService()

# --- ARAYUZ (UI) TASARIMI ---
st.title("🕵️ Human or AI?")

# -- YAN PANEL (SIDEBAR) --
st.sidebar.header("⚙️ Ayarlar")
st.sidebar.info("Analiz icin kullanmak istediginiz algoritmayi secin.")

# Model Secim Kutusu
model_choice = st.sidebar.selectbox(
    "Model Secimi:",
    ("Logistic Regression", "Random Forest", "Naive Bayes")
)

st.sidebar.success(f"✅ {model_choice} Aktif!")
st.sidebar.markdown("---")
st.sidebar.caption(f"Tahmin {model_choice} modeli kullanilarak yapilacak.")

# -- ANA EKRAN --
st.write(f"Su an **{model_choice}** modeli ile analiz yapiyorsunuz.")
st.write("Asagidaki alana Ingilizce dilinde yazilmis akademik bir makale ozeti yapistirin.")

# Metin Giris Alani
user_input = st.text_area("Metni Buraya Giriniz:", height=200)

# --- ANALIZ BUTONU VE MANTIK AKISI ---
if st.button("Analiz Et", type="primary"):
    if user_input:
        with st.spinner('Yapay Zeka Modeli calisiyor...'):
            # 1. ADIM: Hesaplama Servisini Cagir (AIResultService)
            # Model ID'leri veya veritabani islerini bu servis bilmez. Sadece hesaplar.
            ai_prob = ai_service.predict_ai_probability(model_choice, user_input)
            
            # Insan olasiligini matematikle buluyoruz
            human_prob = 1.0 - ai_prob
            
            # Sonuc Etiketi Belirleme (DB kaydi icin lazim olacak)
            is_ai = ai_prob > 0.5

        # 2. ADIM: Sonuclari Ekrana Bas
        st.subheader("Sonuç:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Insan Yazimi Olasiligi", value=f"%{human_prob*100:.1f}")
            st.progress(human_prob)
            
        with col2:
            st.metric(label="Yapay Zeka Olasiligi", value=f"%{ai_prob*100:.1f}")
            st.progress(ai_prob)
            
        # Renkli Uyari Mesaji
        if is_ai:
            st.error(f"🚨 **YAPAY ZEKA** tespiti! (Guven: %{ai_prob*100:.1f})")
        else:
            st.success(f"✅ **INSAN** yazimi tespiti. (Guven: %{human_prob*100:.1f})")

        # 3. ADIM: Veritabanina Kayit (DBConnectorService)
        # Analiz bitti, simdi sonucu 'Log' olarak kaydediyoruz.
        try:
            db_service.insert_log(user_input, model_choice, ai_prob, is_ai)
            st.toast("Analiz sonucu veritabanina kaydedildi!", icon="💾")
        except Exception as e:
            st.error(f"Veritabani hatasi: {str(e)}")

    else:
        st.warning("Lutfen analiz icin bir metin girin.")

# --- GECMIS KAYITLAR BOLUMU (Guncellenmis) ---
st.markdown("---")
st.subheader("🗄️ Analiz Geçmişi")

# Veritabanindan verileri cek
try:
    history_df = db_service.get_logs_dataframe()
    
    if not history_df.empty:
        
        # --- TABLO DUZENLEME VE SECIM MANTIGI ---
        
        # "Seç" sütunu ekle (varsayılan False)
        if "Seç" not in history_df.columns:
            history_df.insert(0, "Seç", False)

        # "Hepsini Seç" Checkbox'ı
        hepsini_sec = st.checkbox("Hepsini Seç")
        if hepsini_sec:
            history_df["Seç"] = True
        else:
            if history_df["Seç"].all():
                history_df["Seç"] = False

        # Data Editor (Düzenlenebilir Tablo)
        edited_df = st.data_editor(
            history_df,
            column_config={
                "Seç": st.column_config.CheckboxColumn(
                    "Sil?",
                    help="Silmek istediklerini seç",
                    default=False,
                )
            },
            disabled=history_df.columns.drop("Seç"), # Sadece "Seç" sütunu değiştirilebilir
            hide_index=True,
            key="editor",
            use_container_width=True,
            height=300
        )
        
        # --- BUTONLAR ---
        col_btn1, col_btn2 = st.columns([1, 4])
        
        with col_btn1:
            sil_btn = st.button("Seçilenleri Sil", type="primary")
            
        with col_btn2:
            # CSV Indirme Butonu
            csv = edited_df.drop(columns=["Seç"]).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Geçmişi İndir (CSV)",
                data=csv,
                file_name='analiz_gecmisi.csv',
                mime='text/csv',
            )
            
        # --- SILME ISLEMI MANTIGI ---
        if sil_btn:
            # Seçili olan satırları filtrele
            silinecekler = edited_df[edited_df["Seç"] == True]
            
            if silinecekler.empty:
                # HATA DURUMU: Hiçbir şey seçilmediyse sağ altta uyarı ver
                st.toast("⚠️ Silinecek kayıt bulunamadı!", icon="🚫")
            else:
                # BAŞARILI DURUM: Seçilenleri veritabanından sil
                # Not: DBConnectorService icindeki delete_log fonksiyonunu kullaniyoruz.
                count = 0
                for index, row in silinecekler.iterrows():
                    # 'id' sütununun veritabanından gelen sütun adı olduğunu varsayıyoruz.
                    # Eğer hata alırsanız veritabanındaki ID sütun adını kontrol edin (örn: 'ID', 'log_id')
                    db_service.delete_log(row['id']) 
                    count += 1
                
                st.success(f"{count} kayıt başarıyla silindi!")
                time.sleep(1) # Kullanıcı mesajı görsün diye kısa bekleme
                st.rerun() # Sayfayı yenile ve güncel tabloyu göster

    else:
        st.info("Henuz hicbir analiz kaydi bulunmamaktadir.")

except Exception as e:
    st.error(f"Gecmis yuklenirken hata: {str(e)}")