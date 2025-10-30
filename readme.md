# Dashboard Analisis PHK Global (2020-2025)

## 🎯 Deskripsi

Dashboard interaktif untuk menganalisis tren pemutusan hubungan kerja (PHK) global dari tahun 2020-2025. Dashboard ini menyediakan visualisasi yang informatif dan elegant dengan berbagai fitur analisis.

## ✨ Fitur Utama

### 1. 📊 Visualisasi Interaktif
- Distribusi Total PHK (dengan log scale)
- Distribusi Persentase PHK
- Boxplot Pendanaan Perusahaan
- Matriks Korelasi Variabel Numerik
- Tren PHK per Kuartal
- Total PHK Berdasarkan Benua
- Distribusi Skala PHK Perusahaan

### 2. 🔍 Analisis Detail
- Top 10 Perusahaan dengan PHK Terbanyak
- Top 10 Industri dengan PHK Terbanyak
- Top 10 Negara dengan PHK Terbanyak
- Analisis PHK per Tahun
- Analisis Berdasarkan Tahap Perusahaan

### 3. 🤖 Machine Learning
- Prediksi jumlah PHK menggunakan Linear Regression
- Visualisasi Actual vs Predicted
- Feature Importance Analysis
- Custom Prediction Tool
- Model Performance Metrics (R², MAE)

### 4. ⚖️ Comparison Tool
- Perbandingan antar Region
- Perbandingan antar Industri
- Perbandingan antar Negara
- Perbandingan antar Tahun
- Tren waktu untuk setiap kategori

### 5. 🎨 User Interface
- **Mode Gelap & Terang** (Dark/Light Mode Toggle)
- Sidebar dengan multiple filters:
  - Filter Tahun
  - Filter Region
  - Filter Industri
  - Filter Negara
  - Filter Skala PHK
- Responsive design
- Interactive charts dengan hover tooltips, zoom, dan pan

### 6. 📥 Export Functionality
- Download data dalam format CSV
- Download data dalam format Excel
- Preview data yang difilter
- Statistik deskriptif

## 🚀 Cara Menjalankan

### Instalasi Dependencies

```bash
cd streamlit_dashboard
pip install -r requirements.txt
```

### Menjalankan Dashboard

```bash
streamlit run app.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`

## 📁 Struktur Folder

```
streamlit_dashboard/
├── app.py                          # Main dashboard application
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation (file ini)
├── data/
│   └── layoffs_cleaned_featured.csv   # Dataset
└── utils/                          # Helper functions (optional)
```

## 📊 Dataset

Dataset yang digunakan adalah `layoffs_cleaned_featured.csv` yang berisi:
- **company**: Nama perusahaan
- **location**: Lokasi perusahaan
- **total_laid_off**: Jumlah total PHK
- **date**: Tanggal PHK
- **percentage_laid_off**: Persentase PHK
- **industry**: Industri perusahaan
- **source**: Sumber data
- **stage**: Tahap perusahaan (Startup, Series A, B, C, dll)
- **funds_raised_(millions$)**: Dana yang dihimpun (dalam juta USD)
- **country**: Negara
- **region**: Benua/Region
- **layoff_scale**: Skala PHK (Kecil, Sedang, Besar)
- **year**: Tahun
- **month**: Bulan
- **quarter**: Kuartal

## 🎨 Teknologi yang Digunakan

- **Streamlit**: Framework untuk dashboard interaktif
- **Plotly**: Library untuk visualisasi interaktif
- **Pandas**: Data manipulation dan analysis
- **Scikit-learn**: Machine learning (Linear Regression)
- **OpenPyXL**: Excel file handling
- **ReportLab**: PDF generation

## 📝 Fitur yang Akan Datang (Future Enhancements)

- [ ] PDF Report Generation dengan visualisasi
- [ ] More advanced ML models (Random Forest, XGBoost)
- [ ] Real-time data updates
- [ ] Multi-language support
- [ ] Advanced filtering dengan date range picker
- [ ] Dashboard personalization

## 💡 Tips Penggunaan

1. **Filter Data**: Gunakan sidebar untuk memfilter data berdasarkan kriteria yang Anda inginkan
2. **Dark/Light Mode**: Toggle mode tampilan dengan tombol 🌓 di sidebar
3. **Interactive Charts**: Hover pada chart untuk melihat detail, zoom in/out, dan pan
4. **Comparison**: Gunakan tab "Perbandingan" untuk membandingkan hingga 5 item sekaligus
5. **ML Prediction**: Coba fitur prediksi custom di tab "Prediksi ML"
6. **Export**: Download data yang sudah difilter di tab "Export"

## 📧 Kontak

Jika ada pertanyaan atau saran, silakan hubungi developer.

---

**Dibuat dengan ❤️ menggunakan Streamlit & Plotly**
