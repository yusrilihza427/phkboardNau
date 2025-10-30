import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import base64

# Page configuration
st.set_page_config(
    page_title="Dashboard Analisis PHK Global 2020-2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/light mode
def load_custom_css(theme):
    if theme == "Dark":
        st.markdown("""
        <style>
        html, body, [class*="css"], .block-container, .main {
            background-color: #0e1117 !important;
            color: #f1f1f1 !important;
            transition: all 0.4s ease-in-out;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a23 0%, #0e1117 100%) !important;
            border-right: 1px solid #2b2b36 !important;
        }
        div[data-testid="stMetric"] {
            background: #1e1e26 !important;
            color: #fafafa !important;
            border-radius: 12px !important;
            padding: 20px 10px !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6);
        }
        h1, h2, h3, h4, h5, h6, label, p {
            color: #fafafa !important;
        }
        </style>
        """, unsafe_allow_html=True)

    else:  # Light Mode
        st.markdown("""
        <style>
        html, body, [class*="css"], .block-container, .main {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            transition: all 0.4s ease-in-out;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f4f5f8 100%) !important;
            border-right: 1px solid #dcdcdc !important;
        }
        div[data-testid="stMetric"] {
            background: #ffffff !important;
            color: #1a1a1a !important;
            border-radius: 12px !important;
            padding: 20px 10px !important;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 14px rgba(0, 0, 0, 0.12);
        }
        .stTabs [role="tablist"] button {
            background-color: #f4f5f8 !important;
            color: #1a1a1a !important;
            border-radius: 8px;
            margin-right: 6px;
        }
        .stTabs [role="tablist"] button[aria-selected="true"] {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
        }
        h1, h2, h3, h4, h5, h6, label, p {
            color: #1a1a1a !important;
        }
        </style>
        """, unsafe_allow_html=True)

st.markdown("""
<style>
/* Animasi fade-in untuk konten utama setiap kali tab berubah */
[data-testid="stVerticalBlock"] > div {
    animation: fadeIn 0.5s ease-in-out;
}

/* Definisi animasi */
@keyframes fadeIn {
    0% {
        opacity: 0;
        transform: translateY(10px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""", unsafe_allow_html=True)





# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv('data/layoffs_cleaned_featured.csv')
    
    # Convert date columns
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    
    # Extract year, month, quarter if not present
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month
    if 'quarter' not in df.columns:
        df['quarter'] = df['date'].dt.quarter
        
    return df

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("<div style='display:flex; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
    st.image("pngPHK2.jpg", width=150)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Theme toggle
    st.markdown("---")
    theme_col1, theme_col2 = st.columns([3, 1])
    with theme_col1:
        st.write("**Mode Tampilan:**")
    with theme_col2:
        if st.button("🌓"):
            st.session_state.theme = 'Dark' if st.session_state.theme == 'Light' else 'Light'
            st.rerun()
    
    st.info(f"Mode: **{st.session_state.theme}**")
    
    st.markdown("---")
    st.subheader("📅 Filter Data")
    
    # Year filter
    years = sorted(df['year'].dropna().unique())
    selected_years = st.multiselect(
        "Pilih Tahun",
        options=years,
        default=years
    )
    
    # Region filter
    regions = sorted(df['region'].dropna().unique())
    selected_regions = st.multiselect(
        "Pilih Region",
        options=regions,
        default=regions
    )
    
    # Industry filter
    industries = sorted(df['industry'].dropna().unique())
    selected_industries = st.multiselect(
        "Pilih Industri",
        options=industries,
        default=industries
    )
    
    # Country filter
    countries = sorted(df['country'].dropna().unique())
    selected_countries = st.multiselect(
        "Pilih Negara",
        options=countries,
        default=countries
    )
    
    # Layoff scale filter
    if 'layoff_scale' in df.columns:
        scales = df['layoff_scale'].dropna().unique()
        selected_scales = st.multiselect(
            "Pilih Skala PHK",
            options=scales,
            default=scales
        )
    else:
        selected_scales = None
    
    st.markdown("---")
    st.markdown("### 📥 Download Data")
    
    # Download filtered data as CSV
    @st.cache_data
    def convert_df_to_csv(dataframe):
        return dataframe.to_csv(index=False).encode('utf-8')
    
    # Download filtered data as Excel
    @st.cache_data
    def convert_df_to_excel(dataframe):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            dataframe.to_excel(writer, index=False, sheet_name='Layoffs Data')
        return output.getvalue()

# Apply custom CSS
load_custom_css(st.session_state.theme)

# Filter data
filtered_df = df[
    (df['year'].isin(selected_years)) &
    (df['region'].isin(selected_regions)) &
    (df['industry'].isin(selected_industries)) &
    (df['country'].isin(selected_countries))
]

if selected_scales and 'layoff_scale' in df.columns:
    filtered_df = filtered_df[filtered_df['layoff_scale'].isin(selected_scales)]

# Main content
st.title("📊 Dashboard Analisis PHK Global (2020-2025)")
st.markdown("### Analisis Komprehensif Tren Pemutusan Hubungan Kerja di Seluruh Dunia")

st.markdown("---")

# Key Metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_layoffs = filtered_df['total_laid_off'].sum()
    st.metric("Total PHK", f"{total_layoffs:,.0f}")

with col2:
    total_companies = filtered_df['company'].nunique()
    st.metric("Total Perusahaan", f"{total_companies:,}")

with col3:
    avg_percentage = filtered_df['percentage_laid_off'].mean()
    st.metric("Rata-rata %", f"{avg_percentage:.1f}%")

with col4:
    total_funding = filtered_df['funds_raised_(millions$)'].sum()
    st.metric("Total Funding", f"${total_funding:,.0f}M")

with col5:
    total_countries = filtered_df['country'].nunique()
    st.metric("Total Negara", f"{total_countries}")

st.markdown("---")

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Visualisasi Utama", 
    "🔍 Analisis Detail",
    "🤖 Prediksi ML",
    "⚖️ Perbandingan",
    "📥 Export"
])

# TAB 1: Main Visualizations
with tab1:
    st.header("Visualisasi Utama")
    
    # 1. Distribution of Total Layoffs
    st.subheader("1️⃣ Distribusi Total PHK")
    fig1 = px.histogram(
        filtered_df,
        x='total_laid_off',
        nbins=50,
        title='Distribusi Total PHK (Log Scale)',
        labels={'total_laid_off': 'Jumlah PHK', 'count': 'Frekuensi'},
        color_discrete_sequence=["#fb0303"]
    )
    fig1.update_xaxes(type="log")
    fig1.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Distribution of Layoff Percentage
    st.subheader("2️⃣ Distribusi Persentase PHK")
    fig2 = px.histogram(
        filtered_df,
        x='percentage_laid_off',
        nbins=40,
        title='Distribusi Persentase PHK',
        labels={'percentage_laid_off': 'Persentase PHK (%)', 'count': 'Frekuensi'},
        color_discrete_sequence=['#764ba2']
    )
    fig2.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Boxplot of Funds Raised
    st.subheader("3️⃣ Distribusi Pendanaan Perusahaan")
    fig3 = px.box(
        filtered_df,
        y='funds_raised_(millions$)',
        title='Boxplot Pendanaan Perusahaan (Millions USD)',
        labels={'funds_raised_(millions$)': 'Dana yang Dihimpun (M$)'},
        color_discrete_sequence=['#f093fb']
    )
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Correlation Heatmap
    st.subheader("4️⃣ Korelasi Antar Variabel Numerik")
    corr_cols = ['total_laid_off', 'percentage_laid_off', 'funds_raised_(millions$)']
    corr_data = filtered_df[corr_cols].corr()
    
    fig4 = px.imshow(
        corr_data,
        text_auto='.2f',
        title='Matriks Korelasi',
        color_continuous_scale='RdBu_r',
        aspect='auto',
        labels={'color': 'Korelasi'}
    )
    fig4.update_layout(height=500)
    st.plotly_chart(fig4, use_container_width=True)
    
    # 5. Trend per Quarter
    st.subheader("5️⃣ Tren Jumlah PHK per Kuartal (2020-2025)")
    
    # Prepare quarterly data
    quarterly_df = filtered_df.copy()
    quarterly_df = quarterly_df[quarterly_df['month'].isin([3, 6, 9, 12])]
    layoff_quarter = quarterly_df.groupby(['year', 'month'])['total_laid_off'].sum().reset_index()
    layoff_quarter = layoff_quarter.sort_values(['year', 'month'])
    layoff_quarter['year_month'] = layoff_quarter['year'].astype(str) + '-Q' + (layoff_quarter['month'] // 3).astype(str)
    
    fig5 = px.line(
        layoff_quarter,
        x='year_month',
        y='total_laid_off',
        title='Tren PHK per Kuartal',
        labels={'year_month': 'Tahun-Kuartal', 'total_laid_off': 'Total PHK'},
        markers=True,
        color_discrete_sequence=['#667eea']
    )
    fig5.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)
    
    # 6. Total Layoffs by Region
    st.subheader("6️⃣ Total PHK Berdasarkan Benua")
    region_summary = filtered_df.groupby('region')['total_laid_off'].sum().sort_values(ascending=False).reset_index()
    
    fig6 = px.bar(
        region_summary,
        x='region',
        y='total_laid_off',
        title='Total PHK Berdasarkan Benua',
        labels={'region': 'Benua', 'total_laid_off': 'Jumlah PHK'},
        color='total_laid_off',
        color_continuous_scale='Viridis'
    )
    fig6.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig6, use_container_width=True)
    
    # 7. Distribution of Layoff Scale
    if 'layoff_scale' in filtered_df.columns:
        st.subheader("7️⃣ Distribusi Skala PHK Perusahaan")
        scale_counts = filtered_df['layoff_scale'].value_counts().reset_index()
        scale_counts.columns = ['layoff_scale', 'count']
        
        # Custom order
        scale_order = ['Kecil', 'Sedang', 'Besar']
        scale_counts['layoff_scale'] = pd.Categorical(scale_counts['layoff_scale'], categories=scale_order, ordered=True)
        scale_counts = scale_counts.sort_values('layoff_scale')
        
        fig7 = px.bar(
            scale_counts,
            x='layoff_scale',
            y='count',
            title='Distribusi Skala PHK (Kecil: <50, Sedang: 50-500, Besar: >500)',
            labels={'layoff_scale': 'Kategori Skala PHK', 'count': 'Jumlah Perusahaan'},
            color='layoff_scale',
            color_discrete_map={'Kecil': 'green', 'Sedang': 'orange', 'Besar': 'red'}
        )
        fig7.update_layout(height=500)
        st.plotly_chart(fig7, use_container_width=True)

# TAB 2: Detailed Analysis
with tab2:
    st.header("Analisis Detail")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Top 10 Perusahaan dengan PHK Terbanyak")
        top_companies = filtered_df.groupby('company')['total_laid_off'].sum().sort_values(ascending=False).head(10).reset_index()
        
        fig_top_comp = px.bar(
            top_companies,
            y='company',
            x='total_laid_off',
            orientation='h',
            title='Top 10 Perusahaan',
            labels={'company': 'Perusahaan', 'total_laid_off': 'Total PHK'},
            color='total_laid_off',
            color_continuous_scale='Reds'
        )
        fig_top_comp.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_comp, use_container_width=True)
    
    with col2:
        st.subheader("🏭 Top 10 Industri dengan PHK Terbanyak")
        top_industries = filtered_df.groupby('industry')['total_laid_off'].sum().sort_values(ascending=False).head(10).reset_index()
        
        fig_top_ind = px.bar(
            top_industries,
            y='industry',
            x='total_laid_off',
            orientation='h',
            title='Top 10 Industri',
            labels={'industry': 'Industri', 'total_laid_off': 'Total PHK'},
            color='total_laid_off',
            color_continuous_scale='Blues'
        )
        fig_top_ind.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_ind, use_container_width=True)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🌍 Top 10 Negara dengan PHK Terbanyak")
        top_countries = filtered_df.groupby('country')['total_laid_off'].sum().sort_values(ascending=False).head(10).reset_index()
        
        fig_top_country = px.pie(
            top_countries,
            values='total_laid_off',
            names='country',
            title='Top 10 Negara',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_top_country.update_layout(height=500)
        st.plotly_chart(fig_top_country, use_container_width=True)
    
    with col4:
        st.subheader("📅 PHK per Tahun")
        yearly_layoffs = filtered_df.groupby('year')['total_laid_off'].sum().reset_index()
        
        fig_yearly = px.bar(
            yearly_layoffs,
            x='year',
            y='total_laid_off',
            title='Total PHK per Tahun',
            labels={'year': 'Tahun', 'total_laid_off': 'Total PHK'},
            color='total_laid_off',
            color_continuous_scale='Purples'
        )
        fig_yearly.update_layout(height=500)
        st.plotly_chart(fig_yearly, use_container_width=True)
    
    st.markdown("---")
    
    # Stage analysis
    if 'stage' in filtered_df.columns:
        st.subheader("🚀 Analisis Berdasarkan Tahap Perusahaan")
        stage_analysis = filtered_df.groupby('stage').agg({
            'total_laid_off': 'sum',
            'company': 'count',
            'funds_raised_(millions$)': 'mean'
        }).reset_index()
        stage_analysis.columns = ['Stage', 'Total PHK', 'Jumlah Perusahaan', 'Avg Funding (M$)']
        
        fig_stage = px.scatter(
            stage_analysis,
            x='Avg Funding (M$)',
            y='Total PHK',
            size='Jumlah Perusahaan',
            color='Stage',
            title='Hubungan Funding, PHK, dan Jumlah Perusahaan per Stage',
            labels={'Avg Funding (M$)': 'Rata-rata Funding (M$)', 'Total PHK': 'Total PHK'},
            hover_data=['Jumlah Perusahaan']
        )
        fig_stage.update_layout(height=500)
        st.plotly_chart(fig_stage, use_container_width=True)

# TAB 3: ML Predictions
with tab3:
    st.header("🤖 Prediksi Menggunakan Machine Learning")
    
    st.markdown("""
    Model **Linear Regression** digunakan untuk memprediksi jumlah PHK berdasarkan variabel-variabel lainnya.
    """)
    
    # Prepare data for ML
    ml_data = filtered_df[['total_laid_off', 'percentage_laid_off', 'funds_raised_(millions$)', 'year', 'quarter']].dropna()
    
    if len(ml_data) > 10:
        X = ml_data[['percentage_laid_off', 'funds_raised_(millions$)', 'year', 'quarter']]
        y = ml_data['total_laid_off']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("MAE", f"{mae:,.0f}")
        with col3:
            st.metric("Samples Used", f"{len(ml_data):,}")
        
        st.markdown("---")
        
        # Actual vs Predicted
        st.subheader("📊 Actual vs Predicted Values")
        comparison_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        fig_ml = px.scatter(
            comparison_df,
            x='Actual',
            y='Predicted',
            title='Actual vs Predicted Layoffs',
            labels={'Actual': 'Actual PHK', 'Predicted': 'Predicted PHK'},
            color_discrete_sequence=['#667eea']
        )
        
        # Add diagonal line for perfect prediction
        min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
        max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
        fig_ml.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig_ml.update_layout(height=500)
        st.plotly_chart(fig_ml, use_container_width=True)
        
        st.markdown("---")
        
        # Feature Importance
        st.subheader("📈 Koefisien Model (Feature Importance)")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Koefisien Linear Regression',
            labels={'Coefficient': 'Koefisien', 'Feature': 'Fitur'},
            color='Coefficient',
            color_continuous_scale='RdYlGn'
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("---")
        
        # Prediction Tool
        st.subheader("🔮 Prediksi Custom")
        st.markdown("Masukkan nilai untuk memprediksi jumlah PHK:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            input_percentage = st.number_input("Persentase PHK (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        with col2:
            input_funding = st.number_input("Funding (M$)", min_value=0.0, value=100.0, step=10.0)
        with col3:
            input_year = st.number_input("Tahun", min_value=2020, max_value=2030, value=2024, step=1)
        with col4:
            input_quarter = st.selectbox("Kuartal", options=[1, 2, 3, 4], index=0)
        
        if st.button("🔍 Prediksi", type="primary"):
            input_data = np.array([[input_percentage, input_funding, input_year, input_quarter]])
            prediction = model.predict(input_data)[0]
            
            st.success(f"### Prediksi Jumlah PHK: **{prediction:,.0f}** orang")
    else:
        st.warning("Data tidak cukup untuk melakukan prediksi ML. Silakan sesuaikan filter.")

# TAB 4: Comparison Tool
with tab4:
    st.header("⚖️ Perbandingan Antar Region/Industri")
    
    comparison_type = st.radio(
        "Pilih Tipe Perbandingan:",
        options=["Region", "Industri", "Negara", "Tahun"],
        horizontal=True
    )
    
    if comparison_type == "Region":
        comparison_col = 'region'
        options = sorted(filtered_df[comparison_col].dropna().unique())
        default_selection = options[:3] if len(options) >= 3 else options
    elif comparison_type == "Industri":
        comparison_col = 'industry'
        options = sorted(filtered_df[comparison_col].dropna().unique())
        default_selection = options[:3] if len(options) >= 3 else options
    elif comparison_type == "Negara":
        comparison_col = 'country'
        options = sorted(filtered_df[comparison_col].dropna().unique())
        default_selection = options[:3] if len(options) >= 3 else options
    else:  # Year
        comparison_col = 'year'
        options = sorted(filtered_df[comparison_col].dropna().unique())
        default_selection = options[:3] if len(options) >= 3 else options
    
    selected_items = st.multiselect(
        f"Pilih {comparison_type} untuk dibandingkan (max 5):",
        options=options,
        default=default_selection,
        max_selections=5
    )
    
    if selected_items:
        comparison_data = filtered_df[filtered_df[comparison_col].isin(selected_items)]
        
        # Summary comparison
        st.subheader(f"📊 Perbandingan {comparison_type}")
        
        summary_comparison = comparison_data.groupby(comparison_col).agg({
            'total_laid_off': 'sum',
            'company': 'nunique',
            'percentage_laid_off': 'mean',
            'funds_raised_(millions$)': 'mean'
        }).reset_index()
        
        summary_comparison.columns = [
            comparison_type,
            'Total PHK',
            'Jumlah Perusahaan',
            'Avg % PHK',
            'Avg Funding (M$)'
        ]
        
        st.dataframe(summary_comparison, use_container_width=True)
        
        st.markdown("---")
        
        # Multiple charts for comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_comp1 = px.bar(
                summary_comparison,
                x=comparison_type,
                y='Total PHK',
                title=f'Total PHK per {comparison_type}',
                color=comparison_type,
                labels={comparison_type: comparison_type, 'Total PHK': 'Total PHK'}
            )
            fig_comp1.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_comp1, use_container_width=True)
        
        with col2:
            fig_comp2 = px.bar(
                summary_comparison,
                x=comparison_type,
                y='Jumlah Perusahaan',
                title=f'Jumlah Perusahaan per {comparison_type}',
                color=comparison_type,
                labels={comparison_type: comparison_type, 'Jumlah Perusahaan': 'Jumlah Perusahaan'}
            )
            fig_comp2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_comp2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig_comp3 = px.bar(
                summary_comparison,
                x=comparison_type,
                y='Avg % PHK',
                title=f'Rata-rata % PHK per {comparison_type}',
                color=comparison_type,
                labels={comparison_type: comparison_type, 'Avg % PHK': 'Rata-rata % PHK'}
            )
            fig_comp3.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_comp3, use_container_width=True)
        
        with col4:
            fig_comp4 = px.bar(
                summary_comparison,
                x=comparison_type,
                y='Avg Funding (M$)',
                title=f'Rata-rata Funding per {comparison_type}',
                color=comparison_type,
                labels={comparison_type: comparison_type, 'Avg Funding (M$)': 'Rata-rata Funding (M$)'}
            )
            fig_comp4.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_comp4, use_container_width=True)
        
        st.markdown("---")
        
        # Time series comparison
        st.subheader(f"📈 Tren Waktu per {comparison_type}")
        
        time_series_data = comparison_data.groupby([comparison_col, 'year'])['total_laid_off'].sum().reset_index()
        
        fig_time_comp = px.line(
            time_series_data,
            x='year',
            y='total_laid_off',
            color=comparison_col,
            title=f'Tren PHK per Tahun - Perbandingan {comparison_type}',
            labels={'year': 'Tahun', 'total_laid_off': 'Total PHK', comparison_col: comparison_type},
            markers=True
        )
        fig_time_comp.update_layout(height=500)
        st.plotly_chart(fig_time_comp, use_container_width=True)
    else:
        st.info(f"Silakan pilih minimal 1 {comparison_type} untuk membandingkan.")

# TAB 5: Export
with tab5:
    st.header("📥 Export Data & Report")
    
    st.markdown("""
    Download data yang sudah difilter atau generate laporan lengkap dalam berbagai format.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Export Data")
        
        # CSV Download
        csv_data = convert_df_to_csv(filtered_df)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"layoffs_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Excel Download
        excel_data = convert_df_to_excel(filtered_df)
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"layoffs_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.markdown("---")
        
        st.info(f"**Total rows yang akan diexport:** {len(filtered_df):,}")
    
    with col2:
        st.subheader("📄 Data Preview")
        st.dataframe(filtered_df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Summary statistics
    st.subheader("📈 Statistik Deskriptif")
    
    descriptive_stats = filtered_df[['total_laid_off', 'percentage_laid_off', 'funds_raised_(millions$)']].describe()
    st.dataframe(descriptive_stats, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📊 Dashboard Analisis PHK Global (2020-2025)</p>
    <p>Dibuat dengan ❤️ menggunakan Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)


















