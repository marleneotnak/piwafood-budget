import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import io
import base64

# Konfigurera sidan
st.set_page_config(
    page_title="PiwaFood Budgetprognos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Anpassa utseendet
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
    }
    .stDownloadButton>button {
        background-color: #27ae60;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Huvudrubrik
st.markdown("<h1 class='main-header'>PiwaFood Budgetprognosverktyg</h1>", unsafe_allow_html=True)

# Sidofält för kontroller och filuppladdning
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Inställningar</h2>", unsafe_allow_html=True)
    
    st.markdown("### Ladda upp data")
    sales_file = st.file_uploader("Välj försäljningsdata (TSV)", type=["tsv", "txt"], key="sales_data")
    budget_file = st.file_uploader("Välj budgetdata (TSV)", type=["tsv", "txt"], key="budget_data")
    
    st.markdown("### Prognosalternativ")
    forecast_months = st.slider("Antal månader att prognostisera", min_value=1, max_value=12, value=3)
    model_choice = st.selectbox(
        "Välj prognosmodell",
        options=["Linjär Regression", "Random Forest"],
        index=0
    )
    
    include_seasonality = st.checkbox("Inkludera säsongsvariation", value=True)
    confidence_interval = st.checkbox("Visa konfidensintervall", value=True)
    
    st.markdown("### Visualiseringsalternativ")
    chart_type = st.selectbox(
        "Välj diagramtyp",
        options=["Linje", "Stapel", "Område", "Kombinerat"],
        index=0
    )
    
    color_theme = st.selectbox(
        "Välj färgtema",
        options=["Standard", "PiwaFood", "Blå-grön", "Varm", "Monokrom"],
        index=1
    )
    
    st.markdown("### Export")
    export_format = st.selectbox(
        "Välj exportformat",
        options=["Excel (.xlsx)", "CSV", "TSV"],
        index=0
    )

# Färger och teman
color_schemes = {
    "Standard": px.colors.qualitative.Plotly,
    "PiwaFood": ["#1e88e5", "#43a047", "#fb8c00", "#e53935", "#5e35b1"],
    "Blå-grön": px.colors.sequential.Viridis,
    "Varm": px.colors.sequential.Inferno,
    "Monokrom": px.colors.sequential.Greys
}

# Hjälpfunktioner
def parse_date(year_month):
    """Konvertera ÅrMånad (202101) till datetime-objekt."""
    if isinstance(year_month, str):
        if len(year_month) == 6:
            year = int(year_month[:4])
            month = int(year_month[4:])
            return datetime(year, month, 1)
    elif isinstance(year_month, (int, float)):
        year_month_str = str(int(year_month))
        if len(year_month_str) == 6:
            year = int(year_month_str[:4])
            month = int(year_month_str[4:])
            return datetime(year, month, 1)
    return None

def add_months(dt, months):
    """Lägg till månader till ett datetime-objekt."""
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    return datetime(year, month, 1)

def validate_data(df, required_columns, file_type):
    """Validera dataramens struktur."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Följande kolumner saknas i {file_type}: {', '.join(missing_columns)}"
    
    # Kontrollera om det finns tomma värden i viktiga kolumner
    for col in required_columns:
        if df[col].isna().any():
            return False, f"Det finns tomma värden i kolumnen '{col}' i {file_type}"
    
    return True, "Data validerad"

def get_table_download_link(df, filename, file_format):
    """Skapa en nedladdningslänk för dataframe."""
    if file_format == "Excel (.xlsx)":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Prognosdata')
        data = output.getvalue()
        file_ext = "xlsx"
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif file_format == "CSV":
        data = df.to_csv(index=False).encode('utf-8')
        file_ext = "csv"
        mime_type = "text/csv"
    else:  # TSV
        data = df.to_csv(index=False, sep='\t').encode('utf-8')
        file_ext = "tsv"
        mime_type = "text/tab-separated-values"
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}.{file_ext}">Ladda ner {file_format}</a>'
    return href

def format_currency(value):
    """Formatera valutavärden med tusentalsavgränsare."""
    return f"{value:,.0f}".replace(",", " ")

def prepare_forecast_data(sales_df, budget_df, forecast_months, model_type, include_seasonality):
    """Förbered och generera prognosdata."""
    # Förbered datan
    sales_df['Datum'] = sales_df['ÅrMånad'].apply(parse_date)
    sales_df['År'] = sales_df['Datum'].dt.year
    sales_df['Månad'] = sales_df['Datum'].dt.month
    
    # Skapa en dictionary för att lagra prognoser per säljkanal
    forecasts = {}
    forecast_details = {}
    
    # För varje säljkanal, skapa en prognos
    for channel in sales_df['Säljkanal'].unique():
        channel_data = sales_df[sales_df['Säljkanal'] == channel].copy()
        
        if len(channel_data) <= 1:
            continue
        
        # Sortera efter datum
        channel_data = channel_data.sort_values('Datum')
        
        # Skapa features
        X = np.arange(len(channel_data)).reshape(-1, 1)  # Tidstrend
        y = channel_data['Fakt. Belopp'].values
        
        if include_seasonality and len(channel_data) >= 12:
            # Lägg till månad som feature för säsongsvariation
            month_encoder = OneHotEncoder(sparse_output=False, drop='first')
            X_month = month_encoder.fit_transform(channel_data['Månad'].values.reshape(-1, 1))
            X = np.hstack((X, X_month))
        
        # Välj och träna modell
        if model_type == "Linjär Regression":
            model = LinearRegression()
        else:  # Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        # Skapa prognos för kommande månader
        last_date = channel_data['Datum'].iloc[-1]
        last_x_value = len(channel_data) - 1
        
        forecast_dates = [add_months(last_date, i+1) for i in range(forecast_months)]
        forecast_x = np.arange(last_x_value + 1, last_x_value + forecast_months + 1).reshape(-1, 1)
        
        if include_seasonality and len(channel_data) >= 12:
            forecast_months_values = np.array([d.month for d in forecast_dates]).reshape(-1, 1)
            forecast_months_encoded = month_encoder.transform(forecast_months_values)
            forecast_x = np.hstack((forecast_x, forecast_months_encoded))
        
        forecast_y = model.predict(forecast_x)
        
        # Beräkna konfidensintervall för linjär regression
        prediction_intervals = None
        if model_type == "Linjär Regression" and len(channel_data) > 2:
            y_pred = model.predict(X)
            residuals = y - y_pred
            mse = np.mean(residuals ** 2)
            std_err = np.sqrt(mse)
            
            # Enkel uppskattning av 95% konfidensintervall
            lower_bound = forecast_y - 1.96 * std_err
            upper_bound = forecast_y + 1.96 * std_err
            lower_bound = np.maximum(lower_bound, 0)  # Inga negativa försäljningsvärden
            
            prediction_intervals = {
                'lower': lower_bound,
                'upper': upper_bound
            }
        
        # Skapa prognosdata
        forecast_df = pd.DataFrame({
            'Säljkanal': channel,
            'Datum': forecast_dates,
            'ÅrMånad': [d.strftime('%Y%m') for d in forecast_dates],
            'Prognos Belopp': forecast_y
        })
        
        forecasts[channel] = forecast_df
        forecast_details[channel] = {
            'intervals': prediction_intervals,
            'model': model,
            'last_actual_date': last_date,
            'last_actual_value': channel_data['Fakt. Belopp'].iloc[-1]
        }
    
    # Kombinera alla prognoser
    if forecasts:
        combined_forecast = pd.concat(list(forecasts.values()), ignore_index=True)
        
        # Lägg till budget om den finns
        if budget_df is not None:
            budget_forecast = []
            
            for channel in combined_forecast['Säljkanal'].unique():
                channel_forecast = combined_forecast[combined_forecast['Säljkanal'] == channel]
                
                for _, row in channel_forecast.iterrows():
                    year_month = row['ÅrMånad']
                    year = int(year_month[:4])
                    month = int(year_month[4:])
                    
                    # Hitta motsvarande budget
                    budget_value = None
                    channel_budget = budget_df[budget_df['Säljkanal'] == channel]
                    
                    if not channel_budget.empty:
                        month_col = f'Månad {month}'
                        if month_col in channel_budget.columns:
                            budget_row = channel_budget[channel_budget['År'] == year]
                            if not budget_row.empty:
                                budget_value = budget_row[month_col].iloc[0]
                    
                    budget_forecast.append({
                        'Säljkanal': channel,
                        'ÅrMånad': year_month,
                        'Datum': row['Datum'],
                        'Prognos Belopp': row['Prognos Belopp'],
                        'Budget': budget_value
                    })
            
            combined_forecast = pd.DataFrame(budget_forecast)
        
        return combined_forecast, forecast_details
    else:
        return None, None

def visualize_data(sales_df, forecast_df, forecast_details, show_confidence, chart_type, colors):
    """Skapa visualiseringar av historisk data och prognoser."""
    tabs = st.tabs(["Översikt", "Per säljkanal", "Detaljer", "Data"])
    
    with tabs[0]:
        st.markdown("<h3 class='sub-header'>Översikt över alla säljkanaler</h3>", unsafe_allow_html=True)
        
        # Aggregera all försäljningsdata per månad
        agg_sales = sales_df.groupby('Datum')['Fakt. Belopp'].sum().reset_index()
        
        # Aggregera all prognosdata per månad
        if forecast_df is not None:
            agg_forecast = forecast_df.groupby('Datum')['Prognos Belopp'].sum().reset_index()
            agg_budget = None
            if 'Budget' in forecast_df.columns:
                agg_budget = forecast_df.groupby('Datum')['Budget'].sum().reset_index()
        
        fig = go.Figure()
        
        # Lägg till historisk försäljning
        fig.add_trace(
            go.Scatter(
                x=agg_sales['Datum'],
                y=agg_sales['Fakt. Belopp'],
                mode='lines+markers',
                name='Historisk försäljning',
                line=dict(color=colors[0], width=2),
                marker=dict(size=8)
            )
        )
        
        # Lägg till prognos
        if forecast_df is not None:
            fig.add_trace(
                go.Scatter(
                    x=agg_forecast['Datum'],
                    y=agg_forecast['Prognos Belopp'],
                    mode='lines+markers',
                    name='Prognos',
                    line=dict(color=colors[1], width=2, dash='dash'),
                    marker=dict(size=8)
                )
            )
            
            # Lägg till budget om den finns
            if agg_budget is not None:
                fig.add_trace(
                    go.Scatter(
                        x=agg_budget['Datum'],
                        y=agg_budget['Budget'],
                        mode='lines',
                        name='Budget',
                        line=dict(color=colors[2], width=2, dash='dot')
                    )
                )
        
        fig.update_layout(
            title='Total försäljning och prognos',
            xaxis_title='Datum',
            yaxis_title='Belopp (SEK)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="x unified"
        )
        
        fig.update_yaxes(tickformat=",", ticksuffix=" kr")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Visa sammanfattningsstatistik
        st.markdown("<h4>Sammanfattning</h4>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        total_hist_sales = agg_sales['Fakt. Belopp'].sum()
        
        with col1:
            st.metric(
                label="Total historisk försäljning", 
                value=f"{format_currency(total_hist_sales)} kr"
            )
        
        if forecast_df is not None:
            total_forecast = agg_forecast['Prognos Belopp'].sum()
            forecast_growth = ((total_forecast / (total_hist_sales / len(agg_sales) * len(agg_forecast))) - 1) * 100
            
            with col2:
                st.metric(
                    label="Total prognostiserad försäljning", 
                    value=f"{format_currency(total_forecast)} kr",
                    delta=f"{forecast_growth:.1f}%"
                )
            
            if agg_budget is not None:
                total_budget = agg_budget['Budget'].sum()
                budget_diff = ((total_forecast / total_budget) - 1) * 100
                
                with col3:
                    st.metric(
                        label="Total budget", 
                        value=f"{format_currency(total_budget)} kr",
                        delta=f"{budget_diff:.1f}% vs prognos"
                    )
    
    with tabs[1]:
        st.markdown("<h3 class='sub-header'>Analys per säljkanal</h3>", unsafe_allow_html=True)
        
        # Lista över alla säljkanaler
        channels = sorted(sales_df['Säljkanal'].unique())
        selected_channel = st.selectbox("Välj säljkanal", channels)
        
        # Filtrera data för vald säljkanal
        channel_sales = sales_df[sales_df['Säljkanal'] == selected_channel]
        channel_sales = channel_sales.sort_values('Datum')
        
        if forecast_df is not None:
            channel_forecast = forecast_df[forecast_df['Säljkanal'] == selected_channel]
            channel_forecast = channel_forecast.sort_values('Datum')
        
        fig = go.Figure()
        
        # Välj typ av diagram
        if chart_type == "Stapel":
            # Historisk försäljning som staplar
            fig.add_trace(
                go.Bar(
                    x=channel_sales['Datum'],
                    y=channel_sales['Fakt. Belopp'],
                    name='Historisk försäljning',
                    marker_color=colors[0]
                )
            )
            
            # Prognos som staplar
            if forecast_df is not None and not channel_forecast.empty:
                first_forecast = channel_forecast['Prognos Belopp'].iloc[0]
                forecast_vs_latest = ((first_forecast / latest_value) - 1) * 100 if latest_value > 0 else 0
                
                with col2:
                    st.metric(
                        label="Första prognosperiodens försäljning", 
                        value=f"{format_currency(first_forecast)} kr",
                        delta=f"{forecast_vs_latest:.1f}% vs senaste faktiska"
                    )
                
                if 'Budget' in channel_forecast.columns and not channel_forecast['Budget'].isna().all():
                    first_budget = channel_forecast['Budget'].iloc[0]
                    budget_diff = ((first_forecast / first_budget) - 1) * 100 if first_budget > 0 else 0
                    
                    with col3:
                        st.metric(
                            label="Budget för första prognosperioden", 
                            value=f"{format_currency(first_budget)} kr",
                            delta=f"{budget_diff:.1f}% vs prognos"
                        )
        
        # Visa säsongsvariation
        if len(channel_sales) >= 12:
            st.markdown("<h4>Säsongsvariation</h4>", unsafe_allow_html=True)
            
            # Gruppera efter månad och beräkna genomsnitt
            channel_sales['Månad'] = channel_sales['Datum'].dt.month
            monthly_avg = channel_sales.groupby('Månad')['Fakt. Belopp'].mean().reset_index()
            
            # Skapa ett stapeldiagram för säsongsvariation
            fig_seasonal = go.Figure()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
            
            fig_seasonal.add_trace(
                go.Bar(
                    x=[month_names[m-1] for m in monthly_avg['Månad']],
                    y=monthly_avg['Fakt. Belopp'],
                    marker_color=colors[0]
                )
            )
            
            fig_seasonal.update_layout(
                title='Genomsnittlig månadsförsäljning',
                xaxis_title='Månad',
                yaxis_title='Genomsnittligt belopp (SEK)',
                height=400,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            fig_seasonal.update_yaxes(tickformat=",", ticksuffix=" kr")
            
            st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with tabs[2]:
        st.markdown("<h3 class='sub-header'>Detaljerad analys</h3>", unsafe_allow_html=True)
        
        if forecast_df is not None and forecast_details:
            # Visa prestandamått för modellen
            st.markdown("<h4>Modellprestanda</h4>", unsafe_allow_html=True)
            
            # För varje säljkanal, beräkna och visa modellprestanda
            performance_data = []
            
            for channel in forecast_details.keys():
                if channel in sales_df['Säljkanal'].unique():
                    channel_sales = sales_df[sales_df['Säljkanal'] == channel]
                    
                    if len(channel_sales) > 3:  # Minst 4 datapunkter för att beräkna prestanda
                        # Använd de senaste 3 månaderna som testdata
                        test_size = min(3, len(channel_sales) // 3)
                        train_data = channel_sales.iloc[:-test_size]
                        test_data = channel_sales.iloc[-test_size:]
                        
                        # Skapa features för träningsdata
                        X_train = np.arange(len(train_data)).reshape(-1, 1)
                        y_train = train_data['Fakt. Belopp'].values
                        
                        # Skapa samma modell som användes för prognos
                        if model_choice == "Linjär Regression":
                            model = LinearRegression()
                        else:  # Random Forest
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        
                        if include_seasonality and len(train_data) >= 12:
                            month_encoder = OneHotEncoder(sparse_output=False, drop='first')
                            X_month = month_encoder.fit_transform(train_data['Månad'].values.reshape(-1, 1))
                            X_train = np.hstack((X_train, X_month))
                            
                            model.fit(X_train, y_train)
                            
                            # Skapa features för testdata
                            X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
                            X_test_month = month_encoder.transform(test_data['Månad'].values.reshape(-1, 1))
                            X_test = np.hstack((X_test, X_test_month))
                        else:
                            model.fit(X_train, y_train)
                            
                            # Skapa features för testdata
                            X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
                        
                        # Gör prognos på testdata
                        y_pred = model.predict(X_test)
                        
                        # Beräkna felprocent
                        y_test = test_data['Fakt. Belopp'].values
                        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                        
                        performance_data.append({
                            'Säljkanal': channel,
                            'MAPE (%)': mape,
                            'Testperioder': test_size
                        })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                perf_df = perf_df.sort_values('MAPE (%)')
                
                # Skapa en tabell för modellprestanda
                fig_perf = go.Figure(data=[
                    go.Bar(
                        x=perf_df['Säljkanal'],
                        y=perf_df['MAPE (%)'],
                        marker_color=colors[0]
                    )
                ])
                
                fig_perf.update_layout(
                    title='Modellfel per säljkanal (lägre är bättre)',
                    xaxis_title='Säljkanal',
                    yaxis_title='MAPE (%)',
                    height=400,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Förklaring av metrik
                st.markdown("""
                <div class='info-box'>
                <strong>MAPE (Mean Absolute Percentage Error)</strong>: Genomsnittligt procentuellt fel i prognosen.
                Ett lägre värde indikerar en mer exakt prognos. Beräkningen baseras på att förutsäga de senaste perioderna
                och jämföra med faktiskt utfall.
                </div>
                """, unsafe_allow_html=True)
            
            # Visa prognosrisker
            st.markdown("<h4>Prognosrisker</h4>", unsafe_allow_html=True)
            
            # Identifiera kanaler med hög volatilitet
            volatility_data = []
            
            for channel in sales_df['Säljkanal'].unique():
                channel_sales = sales_df[sales_df['Säljkanal'] == channel].copy()
                
                if len(channel_sales) >= 3:
                    channel_sales = channel_sales.sort_values('Datum')
                    
                    # Beräkna procentuell förändring från månad till månad
                    channel_sales['Pct_Change'] = channel_sales['Fakt. Belopp'].pct_change()
                    
                    # Exkludera första raden som är NaN
                    volatility = channel_sales['Pct_Change'].iloc[1:].std() * 100  # Standardavvikelse i procent
                    
                    volatility_data.append({
                        'Säljkanal': channel,
                        'Volatilitet (%)': volatility
                    })
            
            if volatility_data:
                vol_df = pd.DataFrame(volatility_data)
                vol_df = vol_df.sort_values('Volatilitet (%)', ascending=False)
                
                # Skapa en tabell för volatilitet
                fig_vol = go.Figure(data=[
                    go.Bar(
                        x=vol_df['Säljkanal'],
                        y=vol_df['Volatilitet (%)'],
                        marker_color=colors[2]
                    )
                ])
                
                fig_vol.update_layout(
                    title='Volatilitet per säljkanal',
                    xaxis_title='Säljkanal',
                    yaxis_title='Volatilitet (%)',
                    height=400,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Förklaring av metrik
                st.markdown("""
                <div class='warning-box'>
                <strong>Volatilitet</strong>: Standardavvikelse av procentuell förändring månad till månad.
                Högre volatilitet indikerar större osäkerhet i prognosen. Var extra uppmärksam på kanaler med hög volatilitet.
                </div>
                """, unsafe_allow_html=True)
            
            # Visa riskklassificering för prognoserna
            st.markdown("<h4>Riskklassificering av prognoser</h4>", unsafe_allow_html=True)
            
            risk_data = []
            
            if performance_data and volatility_data:
                perf_dict = {row['Säljkanal']: row['MAPE (%)'] for row in performance_data}
                vol_dict = {row['Säljkanal']: row['Volatilitet (%)'] for row in volatility_data}
                
                for channel in forecast_details.keys():
                    if channel in perf_dict and channel in vol_dict:
                        mape = perf_dict[channel]
                        volatility = vol_dict[channel]
                        
                        # Beräkna risk baserat på MAPE och volatilitet
                        if mape < 10 and volatility < 15:
                            risk_level = "Låg"
                            risk_color = "green"
                        elif mape > 25 or volatility > 30:
                            risk_level = "Hög"
                            risk_color = "red"
                        else:
                            risk_level = "Medel"
                            risk_color = "orange"
                        
                        risk_data.append({
                            'Säljkanal': channel,
                            'Risknivå': risk_level,
                            'MAPE (%)': mape,
                            'Volatilitet (%)': volatility,
                            'Färg': risk_color
                        })
            
            if risk_data:
                risk_df = pd.DataFrame(risk_data)
                
                # Sortera efter risknivå
                risk_level_order = {"Hög": 0, "Medel": 1, "Låg": 2}
                risk_df['RiskOrdning'] = risk_df['Risknivå'].map(risk_level_order)
                risk_df = risk_df.sort_values('RiskOrdning')
                
                # Skapa en tabell
                risk_table = pd.DataFrame({
                    'Säljkanal': risk_df['Säljkanal'],
                    'Risknivå': risk_df['Risknivå'],
                    'MAPE (%)': risk_df['MAPE (%)'].round(1),
                    'Volatilitet (%)': risk_df['Volatilitet (%)'].round(1)
                })
                
                # Konvertera DataFrame till HTML med färgad risknivå
                html_table = risk_table.to_html(index=False, escape=False)
                
                # Ersätt texten för risknivå med färgad version
                for i, row in risk_df.iterrows():
                    risk_text = row['Risknivå']
                    color = row['Färg']
                    html_table = html_table.replace(
                        f">{risk_text}<", 
                        f"><span style='color:{color}; font-weight:bold'>{risk_text}</span><"
                    )
                
                st.markdown(f"""
                <div style='overflow-x: auto;'>
                {html_table}
                </div>
                """, unsafe_allow_html=True)
                
                # Förklaring av riskklassificering
                st.markdown("""
                <div class='info-box'>
                <strong>Riskklassificering</strong>:
                <ul>
                <li><span style='color:green; font-weight:bold'>Låg risk</span>: Stabil historik, tillförlitlig prognos.</li>
                <li><span style='color:orange; font-weight:bold'>Medel risk</span>: Viss osäkerhet i prognosen.</li>
                <li><span style='color:red; font-weight:bold'>Hög risk</span>: Hög osäkerhet, bör tolkas med försiktighet.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Generera en prognos först för att se detaljerad analys.")
    
    with tabs[3]:
        st.markdown("<h3 class='sub-header'>Datatabell</h3>", unsafe_allow_html=True)
        
        data_tabs = st.tabs(["Historisk data", "Prognosdata"])
        
        with data_tabs[0]:
            # Visa historisk data
            st.dataframe(
                sales_df.sort_values(['Säljkanal', 'Datum']),
                use_container_width=True,
                hide_index=True
            )
            
            # Exportknapp för historisk data
            if not sales_df.empty:
                export_filename = "PiwaFood_Historisk_Forsaljning"
                st.markdown(
                    get_table_download_link(sales_df, export_filename, export_format),
                    unsafe_allow_html=True
                )
        
        with data_tabs[1]:
            # Visa prognosdata om den finns
            if forecast_df is not None:
                st.dataframe(
                    forecast_df.sort_values(['Säljkanal', 'Datum']),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Exportknapp för prognosdata
                export_filename = "PiwaFood_Prognos"
                st.markdown(
                    get_table_download_link(forecast_df, export_filename, export_format),
                    unsafe_allow_html=True
                )
            else:
                st.info("Ingen prognosdata tillgänglig ännu.")


# Huvudfunktion
def main():
    # Initiera sessionsvariabler om de inte finns
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'sales_df' not in st.session_state:
        st.session_state.sales_df = None
    if 'budget_df' not in st.session_state:
        st.session_state.budget_df = None
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = None
    if 'forecast_details' not in st.session_state:
        st.session_state.forecast_details = None
    
    # Container för meddelanden
    message_container = st.container()
    
    # Hantera filuppladdningar
    if sales_file:
        try:
            sales_data = pd.read_csv(sales_file, sep='\t', encoding='utf-8')
            
            # Validera försäljningsdata
            required_cols = ['Säljkanal', 'ÅrMånad', 'Fakt. Belopp']
            valid, message = validate_data(sales_data, required_cols, "försäljningsdata")
            
            if valid:
                # Konvertera ÅrMånad till rätt format och skapa datum
                sales_data['ÅrMånad'] = sales_data['ÅrMånad'].astype(str)
                sales_data['Datum'] = sales_data['ÅrMånad'].apply(parse_date)
                
                # Konvertera numeriska kolumner
                for col in ['Fakt. Belopp', 'Fakt. TB']:
                    if col in sales_data.columns:
                        sales_data[col] = pd.to_numeric(sales_data[col], errors='coerce')
                
                # Sortera och spara data
                sales_data = sales_data.sort_values(['Säljkanal', 'Datum'])
                st.session_state.sales_df = sales_data
                st.session_state.data_loaded = True
                
                with message_container:
                    st.success(f"Försäljningsdata laddad med {len(sales_data)} rader och {len(sales_data['Säljkanal'].unique())} säljkanaler.")
            else:
                with message_container:
                    st.error(message)
        except Exception as e:
            with message_container:
                st.error(f"Fel vid inläsning av försäljningsdata: {str(e)}")
    
    if budget_file:
        try:
            budget_data = pd.read_csv(budget_file, sep='\t', encoding='utf-8')
            
            # Validera budgetdata
            required_cols = ['Säljkanal', 'År']
            valid, message = validate_data(budget_data, required_cols, "budgetdata")
            
            if valid:
                # Konvertera År till heltal
                budget_data['År'] = budget_data['År'].astype(int)
                
                # Spara data
                st.session_state.budget_df = budget_data
                
                with message_container:
                    st.success(f"Budgetdata laddad med {len(budget_data)} rader och {len(budget_data['Säljkanal'].unique())} säljkanaler.")
            else:
                with message_container:
                    st.error(message)
        except Exception as e:
            with message_container:
                st.error(f"Fel vid inläsning av budgetdata: {str(e)}")
    
    # Databearbetning och visualisering
    if st.session_state.data_loaded:
        sales_df = st.session_state.sales_df
        budget_df = st.session_state.budget_df
        
        # Skapa en centrerad knapp för att generera prognos
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_forecast = st.button("Generera försäljningsprognos", use_container_width=True)
        
        if generate_forecast:
            try:
                with st.spinner("Genererar prognos..."):
                    forecast_df, forecast_details = prepare_forecast_data(
                        sales_df, 
                        budget_df, 
                        forecast_months, 
                        model_choice, 
                        include_seasonality
                    )
                
                if forecast_df is not None:
                    st.session_state.forecast_df = forecast_df
                    st.session_state.forecast_details = forecast_details
                    
                    with message_container:
                        st.success(f"Prognos genererad för {forecast_months} månader framåt!")
                else:
                    with message_container:
                        st.warning("Kunde inte generera prognos. Kontrollera att data innehåller tillräckligt med historik.")
            except Exception as e:
                with message_container:
                    st.error(f"Fel vid generering av prognos: {str(e)}")
        
        # Visa data och prognoser
        visualize_data(
            sales_df, 
            st.session_state.forecast_df, 
            st.session_state.forecast_details, 
            confidence_interval, 
            chart_type, 
            color_schemes[color_theme]
        )
    else:
        # Visa instruktioner om ingen data laddats
        st.markdown("""
        <div class='info-box'>
        <h3>Välkommen till PiwaFood Budgetprognosverktyg!</h3>
        <p>För att komma igång:</p>
        <ol>
        <li>Ladda upp försäljningsdata (TSV-format) från Qlik Sense via sidofältet.</li>
        <li>Valfritt: Ladda upp budgetdata (TSV-format) för jämförelse.</li>
        <li>Justera inställningarna för prognosen i sidofältet.</li>
        <li>Klicka på "Generera försäljningsprognos".</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Visa exempel på förväntad datastruktur
        st.markdown("<h3 class='sub-header'>Förväntad datastruktur</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <h4>Försäljningsdata (obligatorisk)</h4>
        <p>TSV-format med följande kolumner:</p>
        <ul>
        <li><strong>Säljkanal</strong>: Namn på säljkanalen</li>
        <li><strong>ÅrMånad</strong>: Period i formatet YYYYMM (ex. 202101)</li>
        <li><strong>Fakt. Belopp</strong>: Faktisk försäljning i SEK</li>
        <li><strong>Fakt. TB</strong>: Faktiskt täckningsbidrag (valfritt)</li>
        </ul>
        </div>
        
        <div class='info-box'>
        <h4>Budgetdata (valfri)</h4>
        <p>TSV-format med följande kolumner:</p>
        <ul>
        <li><strong>Säljkanal</strong>: Namn på säljkanalen</li>
        <li><strong>År</strong>: Budgetår (ex. 2021)</li>
        <li><strong>Månad 1</strong> till <strong>Månad 12</strong>: Budgetvärden för respektive månad</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main():
                fig.add_trace(
                    go.Bar(
                        x=channel_forecast['Datum'],
                        y=channel_forecast['Prognos Belopp'],
                        name='Prognos',
                        marker_color=colors[1]
                    )
                )
                
                # Budget som linje
                if 'Budget' in channel_forecast.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=channel_forecast['Datum'],
                            y=channel_forecast['Budget'],
                            mode='lines+markers',
                            name='Budget',
                            line=dict(color=colors[2], width=2)
                        )
                    )
        
        elif chart_type == "Område":
            # Historisk försäljning som område
            fig.add_trace(
                go.Scatter(
                    x=channel_sales['Datum'],
                    y=channel_sales['Fakt. Belopp'],
                    name='Historisk försäljning',
                    fill='tozeroy',
                    line=dict(color=colors[0], width=0),
                    fillcolor=f'rgba({int(colors[0][1:3], 16)}, {int(colors[0][3:5], 16)}, {int(colors[0][5:7], 16)}, 0.5)'
                )
            )
            
            # Prognos som område
            if forecast_df is not None:
                fig.add_trace(
                    go.Scatter(
                        x=channel_forecast['Datum'],
                        y=channel_forecast['Prognos Belopp'],
                        name='Prognos',
                        fill='tozeroy',
                        line=dict(color=colors[1], width=0),
                        fillcolor=f'rgba({int(colors[1][1:3], 16)}, {int(colors[1][3:5], 16)}, {int(colors[1][5:7], 16)}, 0.5)'
                    )
                )
                
                # Budget som linje
                if 'Budget' in channel_forecast.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=channel_forecast['Datum'],
                            y=channel_forecast['Budget'],
                            mode='lines',
                            name='Budget',
                            line=dict(color=colors[2], width=2, dash='dot')
                        )
                    )
        
        elif chart_type == "Kombinerat":
            # Historisk försäljning som staplar
            fig.add_trace(
                go.Bar(
                    x=channel_sales['Datum'],
                    y=channel_sales['Fakt. Belopp'],
                    name='Historisk försäljning',
                    marker_color=colors[0]
                )
            )
            
            # Prognos som staplar
            if forecast_df is not None:
                fig.add_trace(
                    go.Bar(
                        x=channel_forecast['Datum'],
                        y=channel_forecast['Prognos Belopp'],
                        name='Prognos',
                        marker_color=colors[1]
                    )
                )
                
                # Glidande medelvärde som linje
                sales_values = list(channel_sales['Fakt. Belopp'])
                forecast_values = list(channel_forecast['Prognos Belopp'])
                all_values = sales_values + forecast_values
                
                window_size = min(3, len(sales_values))
                if window_size > 1:
                    rolling_avg = []
                    for i in range(len(sales_values)):
                        if i < window_size - 1:
                            # För de första punkterna, använd tillgängliga värden
                            avg = sum(sales_values[:i+1]) / (i+1)
                        else:
                            # För resten, använd glidande fönster
                            avg = sum(sales_values[i-(window_size-1):i+1]) / window_size
                        rolling_avg.append(avg)
                    
                    # Lägg till glidande medelvärde för prognosen
                    for i in range(len(forecast_values)):
                        if i == 0:
                            # För första prognosvärdet, använd de sista faktiska värdena plus prognosen
                            last_actual = sales_values[-window_size+1:] if window_size > 1 else []
                            window = last_actual + [forecast_values[0]]
                            avg = sum(window) / len(window)
                        else:
                            # För resterande prognosvärden
                            values_to_use = []
                            for j in range(window_size):
                                idx = i - j
                                if idx >= 0:
                                    values_to_use.append(forecast_values[idx])
                                else:
                                    if abs(idx) <= len(sales_values):
                                        values_to_use.append(sales_values[idx])
                            avg = sum(values_to_use) / len(values_to_use)
                        rolling_avg.append(avg)
                    
                    # Kombinera datum för både historisk och prognos
                    all_dates = list(channel_sales['Datum']) + list(channel_forecast['Datum'])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=all_dates,
                            y=rolling_avg,
                            mode='lines',
                            name=f'{window_size}-månaders glidande medelvärde',
                            line=dict(color=colors[3], width=2)
                        )
                    )
                
                # Budget som linje
                if 'Budget' in channel_forecast.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=channel_forecast['Datum'],
                            y=channel_forecast['Budget'],
                            mode='lines+markers',
                            name='Budget',
                            line=dict(color=colors[2], width=2, dash='dot')
                        )
                    )
        
        else:  # Linje (standard)
            # Historisk försäljning som linje
            fig.add_trace(
                go.Scatter(
                    x=channel_sales['Datum'],
                    y=channel_sales['Fakt. Belopp'],
                    mode='lines+markers',
                    name='Historisk försäljning',
                    line=dict(color=colors[0], width=2),
                    marker=dict(size=8)
                )
            )
            
            # Prognos som linje
            if forecast_df is not None:
                fig.add_trace(
                    go.Scatter(
                        x=channel_forecast['Datum'],
                        y=channel_forecast['Prognos Belopp'],
                        mode='lines+markers',
                        name='Prognos',
                        line=dict(color=colors[1], width=2, dash='dash'),
                        marker=dict(size=8)
                    )
                )
                
                # Visa konfidensintervall om det finns och är begärt
                if show_confidence and selected_channel in forecast_details and forecast_details[selected_channel]['intervals'] is not None:
                    intervals = forecast_details[selected_channel]['intervals']
                    
                    fig.add_trace(
                        go.Scatter(
                            x=channel_forecast['Datum'],
                            y=intervals['upper'],
                            mode='lines',
                            name='Övre konfidensintervall',
                            line=dict(color=colors[1], width=0),
                            showlegend=False
                        )
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=channel_forecast['Datum'],
                            y=intervals['lower'],
                            mode='lines',
                            name='Nedre konfidensintervall',
                            line=dict(color=colors[1], width=0),
                            fill='tonexty',
                            fillcolor=f'rgba({int(colors[1][1:3], 16)}, {int(colors[1][3:5], 16)}, {int(colors[1][5:7], 16)}, 0.2)',
                            showlegend=False
                        )
                    )
                
                # Budget som linje
                if 'Budget' in channel_forecast.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=channel_forecast['Datum'],
                            y=channel_forecast['Budget'],
                            mode='lines',
                            name='Budget',
                            line=dict(color=colors[2], width=2, dash='dot')
                        )
                    )
        
        fig.update_layout(
            title=f'Försäljning och prognos för {selected_channel}',
            xaxis_title='Datum',
            yaxis_title='Belopp (SEK)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="x unified"
        )
        
        fig.update_yaxes(tickformat=",", ticksuffix=" kr")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Visa trendanalys för den valda säljkanalen
        st.markdown("<h4>Trendanalys</h4>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        # Beräkna statistik
        if len(channel_sales) >= 2:
            latest_value = channel_sales['Fakt. Belopp'].iloc[-1]
            previous_value = channel_sales['Fakt. Belopp'].iloc[-2]
            change_vs_prev = ((latest_value / previous_value) - 1) * 100 if previous_value > 0 else 0
            
            with col1:
                st.metric(
                    label="Senaste periodens försäljning", 
                    value=f"{format_currency(latest_value)} kr",
                    delta=f"{change_vs_prev:.1f}% vs föregående"
                )
            
            if forecast_df is not None
