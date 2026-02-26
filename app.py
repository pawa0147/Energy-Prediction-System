from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import joblib, os, warnings
from datetime import datetime, date
import requests

warnings.filterwarnings('ignore')
app = Flask(__name__)
BASE = os.path.dirname(__file__)

# ── Hardcoded metrics from energy2.ipynb ──────────────────────────────────────
METRICS = [
    {
        'name':'Ridge Regression','alpha':'α = 10',
        'train_r2':0.9867,'test_r2':0.9752,
        'train_mae':5.3410,'test_mae':9.7104,
        'train_mse':161.73,'test_mse':398.80,
        'train_rmse':12.7174,'test_rmse':19.9699,
        'mape':10.82,'zeroed':[],'note':'Pre-trained · ridge_model.pkl',
    },
    {
        'name':'Lasso Regression','alpha':'α = 1',
        'train_r2':0.9867,'test_r2':0.9751,
        'train_mae':5.2674,'test_mae':9.6652,
        'train_mse':161.94,'test_mse':400.07,
        'train_rmse':12.7256,'test_rmse':20.0017,
        'mape':9.16,
        'zeroed':['latitude','longitude','temperature','DayOfWeek','IsWeekend','Month','Quarter'],
        'note':'7 features zeroed out',
    },
    {
        'name':'Random Forest','alpha':'n=200 d=10',
        'train_r2':0.9935,'test_r2':0.9548,
        'train_mae':4.2172,'test_mae':14.2155,
        'train_mse':79.37,'test_mse':726.41,
        'train_rmse':8.9088,'test_rmse':26.9520,
        'mape':11.86,'zeroed':[],'note':'Mild overfit · R² gap 0.039',
    },
]

FEATURE_IMPORTANCE = [
    {'feature':'Lag1','importance':0.3067},
    {'feature':'Rolling7','importance':0.1813},
    {'feature':'Rolling14','importance':0.1427},
    {'feature':'Rolling30','importance':0.1094},
    {'feature':'Lag7','importance':0.0901},
    {'feature':'Lag14','importance':0.0889},
    {'feature':'longitude','importance':0.0434},
    {'feature':'RollingStd7','importance':0.0229},
    {'feature':'latitude','importance':0.0102},
    {'feature':'humidity','importance':0.0018},
    {'feature':'temperature','importance':0.0013},
    {'feature':'Day','importance':0.0006},
    {'feature':'Month','importance':0.0003},
    {'feature':'DayOfWeek','importance':0.0002},
    {'feature':'Quarter','importance':0.0001},
    {'feature':'IsWeekend','importance':0.0000},
]

FEATURES = ['latitude','longitude','temperature','humidity','DayOfWeek',
            'IsWeekend','Day','Month','Quarter','Lag1','Lag7','Lag14',
            'Rolling7','Rolling14','Rolling30','RollingStd7']

# ── State → lat/lon map ───────────────────────────────────────────────────────
STATE_COORDS = {
    'Andhra Pradesh':   {'lat':14.750429, 'lon':78.570026},
    'Arunachal Pradesh':{'lat':27.100399, 'lon':93.616601},
    'Assam':            {'lat':26.749981, 'lon':94.216667},
    'Bihar':            {'lat':25.785414, 'lon':87.479973},
    'Chandigarh':       {'lat':30.719997, 'lon':76.780006},
    'Chhattisgarh':     {'lat':22.090420, 'lon':82.159987},
    'DNH':              {'lat':20.266578, 'lon':73.016618},
    'Delhi':            {'lat':28.669993, 'lon':77.230004},
    'Goa':              {'lat':15.491997, 'lon':73.818001},
    'Gujarat':          {'lat':22.258700, 'lon':71.192400},
    'HP':               {'lat':31.100025, 'lon':77.166597},
    'Haryana':          {'lat':28.450006, 'lon':77.019991},
    'J&K':              {'lat':33.450000, 'lon':76.240000},
    'Jharkhand':        {'lat':23.800393, 'lon':86.419986},
    'Karnataka':        {'lat':12.570381, 'lon':76.919997},
    'Kerala':           {'lat':8.900373,  'lon':76.569993},
    'MP':               {'lat':21.300391, 'lon':76.130019},
    'Maharashtra':      {'lat':19.250232, 'lon':73.160175},
    'Manipur':          {'lat':24.799971, 'lon':93.950017},
    'Meghalaya':        {'lat':25.570492, 'lon':91.880014},
    'Mizoram':          {'lat':23.710399, 'lon':92.720015},
    'Nagaland':         {'lat':25.666998, 'lon':94.116570},
    'Odisha':           {'lat':19.820430, 'lon':85.900017},
    'Pondy':            {'lat':11.934994, 'lon':79.830000},
    'Punjab':           {'lat':31.519974, 'lon':75.980003},
    'Rajasthan':        {'lat':26.449999, 'lon':74.639981},
    'Sikkim':           {'lat':27.333330, 'lon':88.616647},
    'Tamil Nadu':       {'lat':12.920386, 'lon':79.150042},
    'Telangana':        {'lat':18.112400, 'lon':79.019300},
    'Tripura':          {'lat':23.835404, 'lon':91.279999},
    'UP':               {'lat':27.599981, 'lon':78.050006},
    'Uttarakhand':      {'lat':30.320409, 'lon':78.050006},
    'West Bengal':      {'lat':22.580390, 'lon':88.329947},
}

# ── Load data & ridge model ───────────────────────────────────────────────────
df = None
ridge_model = None

def load_all():
    global df, ridge_model
    csv = os.path.join(BASE, 'energy_with_features.csv')
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        df['Dates'] = pd.to_datetime(df['Dates'])
        print(f"[INFO] Data loaded: {df.shape}")
    for p in ('models/ridge_model.pkl', 'ridge_model.pkl'):
        full = os.path.join(BASE, p)
        if os.path.exists(full):
            ridge_model = joblib.load(full)
            print(f"[INFO] Ridge loaded from {full}")
            break

load_all()

# ── Page routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():         return render_template('index.html')
@app.route('/insights')
def insights():      return render_template('insights.html')
@app.route('/states')
def states():        return render_template('states.html')
@app.route('/models')
def models_page():   return render_template('models.html')
@app.route('/predict')
def predict_page():  return render_template('predict.html')

# ── Data APIs ─────────────────────────────────────────────────────────────────
@app.route('/api/overview')
def api_overview():
    if df is None: return jsonify(error="Data not loaded"), 500
    return jsonify({
        'total_records': int(len(df)),
        'num_states':    int(df['States'].nunique()),
        'date_range':    f"{df['Dates'].min().strftime('%d %b %Y')} – {df['Dates'].max().strftime('%d %b %Y')}",
        'avg_usage':     round(float(df['Usage'].mean()), 2),
        'max_usage':     round(float(df['Usage'].max()), 2),
        'min_usage':     round(float(df['Usage'].min()), 2),
        'total_usage':   round(float(df['Usage'].sum()), 2),
    })

@app.route('/api/timeseries')
def api_timeseries():
    state = request.args.get('state', 'All')
    sub = df if state == 'All' else df[df['States'] == state]
    ts  = sub.groupby('Dates')['Usage'].mean().reset_index().sort_values('Dates')
    return jsonify({
        'dates': ts['Dates'].dt.strftime('%Y-%m-%d').tolist(),
        'usage': [round(v, 2) for v in ts['Usage'].tolist()],
    })

@app.route('/api/monthly')
def api_monthly():
    state = request.args.get('state', 'Overall')
    sub = df if state in ['All', 'Overall'] else df[df['States'] == state]
    m_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
             7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly = sub.groupby('Month')['Usage'].mean().reset_index()
    return jsonify({
        'months': [m_map[m] for m in monthly['Month'].tolist()],
        'usage':  [round(v, 2) for v in monthly['Usage'].tolist()],
    })

@app.route('/api/weekday_weekend')
def api_weekday_weekend():
    grp = df.groupby('IsWeekend')['Usage'].mean()
    return jsonify({
        'labels': ['Weekday','Weekend'],
        'values': [round(float(grp.get(0,0)),2), round(float(grp.get(1,0)),2)],
    })

@app.route('/api/states')
def api_states():
    s = df.groupby('States')['Usage'].agg(['mean','sum','max','min']).reset_index()
    s.columns = ['state','avg','total','max','min']
    
    # Merge latitude and longitude from STATE_COORDS for the map
    records = s.round(2).to_dict(orient='records')
    for row in records:
        loc = STATE_COORDS.get(row['state'])
        if loc:
            row['lat'] = loc['lat']
            row['lon'] = loc['lon']
        else:
            row['lat'], row['lon'] = None, None
            
    return jsonify(records)

@app.route('/api/temp_vs_usage')
def api_temp_vs_usage():
    # group by rounded temperature to downsample slightly and average the usage
    if 'temperature' not in df.columns:
        return jsonify(error="Temperature data missing"), 400
    sub = df.dropna(subset=['temperature', 'Usage'])
    sub['temp_rounded'] = sub['temperature'].round(0)
    agg = sub.groupby('temp_rounded')['Usage'].mean().reset_index()
    points = [{'x': round(row['temp_rounded'], 1), 'y': round(row['Usage'], 2)} for _, row in agg.iterrows()]
    return jsonify(points)

@app.route('/api/state_list')
def api_state_list():
    return jsonify(sorted(df['States'].unique().tolist()))

@app.route('/api/distribution')
def api_distribution():
    state = request.args.get('state', 'Overall')
    sub = df if state in ['All', 'Overall'] else df[df['States'] == state]
    counts, edges = np.histogram(sub['Usage'].dropna(), bins=30)
    mids = [(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)]
    return jsonify({'bins':[round(v,2) for v in mids], 'counts':counts.tolist()})

@app.route('/api/correlation')
def api_correlation():
    cols = ['latitude','longitude','temperature','humidity','DayOfWeek',
            'IsWeekend','Day','Month','Quarter','Lag1','Rolling7','RollingStd7','Usage']
    avail = [c for c in cols if c in df.columns]
    corr  = df[avail].corr().round(3)
    return jsonify({'labels':avail, 'matrix':corr.values.tolist()})

@app.route('/api/models')
def api_models():
    return jsonify(METRICS)

@app.route('/api/feature_importance')
def api_feature_importance():
    return jsonify({
        'features':   [d['feature']    for d in FEATURE_IMPORTANCE],
        'importance': [d['importance'] for d in FEATURE_IMPORTANCE],
    })

# ── Weather from Open-Meteo (free, no API key) ────────────────────────────────
@app.route('/api/weather')
def api_weather():
    lat  = request.args.get('lat',  type=float)
    lon  = request.args.get('lon',  type=float)
    date_str = request.args.get('date', '')
    if not lat or not lon or not date_str:
        return jsonify(error="lat, lon, date required"), 400

    try:
        # Open-Meteo historical/forecast — works for any date
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
            f"&hourly=relativehumidity_2m"
            f"&start_date={date_str}&end_date={date_str}"
            f"&timezone=Asia%2FKolkata"
        )
        r = requests.get(url, timeout=8)
        data = r.json()

        t_max = data['daily']['temperature_2m_max'][0]
        t_min = data['daily']['temperature_2m_min'][0]
        temp  = round((t_max + t_min) / 2, 1) if t_max is not None and t_min is not None else None

        hum_vals = [h for h in data['hourly']['relativehumidity_2m'] if h is not None]
        humidity = round(sum(hum_vals)/len(hum_vals), 1) if hum_vals else None

        return jsonify({'temperature': temp, 'humidity': humidity, 'source': 'open-meteo.com'})
    except Exception as e:
        return jsonify(error=str(e)), 500

# ── Lag & rolling features from dataset ──────────────────────────────────────
@app.route('/api/lag_features')
def api_lag_features():
    state    = request.args.get('state', '')
    date_str = request.args.get('date',  '')
    if not state or not date_str:
        return jsonify(error="state and date required"), 400
    try:
        target = pd.to_datetime(date_str)
        sub = df[df['States'] == state].sort_values('Dates').copy()
        if sub.empty:
            return jsonify(error=f"State '{state}' not found"), 404

        # Past usages relative to target date
        past = sub[sub['Dates'] < target].sort_values('Dates')

        def get_lag(n):
            row = sub[sub['Dates'] == target - pd.Timedelta(days=n)]
            if not row.empty: return float(row['Usage'].iloc[0])
            # fallback: use mean of available history
            return round(float(past['Usage'].mean()), 2) if not past.empty else 0.0

        def rolling_avg(n):
            window = past.tail(n)
            return round(float(window['Usage'].mean()), 2) if not window.empty else 0.0

        def rolling_std(n):
            window = past.tail(n)
            return round(float(window['Usage'].std()), 2) if len(window) > 1 else 0.0

        return jsonify({
            'Lag1':        get_lag(1),
            'Lag7':        get_lag(7),
            'Lag14':       get_lag(14),
            'Rolling7':    rolling_avg(7),
            'Rolling14':   rolling_avg(14),
            'Rolling30':   rolling_avg(30),
            'RollingStd7': rolling_std(7),
        })
    except Exception as e:
        return jsonify(error=str(e)), 500

# ── State coords ──────────────────────────────────────────────────────────────
@app.route('/api/state_coords')
def api_state_coords():
    return jsonify(STATE_COORDS)

# ── Prediction (Ridge only) ───────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if ridge_model is None:
        return jsonify(error="ridge_model.pkl not loaded"), 500
    data = request.get_json()
    try:
        row  = pd.DataFrame([{f: float(data.get(f, 0)) for f in FEATURES}])
        pred = float(ridge_model.predict(row)[0])
        return jsonify({'prediction': round(pred, 2), 'model': 'Ridge Regression (α=10)'})
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run()