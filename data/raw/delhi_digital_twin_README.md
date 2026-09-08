# 🏙️ Delhi Digital Twin — Decision Intelligence System

A full-stack AI-powered simulation platform for urban decision-making.

---

## 📁 Project Structure

```
city-twin/
├── backend/
│   ├── requirements.txt          # Python dependencies
│   ├── .env                      # API keys (fill before running)
│   ├── config.py                 # Settings loader
│   ├── main.py                   # FastAPI server
│   ├── data/
│   │   ├── delhi_network.py      # 12 zones, 20 road segments (BPR model)
│   │   └── live_fetcher.py       # OpenWeather, TomTom, OpenAQ APIs
│   └── simulation/
│       ├── engine.py             # BPR traffic + Gaussian pollution model
│       └── recommender.py        # Rule-based + scoring AI recommendation engine
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx               # Root layout (3-panel mission-control UI)
        ├── api/cityApi.js        # Axios client + WebSocket
        ├── hooks/useCityData.js  # Data fetching hook
        ├── styles/global.css     # Dark cyberpunk theme
        └── components/
            ├── MapView.jsx           # Leaflet map with road overlays
            ├── Dashboard.jsx         # Health ring, metric cards
            ├── ScenarioPanel.jsx     # 14 scenarios + custom params
            ├── ComparisonTable.jsx   # Before/after metrics table
            ├── RecommendationPanel.jsx # Priority-sorted AI recommendations
            ├── WeatherWidget.jsx     # Live weather display
            └── Charts.jsx            # Recharts: bar, pie, comparison
```

---

## ⚡ Quick Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or pnpm

---

## 🔧 Step 1 — Configure API Keys

Edit `backend/.env`:

```env
OPENWEATHER_API_KEY=your_key_here        # https://openweathermap.org/api
TOMTOM_API_KEY=your_key_here             # https://developer.tomtom.com
CPCB_API_KEY=your_key_here               # CPCB (optional, falls back to OpenAQ)
```

**If you don't have keys**, the app will still work using realistic synthetic data generated from Delhi traffic patterns. Just leave placeholders.

---

## 🐍 Step 2 — Start the Backend

```bash
cd city-twin/backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

✅ Backend running at: **http://localhost:8000**
✅ API docs available at: **http://localhost:8000/docs**

---

## ⚛️ Step 3 — Start the Frontend

Open a **new terminal**:

```bash
cd city-twin/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

✅ Frontend running at: **http://localhost:3000**

Open your browser → **http://localhost:3000**

---

## 🎮 How to Use

### 1. City Overview
- Left panel → **Overview tab**
- See real-time health score, AQI, congestion, average speed
- Weather widget shows live Delhi conditions

### 2. Run a Simulation
- Left panel → **Scenario tab**
- Choose from **14 scenarios**:
  - **Presets**: Rush Hour, NH-48 Closure, Heavy Rain, Republic Day, Smart Signals, Ring Road Construction, Odd-Even
  - **Custom**: Road Closure, Traffic Surge (adjustable %), Rainfall, Event Crowd, Signal Optimization, Construction, Emission Reduction
- Set parameters (sliders/dropdowns appear for custom scenarios)
- Click **Run Simulation**

### 3. View Results
- **Map updates** → red/orange roads show new congestion after simulation
- Right panel → **Recs tab** → AI recommendations sorted by priority
- Right panel → **Compare tab** → Before/After table for all metrics
- Right panel → **Charts tab** → Visual comparison bar chart + LOS distribution

### 4. Real-Time Updates
- The dashboard auto-refreshes every 30 seconds via WebSocket
- Click **↻ Refresh** button in the header to force a live data pull

---

## 🧠 Technical Architecture

### Traffic Model (BPR)
Uses the **Bureau of Public Roads (BPR) function**:
```
t(v) = t₀ × (1 + 0.15 × (v/c)⁴)
```
- `t₀` = free-flow travel time
- `v` = volume (vehicles/hr)
- `c` = capacity (PCU/hr)
- Outputs **LOS A–F** and effective speed

### Pollution Model
Gaussian dispersion adapted for urban roads:
- Road-type base emission factors
- Wind speed dispersion coefficient
- Weather penalty (rain, humidity)
- Converts to Indian **AQI scale (0–500)**

### Recommendation Engine
Hybrid rule-based + scoring system:
- Scenario-specific rules (3–4 per scenario type)
- Universal bottleneck detection (V/C > 0.88)
- Pollution emergency alerts (AQI > 300)
- Each recommendation scored 0–100 with **Critical/High/Medium/Low** priority

### API Integrations
| API | Usage | Fallback |
|-----|-------|---------|
| OpenWeatherMap | Real-time temp, humidity, wind, rain | Seasonal Delhi averages |
| TomTom Traffic | Live V/C ratios per road | BPR synthetic from hour-of-day |
| OpenAQ | Real PM2.5 → AQI | Road-based emission model |

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/city/state` | GET | Full city state (roads, zones, weather) |
| `/api/city/zones` | GET | Zone definitions |
| `/api/city/roads` | GET | Road network data |
| `/api/simulation/run` | POST | Run a scenario simulation |
| `/api/simulation/scenarios` | GET | Available preset scenarios |
| `/api/live/weather` | GET | Live weather data |
| `/api/live/traffic` | GET | Live traffic data |
| `/api/live/aqi` | GET | Live AQI data |
| `/api/live/refresh` | POST | Force cache refresh |
| `/ws/city` | WebSocket | Push city state every 30s |

### Simulation Request Example
```json
POST /api/simulation/run
{
  "scenario_id": "traffic_surge",
  "params": {
    "surge_pct": 40
  }
}
```

---

## 🗺️ Delhi Network Coverage

**12 Zones**: Connaught Place, Old Delhi, Dwarka, Rohini, Noida Border,
Gurugram Border, South Delhi, East Delhi, North Delhi, Janakpuri,
Lajpat Nagar, Nehru Place

**20 Road Segments**: Ring Road (E/W), NH-48, NH-24, Mathura Road,
Outer Ring Road (N/S), GT Road, Dwarka Expressway, + 12 connectors

---

## 🚀 Production Build

```bash
cd frontend
npm run build           # Creates dist/ folder
npm run preview         # Preview production build at :4173
```

To serve backend in production:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🐛 Troubleshooting

**Port conflicts**
```bash
# Backend on different port
uvicorn main:app --port 8001

# Update vite.config.js proxy target to match
```

**Frontend can't connect to backend**
- Ensure backend is running on port 8000
- Check browser console for CORS errors
- Verify vite.config.js proxy settings

**osmnx not loading**
```bash
pip install osmnx --upgrade
# Also install: conda install -c conda-forge osmnx  (if using conda)
```

**Leaflet map not showing tiles**
- Map uses OSM tiles, requires internet
- The dark city aesthetic uses CSS filters on the tile layer
