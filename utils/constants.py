# utils/constants.py

# ── Zaman dilimi / model bilgisi ──────────────────────────────────────────────
SF_TZ_OFFSET     = -7
MODEL_VERSION    = "v0.1.0"
MODEL_LAST_TRAIN = "2025-10-04"
CACHE_VERSION    = "v2-geo-poisson"

# ── Anahtar kolon (GEOID eşleşmesi) ───────────────────────────────────────────
KEY_COL = "geoid"
KEY_COL_ALIASES = ["geoid", "GEOID", "id"]

# ── Modelin ürettiği suç türleri (kolonlar) ───────────────────────────────────
CRIME_TYPES = ["assault", "burglary", "theft", "robbery", "vandalism"]

# ── UI listeleri ──────────────────────────────────────────────────────────────
CATEGORIES = ["Assault", "Burglary", "Robbery", "Theft", "Vandalism", "Vehicle Theft"]
DAYS       = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASONS    = ["Winter", "Spring", "Summer", "Autumn"]

# ── Kategori → kolon eşleştirmesi ─────────────────────────────────────────────
CATEGORY_TO_KEYS = {
    "Assault"       : ["assault"],
    "Burglary"      : ["burglary"],
    "Robbery"       : ["robbery"],
    "Theft"         : ["theft", "larceny"],
    "Vandalism"     : ["vandalism"],
    "Vehicle Theft" : ["vehicle_theft", "auto_theft", "motor_vehicle_theft"],
}

# ── Risk seviyeleri ───────────────────────────────────────────────────────────
TIER_LEVELS_5 = ["Çok Düşük", "Düşük", "Orta", "Yüksek", "Çok Yüksek"]

# ── Harita katman adları (UI için) ────────────────────────────────────────────
RISK_LAYER_NAME          = "Tahmin (risk)"
PERM_HOTSPOT_LAYER_NAME  = "Hotspot (kalıcı)"
TEMP_HOTSPOT_LAYER_NAME  = "Hotspot (geçici)"

# ── Varsayılan harita ayarları ────────────────────────────────────────────────
MAP_CENTER = [37.7749, -122.4194]  # San Francisco koordinatları
MAP_ZOOM_START = 12
MAP_TILE_STYLE = "cartodbpositron"

# ── Görselleştirme renk paletleri ─────────────────────────────────────────────
RISK_COLORS = {
    "Çok Düşük": "#c7e9b4",
    "Düşük": "#7fcdbb",
    "Orta": "#41b6c4",
    "Yüksek": "#1d91c0",
    "Çok Yüksek": "#225ea8",
}

# ── UI / katman kontrolü için varsayılan görünürlük ───────────────────────────
DEFAULT_LAYER_VISIBILITY = {
    "risk_layer_show": True,
    "perm_hotspot_show": True,
    "temp_hotspot_show": True,
}
