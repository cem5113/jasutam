# utils/constants.py

# ── Zaman dilimi / model bilgisi ──────────────────────────────────────────────
SF_TZ_OFFSET     = -7
MODEL_VERSION    = "v0.1.0"
MODEL_LAST_TRAIN = "2025-10-04"
CACHE_VERSION    = "v2-geo-poisson"

# ── Ana anahtar sütun adı (VERİDE küçük harf!) ───────────────────────────────
# Not: Veri ve GeoJSON tarafında alan ismi 'geoid' (küçük). Diğer modüller,
# gerektiğinde büyük/küçük eşlemesini kendi içinde zaten tolere ediyor.
KEY_COL = "geoid"

# Uyumlu alan adları (bazı dosyalarda GEOID / id gelebilir)
KEY_COL_ALIASES = ["geoid", "GEOID", "id"]

# ── Modelin ürettiği suç türleri (kolonlar) ───────────────────────────────────
CRIME_TYPES = ["assault", "burglary", "theft", "robbery", "vandalism"]

# ── UI listeleri ──────────────────────────────────────────────────────────────
CATEGORIES = ["Assault", "Burglary", "Robbery", "Theft", "Vandalism", "Vehicle Theft"]
DAYS      = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASONS   = ["Winter", "Spring", "Summer", "Autumn"]

# ── Kategori → kolon eşleme ──────────────────────────────────────────────────
# Not: 'Vehicle Theft' ayrı kolonlar içeriyorsa (vehicle_theft/auto_theft/...),
# bunlar kullanılır; yoksa app tarafındaki filtre 'present columns' kontrolü
# yaptığından sessizce atlanır.
CATEGORY_TO_KEYS = {
    "Assault"       : ["assault"],
    "Burglary"      : ["burglary"],
    "Robbery"       : ["robbery"],
    "Theft"         : ["theft", "larceny"],
    "Vandalism"     : ["vandalism"],
    "Vehicle Theft" : ["vehicle_theft", "auto_theft", "motor_vehicle_theft"],
}

# (Opsiyonel) 5 kademeli tier etiketleri – UI tarafında işinize yararsa:
TIER_LEVELS_5 = ["Çok Düşük", "Düşük", "Orta", "Yüksek", "Çok Yüksek"]
