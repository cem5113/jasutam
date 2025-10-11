# utils/constants.py
# ——— Zaman Dilimi / Model Bilgisi ———
SF_TZ_OFFSET = -7  # America/Los_Angeles (yaz saati için -7, kışın -8 olabilir)
MODEL_VERSION = "v0.1.0"
MODEL_LAST_TRAIN = "2025-10-04"
CACHE_VERSION = "v2-geo-poisson"

# ——— Ana anahtar ———
# Tüm veri birleşimleri ve GeoJSON özellikleri bu kolon adıyla yapılır.
KEY_COL = "GEOID"

# ——— Suç türleri (model çıktısındaki kolonlar) ———
# Not: Şu an model şu 5 sütunu üretir. Ayrı "vehicle_theft" sütunu yoksa buraya ekleme!
CRIME_TYPES = ["assault", "burglary", "theft", "robbery", "vandalism"]

# ——— UI listeleri ———
CATEGORIES = ["Assault", "Burglary", "Robbery", "Theft", "Vandalism", "Vehicle Theft"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]

# ——— Kategori → kolon eşleme ———
# Araç hırsızlığı ayrı kolon olarak üretilmiyorsa (şu an yok),
# filtre işlesin diye "theft"e yönlendiriyoruz. İleride "vehicle_theft"
# kolonu eklersen sadece bu satırı güncelle:
#   "Vehicle Theft": ["vehicle_theft"]
CATEGORY_TO_KEYS = {
    "Assault": ["assault"],
    "Burglary": ["burglary"],
    "Robbery": ["robbery"],
    "Theft": ["theft"],                
    "Vandalism": ["vandalism"],
    "Vehicle Theft": ["theft"],       
}
