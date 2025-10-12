# sutam/config/data_settings.py
import os
from dataclasses import dataclass

@dataclass
class DataSettings:
    data_url: str | None = os.getenv("DATA_URL")
    data_dir: str | None = os.getenv("DATA_DIR")
    gdrive_file_id: str | None = os.getenv("GDRIVE_FILE_ID")

    @property
    def resolved_source(self) -> dict:
        if self.data_url:
            return {"type": "url", "value": self.data_url}
        if self.data_dir:
            return {"type": "dir", "value": self.data_dir}
        if self.gdrive_file_id:
            return {"type": "gdrive", "value": self.gdrive_file_id}
        # fallback
        return {
            "type": "url",
            "value": "https://raw.githubusercontent.com/<user>/<repo>/main/data/sf_crime.csv"
        }

data_settings = DataSettings()
