import os
import pandas as pd
import polars as pl # Optional, for performance with large datasets
from typing import Union, Dict, Any
from fastapi import UploadFile
from unstructured.partition.auto import partition
import os
from dotenv import load_dotenv
import numpy as np


load_dotenv() # Load environment variables from .env file

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # For Google GenAI
    # Add other configurations like model names, file storage paths, etc.
    UPLOAD_FOLDER = "storage/uploaded_files"
    PROCESSED_DATA_FOLDER = "storage/processed_data"
    # Ensure these folders exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

    # LLM Models
    GROQ_MODEL = "llama3-8b-8192" # Or "llama3-70b-8192"
    GEMINI_MODEL = "gemini-pro"


# from config import Config



# Optional polars support
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


class DataProcessor:
    def __init__(self):
        self.upload_folder = Config.UPLOAD_FOLDER

    async def _save_upload_file(self, upload_file: UploadFile) -> str:
        """Save uploaded file to the configured upload folder."""
        file_location = os.path.join(self.upload_folder, upload_file.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(await upload_file.read())
        return file_location

    async def process_file(self, upload_file: UploadFile) -> Union[pd.DataFrame, str, Dict[str, Any], None]:
        """Process an uploaded file based on its extension."""
        file_path = await self._save_upload_file(upload_file)
        file_extension = os.path.splitext(upload_file.filename)[1].lower()

        try:
            # --- CSV / TXT ---
            if file_extension in ['.csv', '.txt']:
                try:
                    return pd.read_csv(file_path)
                except Exception as e:
                    if HAS_POLARS:
                        df = pl.read_csv(file_path)
                        return df.to_pandas()
                    else:
                        print(f"CSV parsing failed: {e}")
                        return None

            # --- Excel Files ---
            elif file_extension in ['.xlsx', '.xls']:
                try:
                    return pd.read_excel(file_path)
                except Exception as e:
                    if HAS_POLARS:
                        try:
                            df = pl.read_excel(file_path)
                            return df.to_pandas()
                        except Exception as pe:
                            print(f"Polars Excel parsing failed: {pe}")
                            return None
                    else:
                        print(f"Excel parsing failed: {e}")
                        return None

            # --- JSON ---
            elif file_extension == '.json':
                try:
                    return pd.read_json(file_path)
                except Exception as e:
                    if HAS_POLARS:
                        try:
                            df = pl.read_json(file_path)
                            return df.to_pandas()
                        except Exception as pe:
                            print(f"Polars JSON parsing failed: {pe}")
                            return None
                    else:
                        print(f"JSON parsing failed: {e}")
                        return None

            # --- Parquet ---
            elif file_extension == '.parquet':
                try:
                    return pd.read_parquet(file_path)
                except Exception as e:
                    if HAS_POLARS:
                        try:
                            df = pl.read_parquet(file_path)
                            return df.to_pandas()
                        except Exception as pe:
                            print(f"Polars Parquet parsing failed: {pe}")
                            return None
                    else:
                        print(f"Parquet parsing failed: {e}")
                        return None

            # --- Document Files (PDF, DOCX, DOC) ---
            elif file_extension in ['.pdf', '.docx', '.doc']:
                try:
                    elements = partition(filename=file_path)
                    document_content = "\n\n".join([str(el) for el in elements])
                    return document_content
                except Exception as e:
                    print(f"Document parsing failed: {e}")
                    return None

            # --- Unsupported format ---
            else:
                print(f"Unsupported file type: {file_extension}")
                return None

        except Exception as e:
            print(f"Error processing file {upload_file.filename}: {e}")
            return None

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return a summary of a pandas DataFrame with JSON-safe values."""
        # Ensure all values are JSON serializable
        summary = {
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "data_types": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            "first_5_rows": df.head()
            .replace({pd.NA: None, np.nan: None})
            .astype(object)  # Convert to Python native objects
            .where(pd.notnull(df), None)
            .applymap(lambda x: x.isoformat() if isinstance(x, (pd.Timestamp, np.datetime64)) else x)
            .to_dict(orient='records')
        }
        return summary


