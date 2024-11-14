from pydantic import BaseModel
from enum import Enum

class MediaType(str, Enum):
    TV = "TV"
    Radio = "Radio"
    Newspaper = "Newspaper"

class RegressionSchema(BaseModel):
    column_name: MediaType