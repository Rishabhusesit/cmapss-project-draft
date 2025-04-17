from pydantic import BaseModel
from typing import List

class SensorData(BaseModel):
    data: List[List[float]]  # shape (50, 21)
