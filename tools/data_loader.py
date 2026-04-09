import pandas as pd
from typing import Tuple, List

def load_dataset(dataset_path: str) -> Tuple[pd.DataFrame, List[str]]:
   df = pd.read_csv(dataset_path)
   return df, df.columns.tolist()
