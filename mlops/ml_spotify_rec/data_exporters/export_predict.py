import pandas as pd
import numpy as np


@data_exporter
def export_data(y_pred, *args, **kwargs):
    y_pred_df = pd.DataFrame(y_pred)

    # Save DataFrame as a CSV file
    csv_file_path = 'answer.csv'
    y_pred_df.to_csv(csv_file_path, index=False)

