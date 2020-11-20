import csv
import pandas as pd
from tableschema import Table

class CSVsCheckers():
    '''
        CSV checker that ensures predict and train CSVs columns and column data types correlate. 
    '''
    def __init__(self, train_csv_path: str, predict_csv_path:str, target_idx: int):
        '''
            target_idx should be 0 based indexing
        '''
        self.train_csv_path = train_csv_path
        self.predict_csv_path = predict_csv_path
        self.target_idx = target_idx
    
    def ensure_csvs_good(self):
        if not self._has_headers():
            self._csv_add_header(self.train_csv_path)
            self._csv_add_header(self.predict_csv_path)
        
        self._remove_target_col_from_predict()
        self._is_column_same_()

    
    def _has_headers(self):
        train_file = open(self.train_csv_path)
        predict_file = open(self.predict_csv_path)
        train_reader = csv.reader(train_file)
        predict_reader = csv.reader(predict_file)

        train_col = next(train_reader)[:self.target_idx:]
        predict_col = next(predict_reader)

        return train_col == predict_col

    def _csv_add_header(self, file_path):
        reader = csv.reader(file_path)
        num_col = len(next(reader))
        df = pd.read_csv(file_path, names=[i for i in range(num_col)])
        df.to_csv(file_path, index=False)

    def _remove_target_col_from_predict(self):
        train_df = pd.read_csv(self.train_csv_path)
        predict_df = pd.read_csv(self.predict_csv_path)

        self.target_col_name = train_df.columns[self.target_idx]

        try:
            predict_df.drop(column=[self.target_col_name])
            predict_df.to_csv(self.predict_csv_path, index=False)
            return
        except Exception:
            return

    def _is_column_same_(self, row_limit: int = 500, confidence: float = 0.85):
        train_table = Table(self.train_csv_path)
        predict_table = Table(self.predict_csv_path)
        train_schema = train_table.infer(limit=row_limit, confidence=confidence)
        predict_schema = predict_table.infer(limit=row_limit, confidence=confidence)

        target_name = train_schema['fields'][self.target_idx]['name']
        train_fields = train_schema['fields'][:self.target_idx:]
        predict_fields = predict_schema['fields']

        num_train_cols = len(train_fields)
        num_predict_cols = len(predict_fields)

        if  num_train_cols > num_predict_cols:
            raise Exception('Train csv and Predict csv have different column counts')

        train_col_names = [d['name'] for d in train_fields]
        predict_col_names = [d['name'] for d in predict_fields]

        num_col_diff = 0
        for train_dict in train_fields:
            train_dtype = train_dict['type']
            train_name = train_dict['name']

            if train_name not in predict_col_names and train_name != target_name:
                raise Exception(f'Column: {train_name} not in Predict CSV')

            is_in_predict = False

            for predict_dict in predict_fields:
                predict_dtype = predict_dict['type']
                predict_name = predict_dict['name']

                if predict_name not in train_col_names:
                    raise Exception(f'Column: {predict_name} not in Train CSV')

                if train_name == predict_name and train_dtype != predict_dtype:
                    raise Exception(f'Column: {train_name} dtype different between train and predict')