import csv
import pandas as pd
from tableschema import Table

class CSVsCheckers():
    '''
        CSV checker that ensures predict and train CSVs columns and column data types correlate. 
    '''
    def __init__(self, train_csv_path: str, predict_csv_path:str, target_name: int):
        '''
            target_idx should be 0 based indexing
        '''
        self.train_csv_path = train_csv_path
        self.predict_csv_path = predict_csv_path
        self.target = target_name
    
    def get_target_idx(self):
        return self.target_idx

    def ensure_csvs_good(self):
        train_file = open(self.train_csv_path)
        predict_file = open(self.predict_csv_path)
        train_reader = csv.reader(train_file)
        predict_reader = csv.reader(predict_file)
        train_col = next(train_reader)
        predict_col = next(predict_reader)

        self.target_idx = train_col.index(self.target)

        if train_col == predict_col:
            return
        elif train_col[:self.target_idx:] == predict_col:
            return
        else:
            raise Exception(f'Columns in train: {train_col}, don\'t correspond to columns in predict: {predict_col}')
        
        self._check_and_add_headers()        
        self._remove_target_col_from_predict()
        self._is_column_same_()

    def _check_and_add_headers(self, num_bytes: int = 1000):
        train_file = open(self.train_csv_path)
        predict_file = open(self.predict_csv_path)
        train_reader = csv.reader(train_file)
        predict_reader = csv.reader(predict_file)

        sniffer = csv.Sniffer()
        has_header_train = sniffer.has_header(train_file.read(num_bytes))
        has_header_predict = sniffer.has_header(predict_file.read(num_bytes))

        if has_header_train and has_header_predict:
            return
        
        train_col = next(train_reader)
        predict_col = next(predict_reader)

        if has_header_train and not has_header_predict:
            self._csv_add_header(self.predict_csv_path, train_col)
        elif has_header_predict and not has_header_train:
            self._csv_add_header(self.train_csv_path, predict_col)
        elif not has_header_predict and not has_header_train:
            self._csv_add_header(self.train_csv_path)
            self._csv_add_header(self.predict_csv_path)

    def _csv_add_header(self, file_path, headers: list = None):
        reader = csv.reader(file_path)
        num_col = len(next(reader))
        headers = [i for i in range(num_col)] if headers is None else headers
        df = pd.read_csv(file_path, names=headers)
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
            raise Exception('Train CSV and Predict CSV have different column counts. Both CSVs should have same number of features')

        train_col_names = [d['name'] for d in train_fields]
        predict_col_names = [d['name'] for d in predict_fields]

        num_col_diff = 0
        for train_dict in train_fields:
            train_dtype = train_dict['type']
            train_name = train_dict['name']

            if train_name not in predict_col_names and train_name != target_name:
                raise Exception(f'Column: "{train_name}" does not exist in Predict CSV')

            is_in_predict = False

            for predict_dict in predict_fields:
                predict_dtype = predict_dict['type']
                predict_name = predict_dict['name']

                if predict_name not in train_col_names:
                    raise Exception(f'Column: "{predict_name}" does not exist in Train CSV')

                if train_name == predict_name and train_dtype != predict_dtype:
                    raise Exception(f'Column: "{train_name}" has different dtype different between train and predict')

