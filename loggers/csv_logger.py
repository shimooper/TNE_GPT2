import csv

class CSVLogger:
    def __init__(self, span_csv_file_path, stop_csv_file_path):
        if span_csv_file_path:
            self.span_csv_file = open(span_csv_file_path, 'w', encoding='utf-8')
            self.span_csv_writer = csv.writer(self.span_csv_file)
            header = ['epoch_id', 'text_id', 'anchor_id', 'prep_id', 'span_start', 'span_end', 'pred_start', 'pred_end', 'loss']
            self.span_csv_writer.writerow(header)
        else:
            self.span_csv_file = None
            self.span_csv_writer = None

        if stop_csv_file_path:
            self.stop_csv_file = open(stop_csv_file_path, 'w', encoding='utf-8')
            self.stop_csv_writer = csv.writer(self.stop_csv_file)
            header = ['epoch_id', 'text_id', 'anchor_id', 'prep_id', 'num_complements', 'curr_num_complements', 'loss']
            self.stop_csv_writer.writerow(header)
        else:
            self.stop_csv_file = None
            self.stop_csv_writer = None    

    def __del__(self):
        try:
            if self.span_csv_file:
                self.span_csv_file.close()
            if self.stop_csv_file:
                self.stop_csv_file.close()
        except:
            print("failed closing csv files")

    def log_span(self, epoch, text_id, amchor_id, prep_id, span_start, span_end, pred_start, pred_end, loss):
        row = [epoch, text_id, amchor_id, prep_id, span_start, span_end, pred_start, pred_end, loss]
        self.span_csv_writer.writerow(row)
        self.span_csv_file.flush()

    def log_stop(self, epoch_id, text_id, anchor_id, prep_id, num_complements, curr_num_complements, loss):
        row = [epoch_id, text_id, anchor_id, prep_id, num_complements, curr_num_complements, loss]
        self.stop_csv_writer.writerow(row)
        self.stop_csv_file.flush()
