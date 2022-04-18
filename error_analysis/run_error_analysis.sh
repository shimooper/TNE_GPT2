python error_analysis.py --span_csv_path ../../logs/logs/L1_variants/L1/eval_span_concurrent.csv --stop_csv_path ../../logs/logs/L1_variants/L1/eval_stop_concurrent.csv --output_dir out/concurrent/L1

python error_analysis.py --span_csv_path ../../logs/logs/L2_variants/L2/eval_span_concurrent.csv --stop_csv_path ../../logs/logs/L2_variants/L2/eval_stop_concurrent.csv --output_dir out/concurrent/L2

python error_analysis.py --span_csv_path ../../logs/logs/L2_variants/span_stop/eval_span_only.csv --output_dir out/span_stop/L2
