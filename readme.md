# 1) скопировать веса в корень, чтобы проще запускать
cp zap_m_full/weights/best.pt ./best.pt

# 2) Инструкция запуска проверки (ROC AUC + таблицы) на датасете в формате Ultralytics Object Detection (экспорт CVAT):

1) Создать и активировать окружение:
   python -m venv .venv
   # Linux/macOS:
   source .venv/bin/activate
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1

2) Установить зависимости:
   pip install -r requirements.txt

3) Запустить оценку на датасете преподавателя:
   python scripts/evaluate.py --data /path/to/data.yaml --weights best.pt --split auto --device cpu --outdir out

   Если есть CUDA:
   python scripts/evaluate.py --data /path/to/data.yaml --weights best.pt --split auto --device cuda:0 --outdir out

Результаты будут сохранены в out/:
- out/summary.json (включая macro_roc_auc)
- out/per_class_auc.csv
- out/per_class_threshold.csv
EOF
