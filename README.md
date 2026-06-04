# Workflow-CI

Repository ini dibuat untuk Kriteria 3 submission Membangun Sistem Machine Learning.

Struktur utama:

```text
Workflow-CI
├── .github/workflows/mlflow-ci.yml
├── .workflow/mlflow-ci.yml
├── MLProject
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLProject
│   ├── requirements.txt
│   └── breast_cancer_preprocessing
└── README.md
```

Workflow CI menjalankan retraining model menggunakan MLflow Project setiap kali ada push ke branch `main` atau saat dijalankan manual dari tab Actions.

Cara menjalankan lokal:

```bash
pip install -r MLProject/requirements.txt
mlflow run MLProject --env-manager=local
```

Setelah repository ini diupload ke GitHub, buka tab **Actions**, pilih **MLflow Project CI Retraining**, lalu klik **Run workflow**. Pastikan hasilnya hijau/success sebelum melakukan submit ulang.
