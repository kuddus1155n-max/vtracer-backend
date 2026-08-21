web: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 gunicorn vtracer_server:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --preload
