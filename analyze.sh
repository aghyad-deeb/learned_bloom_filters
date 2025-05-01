RUNS=(run_20250429_104423_emb4_hid2 run_20250429_110502_emb8_hid4 run_20250429_112526_emb16_hid8 run_20250429_114727_emb64_hid32 run_20250429_120849_emb128_hid64 run_20250429_123027_emb256_hid128 run_20250427_115026_emb32_hid16)
PLOT_TRADITIONAL=true
PLOT_LOG_SCALE=true

if [ "$PLOT_TRADITIONAL" = true ]; then
    TRADITIONAL="--traditional"
fi
if [ "$PLOT_LOG_SCALE" = true ]; then
    LOG_SCALE="--log-scale"
fi
python analyze_results.py -models_path models -metrics total inference tensor bloom $TRADITIONAL $LOG_SCALE
