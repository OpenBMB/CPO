export PATH=$PATH:/home/jeeves/.local/bin

pip install datasets -U
pip install -U transformers
pip install fire
pip install accelerate


python ./src/test_harmlessness.py \
--base_model /data/checkpoints/ \
--output_dir /data/results/harmlessness_h3_cdpo_0227_6w_1_1_harmful1.json \
--prompt_template mistral \
--data_dir "./src/data_harmlessness/harmful_bench_1.json" \
--output_dir_1 /data/results/harmlessness_h3_cdpo_0227_6w_1_1_harmful0.json \
--data_dir_1 "./src/data_harmlessness/harmful_bench_0.json"




