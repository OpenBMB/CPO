export PATH=$PATH:/home/jeeves/.local/bin

pip install datasets -U
pip install -U transformers
pip install fire
pip install accelerate


python ./src/test_honesty.py \
--base_model /data/checkpoints/ \
--output_dir /data/results/honesty \
--prompt_template mistral \
--data_dir "./src/data_honesty/5.json"


