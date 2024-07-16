export PATH=$PATH:/home/jeeves/.local/bin

pip install datasets -U
pip install deepspeed
pip install peft
pip install -U transformers

deepspeed --num_gpus=8 ./src/CPSFT/cpsft/train_sft.py\
 --data_path sft.json \
 --deepspeed ./src/deepspeed_config/ZeRO_3_cpu.json \
 --output_dir /data/checkpoints/mistral_sft/ \
 --eval_steps 20 \
 --save_steps 100 \
 --base_model /model/mistral-7b-hf \
 --prompt_template_name mistral_delete




