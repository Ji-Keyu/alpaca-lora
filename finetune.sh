/home/jikeyu/miniconda3/envs/finetune/bin/python finetune.py \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path 'skg/toxigen-data' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 2 \
    --learning_rate 1e-5 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --prompt_template_name "toxigen.json"
    # --resume_from_checkpoint './lora-alpaca/checkpoint-800'
