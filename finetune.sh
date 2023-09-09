/home/jikeyu/miniconda3/envs/finetune/bin/python finetune.py \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path '/home/jikeyu/TOXIGEN/prompts/hate_black_1k.txt' \
    --output_dir './lora-alpaca' \
    --num_epochs 10 \
    --cutoff_len 512 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_dropout 0.05 \
    --micro_batch_size=8 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj, gate_obj, up_proj, down_proj, embed_tokens, lm_head]' \
    --prompt_template_name "toxigen"
    # --resume_from_checkpoint './lora-alpaca/checkpoint-800'
