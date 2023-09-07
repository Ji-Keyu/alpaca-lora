- figure out the use of the parameter
- revise the alpaca instruction, input and output
  - ? use the alice to generate output of counter arguments
- train on a800
  - the easy part

# finetune.py
- --base_model=BASE_MODEL
    Type: str
    Default: ''
- -d, --data_path=DATA_PATH
    Type: str
    Default: 'yahma/alpaca-cleaned'
- -o, --output_dir=OUTPUT_DIR
    Type: str
    Default: './lora-alpaca'
- --batch_size=BATCH_SIZE
    Type: int
    Default: 128
- -m, --micro_batch_size=MICRO_BATCH_SIZE
    Type: int
    Default: 4
- -n, --num_epochs=NUM_EPOCHS
    Type: int
    Default: 3
- --learning_rate=LEARNING_RATE
    Type: float
    Default: 0.0003
- -c, --cutoff_len=CUTOFF_LEN
    Type: int
    Default: 256
    Description: ?
- -v, --val_set_size=VAL_SET_SIZE
    Type: int
    Default: 2000
    Description: ?
- --lora_r=LORA_R
    Type: int
    Default: 8
    Description: lora matrix rank
- --lora_alpha=LORA_ALPHA
    Type: int
    Default: 16
    Description: a scaling factor that adjusts the magnitude of the combined result
- --lora_dropout=LORA_DROPOUT
    Type: float
    Default: 0.05
- --lora_target_modules=LORA_TARGET_MODULES
    Type: List
    Default: ['q_proj', 'v_proj']
- -t, --train_on_inputs=TRAIN_ON_INPUTS
    Type: bool
    Default: True
- -a, --add_eos_token=ADD_EOS_TOKEN
    Type: bool
    Default: False
    Description: ? what is the effect of end of sequence token here
- -g, --group_by_length=GROUP_BY_LENGTH
    Type: bool
    Default: False
    Description: ?
- --wandb_project=WANDB_PROJECT
    Type: str
    Default: ''
- --wandb_run_name=WANDB_RUN_NAME
    Type: str
    Default: ''
- --wandb_watch=WANDB_WATCH
    Type: str
    Default: ''
- --wandb_log_model=WANDB_LOG_MODEL
    Type: str
    Default: ''
- -r, --resume_from_checkpoint=RESUME_FROM_CHECKPOINT
    Type: Optional[str]
    Default: None
- -p, --prompt_template_name=PROMPT_TEMPLATE_NAME
    Type: str
    Default: 'alpaca'
    Description: ? what other templates are supported


# generate.py
- --load_8bit=LOAD_8BIT
      Type: bool
      Default: False
- -b, --base_model=BASE_MODEL
      Type: str
      Default: ''
- --lora_weights=LORA_WEIGHTS
      Type: str
      Default: 'tloen/alpaca-lora-7b'
- -p, --prompt_template=PROMPT_TEMPLATE
      Type: str
      Default: ''
- --server_name=SERVER_NAME
      Type: str
      Default: '0.0.0.0'
- --share_gradio=SHARE_GRADIO
      Type: bool
      Default: False


# Output
- temperature
  Low temp => more predictable
  High temp => less predictable
- topk
  Ignore anything below the k'th most probable token
- topp
- 