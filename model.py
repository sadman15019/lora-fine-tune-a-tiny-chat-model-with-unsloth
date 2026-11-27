"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel

def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype= None,
        load_in_4bit = True
    )

    return model,tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    return sum(param.numel() for param in model.parameters())
    pass

# Step 3 - is_model_4bit_quantized
import bitsandbytes as bnb
def is_model_4bit_quantized(model):
    for modules in model.modules():
        if isinstance(modules, bnb.nn.Linear4bit):
            return True

    return False
    pass

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
    pass

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    return ["q_proj", "k_proj", "v_proj", "o_proj"]
    pass

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    if target_modules is None:
        target_modules = get_lora_target_modules()
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules= target_modules,
        lora_alpha = lora_alpha,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )

    return model

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    return float(trainable_count/total_count)

# Step 9 - build_instruction_examples
def build_instruction_examples():

    examples = [
        {
            "instruction": "Say hello in one short sentence.",
            "response": "Hello, nice to meet you!"
        },
        {
            "instruction": "What is the capital of Bangladesh?",
            "response": "Dhaka is the capital of Bangladesh."
        },
        {
            "instruction": "What is 5 plus 3?",
            "response": "The answer is 8."
        },
        {
            "instruction": "Classify the sentiment: I love this movie.",
            "response": "Positive"
        },
        {
            "instruction": "Explain machine learning in one sentence.",
            "response": "Machine learning allows computers to learn patterns from data."
        },
        {
            "instruction": "Translate 'Thank you' into Bengali.",
            "response": "ধন্যবাদ।"
        }
    ]

    return examples

# Step 10 - format_instruction_example
def format_instruction_example(example):
   str =  '### Instruction:\n'+example['instruction']+'\n\n### Response:\n'+example['response']
   return str

# Step 11 - format_all_examples
def format_all_examples(examples):
    formatedList  = []
    for example in examples:
        str = format_instruction_example(example)
        formatedList.append(str)

    return formatedList

# Step 12 - build_text_dataset
from datasets import Dataset

def build_text_dataset(texts):
    dataset = Dataset.from_dict({"text":texts})
    return dataset

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    encoded = tokenizer(
        text,
        padding = False,
        truncation = False
    )

    return encoded['input_ids']

# Step 14 - count_tokens
def count_tokens(input_ids):
    return len(input_ids)

# Step 15 - build_training_arguments
import torch
from transformers import TrainingArguments

def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    use_bf16 = torch.cuda.is_bf16_supported()

    trainingArguments = TrainingArguments(
        output_dir=output_dir,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,

        max_steps=max_steps,
        learning_rate=learning_rate,

        bf16=use_bf16,
        fp16=not use_bf16,

        logging_steps=1,
        optim="adamw_8bit",

        report_to="none"
    )

    return trainingArguments

# Step 16 - build_sft_trainer
from trl import SFTTrainer, SFTConfig

def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    sft_config = SFTConfig(
        output_dir = training_args.output_dir,
        per_device_train_batch_size = training_args.per_device_train_batch_size,
        gradient_accumulation_steps= training_args.gradient_accumulation_steps,
        max_steps= training_args.max_steps,
        learning_rate= training_args.learning_rate,
        bf16= training_args.bf16,
        fp16= training_args.fp16,
        logging_steps=training_args.logging_steps,
        optim= training_args.optim,
        report_to= training_args.report_to,
        dataset_text_field = "text",
        max_length = max_seq_length,
        packing = "False"
    )

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = sft_config
    )

    return trainer

# Step 17 - run_sft_training
def run_sft_training(trainer):

    training_result = trainer.train()

    metrics = training_result.metrics

    return float(metrics["train_loss"])

# Step 18 - switch_to_inference_mode
def switch_to_inference_mode(model):

    FastLanguageModel.for_inference(model)

    return model

# Step 19 - build_chat_prompt
def build_chat_prompt(tokenizer, instruction):

    messages = [
        {
        "role":"user",
        "content": instruction
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt= True
    )

    return prompt

# Step 20 - generate_reply
def generate_reply(model, tokenizer, prompt, max_new_tokens=32):

    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        add_special_tokens= False
    ).to(model.device)


    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample= False
        )

    input_length = inputs["input_ids"].shape[1]

    generated_ids = output[0, input_length:]

    reply = tokenizer.decode(
        generated_ids,
        skip_special_tokens = True
    )

    return reply

