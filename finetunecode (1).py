
import os
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer


print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))


hf_token = "hf_token_here"  
model_id = "meta-llama/Llama-3.1-8B-Instruct"


def load_and_format_data(filepath):
    df = pd.read_json(filepath)
    
    formatted_data = [{
        "text": f"<s>[INST] {row['input']} [/INST] {row['output']}</s>"  
    } for _, row in df.iterrows()]
    return Dataset.from_pandas(pd.DataFrame(formatted_data))


full_dataset = load_and_format_data('train_dataset.json')


tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")


model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    token=hf_token,
    load_in_8bit=True
)


model.gradient_checkpointing_enable()


model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=8,  
    lora_alpha=16,  
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()




training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,   
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",  
    eval_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,  
    num_train_epochs=3,
    warmup_ratio=0.1,  
    weight_decay=0.01,  
    fp16=True,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    report_to="none",
    push_to_hub=True,
    hub_model_id="MriKalani/Llama-3.1-8B-Instruct",
    hub_strategy="end",  
    hub_token=hf_token,  
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


try:
    trainer = SFTTrainer (
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        eval_dataset=full_dataset,
        data_collator=data_collator,
        formatting_func=lambda x: x["text"],
        
    )
except Exception as e:
    print(f"SFTTrainer error: {e}")
    
    
    from transformers import Trainer
    
    
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,  
            padding="max_length",
            return_tensors="pt"
        )
    
    print("Tokenizing datasets...")
    tokenized_train_dataset = full_dataset.map(preprocess_function, batched=True)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        
        data_collator=data_collator
    )

trainer.push_to_hub(commit_message="Add fine-tuned Llama-3.1-8B-Instruct model")

print("Starting training...")
trainer.train()


print("Saving model and tokenizer...")
output_dir = "./trained_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete!")
