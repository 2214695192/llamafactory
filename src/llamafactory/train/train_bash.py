from typing import Literal, Optional, List
from transformers import Trainer
from ..hparams import ModelArguments, DataArguments, FinetuningArguments
from ..train.trainer_utils import KnowledgeAwareTrainer
from transformers import Seq2SeqTrainingArguments
from transformers.trainer_callback import TrainerCallback

def get_trainer(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"],
    callbacks: Optional[List["TrainerCallback"]] = None,
) -> "Trainer":
    # ... existing code ...
    
    if finetuning_args.finetuning_type == "lora" and finetuning_args.use_dynamic_rank:
        trainer = KnowledgeAwareTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
    
    return trainer 