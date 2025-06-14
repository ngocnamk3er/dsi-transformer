from data import IndexingTrainDataset, IndexingCollator, QueryEvalCollator
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
)
from trainer import IndexingTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer, get_scheduler
from torch.optim import AdamW 
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

class QueryEvalCallback(TrainerCallback):
    def __init__(
        self,
        test_dataset,
        logger,
        restrict_decode_vocab,
        args: TrainingArguments,
        tokenizer: T5Tokenizer,
    ):
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.restrict_decode_vocab = restrict_decode_vocab
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=QueryEvalCollator(self.tokenizer, padding="longest"),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        hit_at_1 = 0
        hit_at_10 = 0
        model = kwargs["model"].eval()
        for batch in tqdm(
            self.dataloader, desc="\rEvaluating dev queries", leave=False
        ):
            inputs, labels = batch
            with torch.no_grad():
                batch_beams = model.generate(
                    inputs["input_ids"].to(model.device),
                    max_length=20,
                    num_beams=10,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=10,
                    early_stopping=True,
                ).reshape(inputs["input_ids"].shape[0], 10, -1)
                for beams, label in zip(batch_beams, labels):
                    rank_list = self.tokenizer.batch_decode(
                        beams, skip_special_tokens=True
                    )  # beam search should not return repeated docids but somehow due to T5 tokenizer there some repeats.
                    hits = np.where(np.array(rank_list)[:10] == label)[0]
                    if len(hits) != 0:
                        hit_at_10 += 1
                        if hits[0] == 0:
                            hit_at_1 += 1
        self.logger.log(
            {
                "Hits@1": hit_at_1 / len(self.test_dataset),
                "Hits@10": hit_at_10 / len(self.test_dataset),
            }
        )


def compute_metrics(eval_preds):
    num_predict = 0
    num_correct = 0
    for predict, label in zip(eval_preds.predictions, eval_preds.label_ids):
        num_predict += 1
        if len(np.where(predict == 1)[0]) == 0:
            continue
        if np.array_equal(
            label[: np.where(label == 1)[0].item()],
            predict[
                np.where(predict == 0)[0][0].item()
                + 1 : np.where(predict == 1)[0].item()
            ],
        ):
            num_correct += 1

    return {"accuracy": num_correct / num_predict}


def main():
    model_name = "ngocnamk3er/dsi_transformers_code_t5_base_python_v2"
    L = 32  # only use the first 32 tokens of documents (including title)

    # We use wandb to log Hits scores after each epoch. Note, this script does not save model checkpoints.
    wandb.login(key="c804f1ccb46b89fce13fb3bffe8b517ebb2ffc8a")
    wandb.init(project="DSI-nam-vast-python-codet5", name="dsi_transformers_code_t5_base_python_v2")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="cache")

    train_dataset = IndexingTrainDataset(
        path_to_data="./Vault_multi_task_train_python_clean.json",
        max_length=L,
        cache_dir="cache",
        tokenizer=tokenizer,
    )

    # This eval set is really not the 'eval' set but used to report if the model can memorise (index) all training data points.
    eval_dataset = IndexingTrainDataset(
        path_to_data="./Vault_multi_task_train_python_clean.json",
        max_length=L,
        cache_dir="cache",
        tokenizer=tokenizer,
    )

    # This is the actual eval set.
    test_dataset = IndexingTrainDataset(
        path_to_data="./Vault_valid_python_clean.json",
        max_length=L,
        cache_dir="cache",
        tokenizer=tokenizer,
    )

    ################################################################
    # docid generation constrain, we only generate integer docids.
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS

    ################################################################

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.0005,
        warmup_steps=500,
        # weight_decay=0.01,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy="no",
        # eval_steps=100,
        max_steps=10000,
        save_total_limit=1,
        dataloader_drop_last=False,  # necessary
        report_to="wandb",
        logging_steps=50,
        save_strategy="steps",
        save_steps=1000,
        # fp16=True,  # gives 0/nan loss at some point during training, seems this is a transformers bug.
        dataloader_num_workers=4,
        gradient_accumulation_steps=2,
        push_to_hub=True,
        hub_model_id=f"ngocnamk3er/dsi_transformers_code_t5_base_python_v2",
        hub_strategy="every_save",
        
    )


    optimizer = AdamW(model.parameters(),
                      lr=training_args.learning_rate,
                      eps=training_args.adam_epsilon, 
                      betas=(training_args.adam_beta1, training_args.adam_beta2),
                      weight_decay=training_args.weight_decay) 

    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
        num_cycles=3,
    )

    

    trainer = IndexingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=IndexingCollator(
            tokenizer,
            padding="longest",
        ),
        compute_metrics=compute_metrics,
        callbacks=[
            QueryEvalCallback(
                test_dataset, wandb, restrict_decode_vocab, training_args, tokenizer
            )
        ],
        restrict_decode_vocab=restrict_decode_vocab,
        optimizers=(optimizer, lr_scheduler)
    )
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
