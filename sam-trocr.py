import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric
from transformers import default_data_collator

df = pd.read_fwf('./sam_gt_resized/sam_gt.txt', header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
print(df.head())

train_df, test_df = train_test_split(df, test_size=0.1)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

class SAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

train_dataset = SAMDataset(root_dir='./sam_gt_resized/',
                           df=train_df,
                           processor=processor)
eval_dataset = SAMDataset(root_dir='./sam_gt_resized/',
                           df=test_df,
                           processor=processor)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

encoding = train_dataset[0]
for k,v in encoding.items():
  print(k, v.shape)

#labels = encoding['labels']
#labels[labels == -100] = processor.tokenizer.pad_token_id
#label_str = processor.decode(labels, skip_special_tokens=True)
#print(label_str)

#model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

encoder_checkpoint = "google/vit-base-patch16-384"
#encoder_checkpoint = "google/vit-base-patch16-224-in21k"
#encoder_checkpoint = "microsoft/swinv2-tiny-patch4-window8-256"
decoder_checkpoint = "imvladikon/alephbertgimmel-base-512"
#
# load a fine-tuned image captioning model and corresponding tokenizer and image processor
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
   encoder_checkpoint, decoder_checkpoint
).to("cuda")

# added as per https://github.com/huggingface/transformers/issues/14195#issuecomment-1039204836
model.config.decoder.is_decoder = True
model.config.decoder.add_cross_attention = True
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True, 
    output_dir=f"{encoder_checkpoint}-ft-sam-v2",
    overwrite_output_dir=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=10,
    push_to_hub=False,
    hub_token="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    logging_steps=2,
    save_steps=1000,
    eval_steps=200,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
)

cer_metric = load_metric("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

trainer.train()
#processor.save_pretrained('./sam-train')
#model.save_pretrained('./sam-train')
trainer.save_model()
trainer.push_to_hub()
