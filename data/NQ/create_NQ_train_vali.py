from datasets import load_dataset
import random
import numpy as np
import json

random.seed(313)

NUM_TRAIN = 8000
NUM_EVAL = 2000  
TOTAL_SAMPLES = NUM_TRAIN + NUM_EVAL

try:
    dataset = load_dataset("Fsoft-AIC/the-vault-function", split_set=["train/small"], languages=['python'])
    data = dataset['train_small']
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have internet connection and the dataset exists.")
    exit() 


rand_inds = list(range(len(data)))
random.shuffle(rand_inds)

processed_samples_count = 0


with open('Vault_multi_task_train_python.json', 'w') as tf, \
     open('Vault_valid_python.json', 'w') as vf:

    for ind in rand_inds:
        sample = data[ind]
        code_text = sample.get('code', '') 
        docstring_text = sample.get('docstring', '')
        code_item = json.dumps({'text_id': str(processed_samples_count), 'text': 'code: ' + code_text})
        docstring_item = json.dumps({'text_id': str(processed_samples_count), 'text': 'docstring: ' + docstring_text})

        if processed_samples_count == TOTAL_SAMPLES:
            print("Reached the total sample limit. Stopping further processing.")
            break

        tf.write(code_item + '\n')

        if processed_samples_count < NUM_TRAIN:
            tf.write(docstring_item + '\n')
        elif processed_samples_count < TOTAL_SAMPLES :
            vf.write(docstring_item + '\n')
        else:
             break
        processed_samples_count += 1

        print(f"Creating training and validation dataset: {'{:.1%}'.format(processed_samples_count/TOTAL_SAMPLES)}", end='\r')
print("\nDataset creation complete.")