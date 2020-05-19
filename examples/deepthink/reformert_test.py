
import os
import sys

import torch
from transformers import ReformerModelWithLMHead

# Encoding
def encode(list_of_strings, pad_to_max_length=True, pad_token_id=0):
    max_length = max([len(string) for string in list_of_strings])

    # create emtpy tensors
    attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
    input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

    for idx, string in enumerate(list_of_strings):
        # make sure string is in byte format
        if not isinstance(string, bytes):
            string = str.encode(string)

        input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
        attention_masks[idx, :len(string)] = 1

    return input_ids, attention_masks


# Decoding
def decode(outputs_ids):
    decoded_outputs = []
    for output_ids in outputs_ids.tolist():
        # transform id back to char IDs < 2 are simply transformed to ""
        decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
    return decoded_outputs


model = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")
input_ids, attention_masks = encode(["In 1965, Brooks left IBM to found the Department of"])
print(input_ids)
print(attention_masks)
output = decode(model.generate(input_ids, do_sample=True, max_length=150))
print(output)

