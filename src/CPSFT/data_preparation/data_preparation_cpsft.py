# This file is the data processing for CPSFT.
# Author: Yiju Guo
# Date: 2024-01
# Copyright (c) RUCBM, Renmin University of China. All rights reserved.
# See LICENSE file in the project root for license information.

import json

# JSONL file list
# UltraFeedback: https://huggingface.co/datasets/openbmb/UltraFeedback
# UltraSafety: https://huggingface.co/datasets/openbmb/UltraSafety
jsonl_files = ["evol_instruct.jsonl", "flan.jsonl", "truthful_qa.jsonl","false_qa.jsonl", "sharegpt.jsonl", "ultrachat.jsonl"]

# Build the result list
results = []

for file_name in jsonl_files:
    # Read the JSONL file
    with open(file_name, 'r') as file:
        data = file.readlines()

    # Parse each piece of data
    for line in data:
        json_data = json.loads(line)
        instruction = json_data["instruction"]
        completions = json_data["completions"]
        
        # Extract helpfulness or honesty Rating values from annotations for each completion
        for completion in completions:
            
            annotations = completion["annotations"]
            helpfulness = "< helpfulness: " + annotations["helpfulness"]["Rating"] +" >"
            honesty = "< honesty: " + annotations["honesty"]["Rating"] +" >"
        
            # Concatenate the values of instruction and helpfulness
            # result = helpfulness + " " + honesty+ " "  + instruction 
            # result = helpfulness + " " + instruction 
            result = honesty+ " "  + instruction 
            x = {
                "instruction": result,
                "input": "",
                "output": completion["response"]
            }

            results.append(x)
print(len(results))
# Write the result to a JSON file
with open('H2_sft.json', 'w') as file:
    json.dump(results, file, indent=4)