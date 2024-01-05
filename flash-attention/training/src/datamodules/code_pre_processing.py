import jsonlines

file_path = '/home/avi_keinan_a_k/data/code/python/final/jsonl/test/python_test_0.jsonl'
text_lines = []  # Container to store text lines

with jsonlines.open(file_path) as reader:
    for line in reader:
        # Process each line of the JSONL file
        code_tokens = line['code_tokens']
        text_line = ' '.join(code_tokens)
        text_lines.append(text_line)  # Collect text_line in the container

new_file_path = '/home/avi_keinan_a_k/data/code/python/final/jsonl/test/new_python_test_0_text.jsonl'
with jsonlines.open(new_file_path, mode='a') as writer:
    for text_line in text_lines:
        writer.write({'text': text_line})
    
print('Done!')
