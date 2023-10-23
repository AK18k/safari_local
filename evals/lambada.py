import torch 
import argparse
import os
import sys
import yaml 
from tqdm import tqdm
import json 
from collections import Counter
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F



sys.path.append(os.environ.get("SAFARI_PATH", "."))

from src.models.sequence.long_conv_lm import ConvLMHeadModel

from transformers import AutoTokenizer, GPT2LMHeadModel
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import GPT2Tokenizer

try:
    from tokenizers import Tokenizer  
except:
    pass

# https://github.com/openai/gpt-2/issues/131#issuecomment-492786058
def preprocess(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


class LAMBADA:
    "LAMBADA (OpenAI) benchmark"
    def __init__(self, data_dir=None, use_stop_filter:bool=False, use_code_data:bool=True):
        self.use_code_data_ = use_code_data
        if self.use_code_data_:
            data_dir = os.environ.get("DATA_DIR", data_dir)
            code_path = os.path.join(data_dir + "/test/python_test_0.jsonl")
            with open(code_path, 'r') as f:
                sample_file = f.readlines()
            self.data = sample_file
        else:
            data_dir = os.environ.get("DATA_DIR", data_dir)
            lambada_path = os.path.join(data_dir + "/lambada/lambada_openai/lambada_test.jsonl")
            self.data = [preprocess(json.loads(line)['text']) for line in open(lambada_path)] 
        
        self.use_stop_filter = use_stop_filter 



    def calc_preplexity(self, model_output_logits, target_ids):
        val = F.cross_entropy(model_output_logits, target_ids, reduction='sum')
        return torch.exp(val / target_ids.shape[0])


    
    def run(self, model_cfg, ckpt_path):
        
        model, tokenizer = self.load_model(model_cfg, ckpt_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if isinstance(tokenizer, Tokenizer):
            vocab_size = tokenizer.get_vocab_size()
        else:
            vocab_size = tokenizer.vocab_size

        stop_filter = torch.zeros(vocab_size, device=device)
        if self.use_stop_filter:
            token_to_idx = {tokenizer.decode([i]):i for i in range(vocab_size)}
            for word in STOP_WORDS:
                if ' '+word in token_to_idx:
                    stop_filter[token_to_idx[' '+word]] = -float('inf')
            
        results = []
        tokenized_target = []
        skiped_prompts_count = 0
        target_token_index = -20
        process_data_samples_limit = 1_000
        for prompt in tqdm(self.data[:process_data_samples_limit]):
            if (self.use_code_data_):
                function_code_and_comments = json.loads(prompt)

                code_tokens= function_code_and_comments['code_tokens']

                # create string seperated by spaces from code tokens
                code_string = ' '.join(code_tokens)

                tokenized_prompt = tokenizer.encode(code_string)
                target_token_index_actual = max(target_token_index, -len(tokenized_prompt))
                target_tokenized = [tokenized_prompt[target_token_index_actual]]
                target_ids = tokenized_prompt
            else:
                target = prompt.split(" ")[-1]
                
                if isinstance(tokenizer, Tokenizer):
                    tokenized_prompt = tokenizer.encode(prompt).ids
                    target_tokenized = tokenizer.encode(' '+target).ids
                else:
                    tokenized_prompt = tokenizer.encode(prompt)
                    target_tokenized = tokenizer(' '+target)['input_ids']

            tokenized_target.append(target_tokenized)

            if len(tokenized_prompt) > 1024:
                skiped_prompts_count += 1
                continue

            out = model(torch.tensor([tokenized_prompt]).to(device=device))
            
            if type(out) == tuple: out = out[0]
            logits = out.logits[0][:-1, :vocab_size] # seq_len - 1, vocab_size

            logits = logits + stop_filter[None]
            preds = logits.argmax(-1)            
            acc = all([pred == answer for pred, answer 
                       #in zip(preds[-len(target_tokenized):], target_tokenized)
                       in zip(preds[target_token_index_actual:], target_tokenized)
                    ])          

            preplexity = self.calc_preplexity(logits, torch.tensor(target_ids[1:]).to(device=device))

            results.append((acc, preplexity.item()))
            
        accuracy_list = [result[0] for result in results]
        print(f"Accuracy {torch.tensor(accuracy_list).float().mean().item()*100:4.2f}")

        preplexity_list = [result[1] for result in results]
        print(f"Preplexity {torch.tensor(preplexity_list).mean().item():4.2f}")

        print(f"Skipped prompts {skiped_prompts_count}")

        # create a list of target tokens
        target_tokens = [tokenizer.decode(tokenized_target[i]) for i in range(len(tokenized_target))]


        # count the frequency of each target token
        token_counts = Counter(target_tokens)

        # calculate the percentage of each token
        total_tokens = sum(token_counts.values())
        token_percentages = {token: count/total_tokens*100 for token, count in token_counts.items()}

        # sort the tokens by their percentage values
        sorted_tokens = sorted(token_percentages.items(), key=lambda x: x[1], reverse=True)

        # plot a histogram of the token percentages
        fig, ax = plt.subplots()
        ax.bar([token[0] for token in sorted_tokens], [token[1] for token in sorted_tokens])
        ax.set_ylabel('Percentage')
        ax.set_xlabel('Token')
        ax.set_ylim([0, 100])
        ax.set_title('Token percentages')
        plt.show(block=True)        


        
            
    def load_model(self, model_cfg, ckpt_path):
        config = yaml.load(open(model_cfg, 'r'), Loader=yaml.FullLoader)
        model = ConvLMHeadModel(**config['model_config'])
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
        if config['tokenizer_name'] == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            tokenizer = None 
        return model, tokenizer
        
        
if __name__ == "__main__":
    
    SAFARI_PATH = os.getenv('SAFARI_PATH', '.')
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data",
        help="Path to data",
    )    
    
    parser.add_argument(
        "--model_cfg",
        default=f"{SAFARI_PATH}/configs/evals/hyena_small_150b.yaml",
    )
    
    parser.add_argument(
        "--ckpt_path",
        default=f"",
        help="Path to model state dict checkpoint"
    )
        
    parser.add_argument(
        "--stop_word_filter",
        type=bool,
        default=False,
        help="Filter out stop words",
    )
        
    args = parser.parse_args()
        
    task = LAMBADA(data_dir=args.data_dir, use_stop_filter=args.stop_word_filter, use_code_data=True)
    task.run(args.model_cfg, args.ckpt_path)