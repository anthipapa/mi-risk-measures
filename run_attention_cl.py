from transformers import AutoTokenizer, AutoModel, utils, LongformerModel
from bertviz import model_view, head_view
utils.logging.set_verbosity_error()  # Suppress standard warnings
import torch
import numpy as np
from nltk.corpus import stopwords
import re, json, os, argparse
#import evaluation

STOPWORDS = set(stopwords.words("english"))
SPECIAL_TOKENS = ['<s>', '</s>', '[CLS]']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE", DEVICE)

def normalize_token(token):
    return token.replace("Ġ", "").replace("Ċ", "")

def get_attention_scores(model, tokenizer, input_text):
    """Get the attentions scores of all tokens across all layers and heads.
    Returns also the list of (normalized) tokens."""
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512)
    inputs = inputs.to(DEVICE)
    model = model.to(DEVICE)
    outputs = model(**inputs)  
    attentions = outputs.attentions    # Retrieve attention from model outputs
    return attentions, inputs 

def avg_attn(score, nr_layers, nr_heads):
    "Take the average of summed attention scores."
    return score/(nr_layers*nr_heads)

def get_token_pair_attentions(attentions, layers_to_use=list(range(12))):
    """ Returns a 2d matrix (array) representing pairs of tokens attending
    to each other. The score at each position in the matrix is the attention
    score for a token pair, averaged over all layers and attention heads.
    @ attentions: all attention scores from model outputs
    @ layers_to_use: lst; list of integers to use indicating the layer number 
                     (starting from 0)
    """
    # Sqeeze tensor: remove all the dimensions of input of size 1 from a tensor 
    # (used for batch_size here, which should always be 1 when using bert_viz).
    squeezed_attn = [layer_attention.squeeze(0) for layer_attention in attentions] # 12Lx12Hxseq_lenxseq_len
    per_token_attn = np.zeros((squeezed_attn[0].size(dim=1), squeezed_attn[0].size(dim=2)))
    for layer_ix, layer_attn in enumerate(squeezed_attn):
        if layer_ix in layers_to_use:
            print("\t\t Layer", layer_ix)
            for head_ix, head_attn in enumerate(layer_attn):
                for t_ix in range(head_attn.size(dim=0)):
                    for t2_ix in range(head_attn.size(dim=1)):
                        # get attention value for a pair of tokens & convert tensor value to float with .item()
                        per_token_attn[t_ix][t2_ix] += head_attn[t_ix][t2_ix].item()
    # vectorize function to be able to apply it to each array element
    avg_attn_vec = np.vectorize(avg_attn) 
    nr_layers = len(layers_to_use)
    avg_per_token_attn = avg_attn_vec(per_token_attn, nr_layers, squeezed_attn[0].size(dim=0))
    return avg_per_token_attn  

def get_attended_tokens(aggr_attn, tokens, query_tkn, threshold=0.03):
    """ Prints attentions scores for a query token. Just a toy example 
    to show how the aggregated scores can be used. """
    for tkn_ix, token in enumerate(tokens):
        if token == query_tkn:
            print("{} (tkn ix: {})".format(token, tkn_ix))
            for tkn2_ix, attended_tkn in enumerate(aggr_attn[tkn_ix]):
                if threshold and aggr_attn[tkn_ix][tkn2_ix] > threshold:
                    print("\t{} \tattn: {} \t(tkn ix: {})".format(tokens[tkn2_ix], round(aggr_attn[tkn_ix][tkn2_ix], 3), tkn2_ix))

def get_char_offsets(word_ids, tokens, inputs):
    """ For each token (subword) of a word, find the start and 
    end character offsets of the word they belong to in the input text.
    """
    char_offsets = []
    for t_ix, w_id in enumerate(word_ids):
        if tokens[t_ix] not in SPECIAL_TOKENS:
            char_offset = inputs.word_to_chars(w_id)
        else:
            char_offset = None
        char_offsets.append(char_offset)
    return char_offsets

def get_mask_offsets(aggr_attn, inputs, tokenizer, person_to_protect, threshold=0.01):
    """Get masking decisions based on attention scores for each token. Attention scores 
    are averaged across all layers and heads. Some scores are filtered: stopwords, 
    non-alphanumeric tokens, special tokens, previous and next tokens. 
    If the attention score of at least one protected name subword attends to a target word
    with a score above the threshold, the whole target word is masked. 
    Returns a mapping with start and end character offsets to a (boolean) masking decision.
    """
    tokens = inputs.tokens()
    tokens = [normalize_token(token) for token in tokens]
    word_ids = inputs.word_ids()
    char_offsets = get_char_offsets(word_ids, tokens, inputs)
    offsets_to_mask = []
    q_inputs = tokenizer.encode(person_to_protect, return_tensors='pt')
    query_tkns = tokenizer.convert_ids_to_tokens(q_inputs[0])  
    query_tkns = [normalize_token(token) for token in query_tkns if token not in SPECIAL_TOKENS]
    print("Tokenized query: ", query_tkns)
    for query_tkn in query_tkns:
        for tkn_ix, token in enumerate(tokens):
            if token == query_tkn:
                for tkn2_ix, attention_score in enumerate(aggr_attn[tkn_ix]):
                    attended_tkn = tokens[tkn2_ix]
                    if attended_tkn not in SPECIAL_TOKENS:
                        start_char_ix, end_char_ix = char_offsets[tkn2_ix]
                        if (start_char_ix, end_char_ix) in offsets_to_mask:
                            continue
                        else:                        
                            if threshold and attention_score > threshold:                     
                                if abs(tkn_ix - tkn2_ix) != 1 and attended_tkn.isalnum() and attended_tkn not in STOPWORDS:
                                    offsets_to_mask.append((start_char_ix, end_char_ix)) 
                                    print("{:<15} {:<10} (q: {:<10})".format(attended_tkn, round(attention_score, 4), query_tkn))
    return offsets_to_mask 

def replace_w_mask_char(input_text, offsets_to_mask):
    "Replace characters with '*' when masked in the original text."
    masked_str = input_text
    for (start_ix, end_ix) in sorted(offsets_to_mask):
        masked_str = masked_str[:start_ix] + "*"*(end_ix-start_ix) + masked_str[end_ix:]
    return masked_str

def mask_w_attention(model_name, input_text, person_to_protect, layers_to_use):
    """ Apply the attention masking method to an input string and return it 
    with words masked with '*'. """
    model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("\t Getting attention scores...")
    attentions, input_ids = get_attention_scores(model, tokenizer, input_text)
    print("\t Getting aggregated token-pair attention...")
    aggr_attn = get_token_pair_attentions(attentions, layers_to_use)
    print("\t Getting masked character offsets...")    
    offsets_to_mask = get_mask_offsets(aggr_attn, input_ids, tokenizer, person_to_protect)
    masked_str = replace_w_mask_char(input_text, offsets_to_mask)
    print("MASKED: ", masked_str)
    return offsets_to_mask
    

def mask_tab(eval_data, model_name, layers_to_use):
    "Apply masking based on attention to TAB."
    # get doc ids, texts and people to protect from TAB JSON
    print("Layers used:", layers_to_use)
    result = {}
    with open(eval_data, "r", encoding="utf-8") as f:
        all_data = json.load(f)
        print("Nr docs: ", len(all_data))
        for doc_obj in all_data:
            doc_id = doc_obj["doc_id"]
            text = doc_obj["text"]
            p_to_protect = doc_obj["task"].split(":")[-1]
            print(doc_id, p_to_protect)
            offsets_to_mask = mask_w_attention(model_name, text, p_to_protect, layers_to_use)
            result[doc_id] = offsets_to_mask
    with open("masked_" + os.path.split(eval_data)[-1], "w", encoding="utf-8") as outf:
        json.dump(result, outf)

# TO DO: run evaluation.py

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Applies an attention-based privacy masking on texts.")
    parser.add_argument("-model_name", type=str, default="roberta-base") #"allenai/longformer-base-4096" issues w BertViz
    parser.add_argument("-eval_data", type=str)
    parser.add_argument("-layers", nargs="+", type=int, default=list(range(12)))
    parser.add_argument("-show_example", action="store_true",  default=False)
    args = parser.parse_args() 
    
    if args.show_example:
        #toy example
        with open("echr_eg_synth.txt") as f:
            input_text = f.read()
        masked_str = mask_w_attention(args.model_name, input_text, "Eric Lumberjack")
        print("ORIG: ", input_text)
        print("MASKED: ", masked_str)    
        
    mask_tab(args.eval_data, args.model_name, args.layers)

    # visualize with BertViz
    # html_model_view = model_view(attentions, tokens, html_action='return') # return view to save
    # with open("model_view.html", 'w') as file:
    #     file.write(html_model_view.data)
    
    # html_head_view = head_view(attentions, tokens, html_action='return')
    # with open("head_view.html", 'w') as file:
    #     file.write(html_head_view.data)


