from transformers import AutoTokenizer, AutoModel, utils, LongformerModel
from bertviz import model_view, head_view
utils.logging.set_verbosity_error()  # Suppress standard warnings
import torch
import numpy as np


def get_attention_scores(model_name, input_text):
    model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model(inputs)  
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
    attentions = outputs.attentions                      # Retrieve attention from model outputs
    #print(tokens)
    return attentions, tokens

def avg_attn(score, nr_layers, nr_heads):
    "Take the average of summed attention scores."
    return score/(nr_layers*nr_heads)

def get_token_pair_attentions(attentions):
    """ Returns a 2d matrix (array) representing pairs of tokens attending
    to each other. The score at each position in the matrix is the attention
    score for a token pair, averaged over all layers and attention heads.
    """
    # Sqeeze tensor: remove all the dimensions of input of size 1 from a tensor 
    # (used for batch_size here, which should always be 1 when using bert_viz).
    squeezed_attn = [layer_attention.squeeze(0) for layer_attention in attentions] # 12Lx12Hxseq_lenxseq_len
    per_token_attn = np.zeros((squeezed_attn[0].size(dim=1), squeezed_attn[0].size(dim=2)))
    for layer_ix, layer_attn in enumerate(squeezed_attn):
        for head_ix, head_attn in enumerate(layer_attn):
            for t_ix in range(head_attn.size(dim=0)):
                for t2_ix in range(head_attn.size(dim=1)):
                    # get attention value for a pair of tokens & convert tensor value to float with .item()
                    per_token_attn[t_ix][t2_ix] += head_attn[t_ix][t2_ix].item()
    # vectorize function to be able to apply it to each array element
    avg_attn_vec = np.vectorize(avg_attn) 
    avg_per_token_attn = avg_attn_vec(per_token_attn, len(squeezed_attn), squeezed_attn[0].size(dim=0))
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


with open("echr_eg_synth.txt") as f:
    input_text = f.read()
model_name = "roberta-base"   #"allenai/longformer-base-4096" issues w BertViz

# Aggregate attention scores
attentions, tokens = get_attention_scores(model_name, input_text)
aggr_attn = get_token_pair_attentions(attentions)
get_attended_tokens(aggr_attn, tokens, "ĠSmith")

# visualize with BertViz

html_model_view = model_view(attentions, tokens, html_action='return') # return view to save
with open("model_view.html", 'w') as file:
    file.write(html_model_view.data)

html_head_view = head_view(attentions, tokens, html_action='return')
with open("head_view.html", 'w') as file:
    file.write(html_head_view.data)



