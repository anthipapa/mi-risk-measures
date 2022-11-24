from scipy import spatial
import numpy as np
import torch
from transformers import BertTokenizerFast, BertModel

'''
To calculate cosine similarity between words embeddings, in a contextualised manner.
target + list of detected PII
'''

def average_embeds(embeds):
    embeds = torch.stack((embeds))
    return embeds.mean(dim=0)

def get_hidden_states(encoded, model, layers):

    with torch.no_grad():
        output = model(encoded['input_ids'], encoded['token_type_ids'], encoded['attention_mask'])

    states = output.hidden_states

    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    return output 


def get_word_vector(sent, tokenizer, model, layers):
    encoded = tokenizer.encode_plus(sent, return_tensors="pt", return_offsets_mapping = True)
    offsets = encoded['offset_mapping']

    return (encoded, model, layers, offsets)

if __name__ == '__main__':
    layers = [-4, -3, -2, -1]
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)

    mapping = {}

    ## Wikipedia example 1
    # sent = "Jim Moodie (1 December 1905 – 6 March 1980) was a former Australian rules footballer who played with Melbourne in the Victorian Football League (VFL)."
    # annots = [(0,10), (12, 27), (30, 42), (57, 67), (74, 84), (101, 110), (118, 149)]  # annotation text spans
    # target = [(0,10)] 


    ##Fake Wikipedia example
    # annots = [(12, 27), (30, 41), (57,62), (69,77), (94, 106), (110, 123)] ##fake ones
    # sent = "Jim Moodie (23 January 1992 – 5 March 2020) was a former Greek rules sculptor who played with South Africa in World War III."
    # target = [(0,10)]


    ## Wikipedia example 2
    # sent = "Helen Clevenger (November 4, 1917 – July 16, 1936) was an American college student murdered in Asheville, North Carolina on July 16, 1936."
    # target = [(0,14)]
    # annots = [(17, 33), (36, 49), (58, 66), (67, 82), (95, 120), (124, 137)]

    ##TAB example (part of)
    sent = "PROCEDURE\n\nThe case originated in an application (no. 19365/02) against the United Kingdom of Great Britain and Northern Ireland lodged with the Court under Article 34 of the Convention for the Protection of Human Rights and Fundamental Freedoms (“the Convention”) by a United Kingdom national, Mr Robert Edward Hill (“the applicant”), on 11 March 2002.\n\nThe applicant, who had been granted legal aid, was represented by Mr S. Creighton, a solicitor practising in London. The United Kingdom Government (“the Government”) were represented by their Agent, Ms E. Willmott of the Foreign and Commonwealth Office, London."
    target = [(298, 316)]
    annots = [(54,62), (76, 128), (270, 284), (295, 316), (339, 352), (421, 436), (464, 470), (476, 501), (554, 568), (576, 706), (609, 615)]


    encoded, model, layers, offsets = get_word_vector(sent, tokenizer, model, layers) 
    word_embeds = get_hidden_states(encoded, model, layers) ##Get embeddings for the entire text
    off = offsets.numpy().tolist()[0]
    
    for offset, embed in zip(off, word_embeds):
        mapping[tuple(offset)] = embed      ##Create a mapping between offsets and their embeddings

    ##Get offsets / keys from mapping that correspond to PII offsets, match offset subsets to annotation offsets
    range_t = range(target[0][0], target[0][1])
    l_t = []
    for key, value in mapping.items():
        if key!= (0,0):     ##Exclude SEP and CLS token embeddings
            range2 = range(key[0], key[1])
            if set(range2).issubset(set(range_t)):
                l_t.append(value)

    target_embed = average_embeds(l_t)

    for annotation in annots:
        if annotation != target[0]:
            range1 = range(annotation[0], annotation[1])
            l = []
            for key, value in mapping.items():
                if key!= (0,0):     ##Exclude SEP and CLS token embeddings
                    range2 = range(key[0], key[1])
                    if set(range2).issubset(set(range1)):
                        l.append(value)

            mean = average_embeds(l)

            print(sent[target[0][0]:target[0][1]], "+", sent[annotation[0]:annotation[1]], ":\t", 1 - spatial.distance.cosine(target_embed, mean))

