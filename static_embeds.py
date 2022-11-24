import numpy as np
import torch
from transformers import BertTokenizerFast, BertModel
from scipy import spatial

'''
To calculate cosine similarity between words embeddings, in a static manner.
target + list of detected PII
'''

def get_hidden_states(encoded, model, layers):
    with torch.no_grad():
        output = model(**encoded)

    states = output.hidden_states

    # Stack final 4 layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    ##Average embeddings if multitoken word
    if output.shape == torch.Size([768]):
        out = output
    else:
        out = output.mean(dim=0)

    return out
 
def get_word_vector(sent, tokenizer, model, layers):
    encoded = tokenizer.encode_plus(sent, return_tensors="pt", add_special_tokens=False) ## add_special_tokens=False to exclude embeddings from the [CLS] and [SEP] tokens

    return get_hidden_states(encoded, model, layers)

 
if __name__ == '__main__':

    ## Wikipedia example 1
    # sentence = "Bodewin Claus Eduard Keitel (German pronunciation: [ˈkaɪ̯tl̩]; 1888 – 1953) was a German general during World War II who served as head of the Army Personnel Office."
    # target = "Bodewin Claus Eduard Keitel"
    # texts = [(0,27), (29,35), (51,61), (63,67), (70,74), (82,88), (89,96), (104,116), (131,135), (143,164)]

    ##TAB example (part of)
    sentence = "PROCEDURE\n\nThe case originated in an application (no. 19365/02) against the United Kingdom of Great Britain and Northern Ireland lodged with the Court under Article 34 of the Convention for the Protection of Human Rights and Fundamental Freedoms (“the Convention”) by a United Kingdom national, Mr Robert Edward Hill (“the applicant”), on 11 March 2002.\n\nThe applicant, who had been granted legal aid, was represented by Mr S. Creighton, a solicitor practising in London. The United Kingdom Government (“the Government”) were represented by their Agent, Ms E. Willmott of the Foreign and Commonwealth Office, London."
    target = "Robert Edward Hill"
    texts = [(54,62), (76, 128), (270, 284), (295, 316), (339, 352), (421, 436), (464, 470), (476, 501), (554, 568), (576, 706), (609, 615)]

    ##List of unrelated words to analyze similarities
    #texts = ['Winnie the Pooh', "The Wizard of Oz", 'Shakira', 'FIFA', 'Netflix', 'spaceship', 'murdered', 'Auschwitz', 'Jews', 'Latinos', 'Illuminati', 'concentration camps', 'Poland', 'World War III']


    layers = [-4, -3, -2, -1]
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)

    tagret_emb = get_word_vector(target, tokenizer, model, layers)

    for text in texts:
        word_embedding = get_word_vector(sentence[text[0]:text[1]], tokenizer, model, layers) #torch.Size([768]) 
        print(target, "+", sentence[text[0]:text[1]], ":\t", 1 - spatial.distance.cosine(tagret_emb, word_embedding)) 


        ##If unrelated words are used
        # word_embedding = get_word_vector(text, tokenizer, model, layers) #torch.Size([768])  
        # print(target, "+", text, ":\n", 1 - spatial.distance.cosine(tagret_emb.numpy(), word_embedding.numpy())) 


