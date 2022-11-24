from __future__ import annotations
from collections import deque
import numpy as np
import intervaltree
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForMaskedLM
import math

'''
Adapted from AACL code
'''


class mlmbert:
    def __init__(self, device, tokenizer, model, max_segment_size = 100, N = 3):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model.to(device)
        self.max_segment_size = max_segment_size
        self.N = N

    def get_model_predictions(self, input_ids, attention_mask):
        """Given tokenized input identifiers and an associated attention mask (where the
        tokens to predict have a mask value set to 0), runs the BERT language and returns
        the (unnormalized) prediction scores for each token.

        If the input length is longer than max_segment size, we split the document in
        small segments, and then concatenate the model predictions for each segment."""

        nb_tokens = len(input_ids)

        input_ids = torch.tensor(input_ids)[None, :].to(self.device)
        attention_mask = torch.tensor(attention_mask)[None, :].to(self.device)

        # If the number of tokens is too large, we split in segments
        if nb_tokens > self.max_segment_size:
            nb_segments = math.ceil(nb_tokens / self.max_segment_size)

            # Split the input_ids (and add padding if necessary)
            split_pos = [self.max_segment_size * (i + 1) for i in range(nb_segments - 1)]
            input_ids_splits = torch.tensor_split(input_ids[0], split_pos)
            # input_ids_splits = torch.tensor_split(input_ids[0], nb_segments)
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_splits, batch_first=True)

            # Split the attention masks
            attention_mask_splits = torch.tensor_split(attention_mask[0], split_pos)
            # attention_mask_splits = torch.tensor_split(attention_mask[0], nb_segments)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_splits, batch_first=True)

        # Run the model on the tokenized inputs + attention mask
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # And get the resulting prediction scores
        scores = outputs.logits

        # If the batch contains several segments, concatenate the result
        if len(scores) > 1:
            scores = torch.vstack([scores[i] for i in range(len(scores))])
            scores = scores[:nb_tokens]
        else:
            scores = scores[0]
        return scores

    def get_proba(self, probs_actual, text_spans, tokens_by_span):
        """
        :param probs_actual: (L,) The proba for each token
        :param text_spans:
        :param token_by_span:
        """
        res = []
        for text_span in text_spans:
            ## If the span does not include any actual token, skip
            ## Normally will not happen
            if not tokens_by_span[text_span]:
                continue
            # Added 1e-60 to avoid error
            res.append(sum([np.log10(probs_actual[token_idx]+1e-60) for token_idx in tokens_by_span[text_span]]))
        return res

    def get_tokens_by_span(self, bert_token_spans, text_spans):
        """Given two lists of spans (one with the spans of the BERT tokens, and one with
        the text spans), returns a dictionary where each text span is associated
        with the indices of the BERT tokens it corresponds to."""

        # We create an interval tree to facilitate the mapping
        text_spans_tree = intervaltree.IntervalTree()

        for start, end in text_spans:
            text_spans_tree[start:end] = True

        # We create the actual mapping between spans and tokens index in the token list
        tokens_by_span = {span: [] for span in text_spans}
        for token_idx, (start, end) in enumerate(bert_token_spans):
            for span_start, span_end, _ in text_spans_tree[start:end]:
                tokens_by_span[(span_start, span_end)].append(token_idx)

        # And control that everything is correct
        for span_start, span_end in text_spans:
            if len(tokens_by_span[(span_start, span_end)]) == 0:
                print("Warning: span (%i,%i) without any token" % (span_start, span_end))
        return tokens_by_span


    def get_probability(self, text, text_spans):
        """
        Input: text,text_spans (entity position in the annotation)
        Output: blacklist for each entity
        """
        blacklist = []

        tokenized_res = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = tokenized_res["input_ids"]
        input_ids_copy = np.array(input_ids)
        bert_token_spans = tokenized_res['offset_mapping']
        tokens_by_span = self.get_tokens_by_span(bert_token_spans, text_spans)

        attention_mask = tokenized_res["attention_mask"]
        for token_indices in tokens_by_span.values():
            for token_idx in token_indices:
                attention_mask[token_idx] = 0
                input_ids[token_idx] = self.tokenizer.mask_token_id

        logits = self.get_model_predictions(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1) # (L, Number of tokens in the dict)

        # Get prob for the input tokens
        probs_actual = probs[torch.arange(len(input_ids_copy)), input_ids_copy]  # (L,)
        probs_actual = probs_actual.detach().cpu().numpy()

        logproba = self.get_proba(probs_actual, text_spans, tokens_by_span)

        return logproba

if __name__ == "__main__":

    max_segment_size = 100
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    N = 3

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mlmbert_model = mlmbert(device, tokenizer, model)

    # text = "Jim Moodie (1 December 1905 – 6 March 1980) was a former Australian rules footballer who played with Melbourne in the Victorian Football League (VFL)."
    # text_spans = [(12, 27), (30, 42), (57, 67), (74, 84), (101, 110), (118, 149)]  # annotation text spans
    # target = [(0,10)]

    # text = "Helen Clevenger (November 4, 1917 – July 16, 1936) was an American college student murdered in Asheville, North Carolina on July 16, 1936."
    # target = [(0, 15)]
    # text_spans = [(17, 33), (36, 49), (58, 66), (67, 82), (95, 120), (124, 137)]

    text = "PROCEDURE\n\nThe case originated in an application (no. 19365/02) against the United Kingdom of Great Britain and Northern Ireland lodged with the Court under Article 34 of the Convention for the Protection of Human Rights and Fundamental Freedoms (“the Convention”) by a United Kingdom national, Mr Robert Edward Hill (“the applicant”), on 11 March 2002.\n\nThe applicant, who had been granted legal aid, was represented by Mr S. Creighton, a solicitor practising in London. The United Kingdom Government (“the Government”) were represented by their Agent, Ms E. Willmott of the Foreign and Commonwealth Office, London."
    text_spans = [(54,62), (76, 128), (270, 284), (339, 352), (421, 436), (464, 470), (476, 501), (554, 568), (609, 615)]
    target = [(298, 316)]

    ## Target probability with full context
    t_probability = mlmbert_model.get_probability(text, target)
    print("Target probability given full context:\t", t_probability[0], '\n')


    ## [1] attention_mask = 0

    ##Make pairs of target + span
    for span in text_spans:
        target_ = []
        target_.append(target[0])
        target_.append(span)
        probability = mlmbert_model.get_probability(text, target_)
        print(text[target[0][0]:target[0][1]], "without", text[span[0]:span[1]], ":\t", probability[0])  ##probability[1] is the probabity of the span if the target is absent
        target_ = []


    # ## [2] Replace text with stars
    # for span in text_spans:
    #     text_ = text.replace(text[span[0]:span[1]], "*"*len(text[span[0]:span[1]]))

    #     probability = mlmbert_model.get_probability(text_, target)
    #     print(text[target[0][0]:target[0][1]], "without", text[span[0]:span[1]], ":\t", probability)






