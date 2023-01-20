1. MI & pre-trained “static” BERT embeddings --> *static_embeds.py*
2. MI & contextualized BERT embeddings --> *context_embeds.py*
3. MI through LM probabilities --> *mi_probabilities.py*

### run with all layers
python run_attention_cl.py -eval_data echr_dev.json 

### run with selected layers 
python run_attention_cl.py -eval_data echr_dev.json -layers 9 10 11
