## SLQA

Multi-Granularity Hierarchical Attention Fusion Networks
for Reading Comprehension and Question Answering    
If anyone would love to test the performance on SQuAD, please tell me the final score. 
You just need to change the coca_reader.py to read in dataset of SQuAD.

update:
- Rewrite the model using allennlp. It seems that it would run successfully. I'm waiting for the performance. 
- Add simple flow layer and the config file is slqa_h.json in package config.

update 12.6:
- It seems that the performance is not good enough.

update 12.7:
= The `text_field_embedder` receive a list of `token embedder` and concatenate their output in an arbitrary order. So if we use `split` wo can't make sure the part with dimension of 1024 is the output of `elmo_token_embedder`. As a result, I split it as three seperate `text_field_embedder`s.
- The fuse function in the paper is wrote as the same one. But I think the fuse layers of passage, question and self-aligned passage representation use different weight and they should not share gradient. So I use three different fuse layer.
- Change the self attention function from `D = AA^T` to `D = AWA^T`