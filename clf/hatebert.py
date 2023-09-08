from transformers import pipeline

toxigen_hatebert = pipeline("text-classification", model="tomh/toxigen_hatebert", tokenizer="bert-base-cased")



print(toxigen_hatebert("black people are inferior"))