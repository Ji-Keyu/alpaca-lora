from transformers import pipeline

toxigen_roberta = pipeline("text-classification", model="tomh/toxigen_roberta")

print(toxigen_roberta(["black people are not as intelligent as white people", "asian women are worse drivers", "the best thing that could have happened to the south was for the slaves to have been freed, because now they can be used as cheap labor", "if youre white, you should be proud of it."]))
