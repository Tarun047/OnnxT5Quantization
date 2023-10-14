from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from timeit import timeit

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
pipe = pipeline(task='text2text-generation', model=model, tokenizer = tokenizer)

globals =dict(map(
    lambda x: (x, eval(x)),
    dir()
))

time_taken = timeit(
    globals = globals,
    number = 5,
    stmt="pipe('Translate English to French: Hi, How are you ?')",
)

print(f"It took about {time_taken} seconds")



