from memory_profiler import profile

from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from timeit import timeit

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
pipe = pipeline(task='text2text-generation', model=model, tokenizer = tokenizer)

@profile
def perform_inference():
    return pipe('Translate English to French: Hi, How are you ?')

if __name__ == '__main__':
    perform_inference()