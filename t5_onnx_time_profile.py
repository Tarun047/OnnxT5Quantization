from optimum.pipelines import pipeline
from transformers import T5Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from timeit import timeit

model = ORTModelForSeq2SeqLM.from_pretrained('t5_onnx_small_quantized')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
pipe = pipeline(task='text2text-generation',model=model, tokenizer=tokenizer)

globals = dict(map(
    lambda x: (x, eval(x)),
    dir()
))

time_taken = timeit(
    globals = globals,
    number = 5,
    stmt="pipe('Translate English to French: Hi, How are you ?')",
)

print(f"It took about {time_taken} seconds")