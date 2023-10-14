from memory_profiler import profile
from optimum.pipelines import pipeline
from transformers import T5Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

model = ORTModelForSeq2SeqLM.from_pretrained('t5_onnx_small_quantized')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
pipe = pipeline(task='text2text-generation',model=model, tokenizer=tokenizer)

@profile
def perform_inference():
    return pipe('Translate English to French: Hi, How are you ?')

if __name__ == '__main__':
    perform_inference()