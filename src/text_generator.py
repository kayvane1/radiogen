import json
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import json

with open('config/textgen.json', 'r') as f:
    config = json.load(f)

def generate_text(prompt,temperature=config.get("temperature"), max_length=config.get("max_length")):
  model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
  tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

  input_ids = tokenizer(prompt, return_tensors="pt").input_ids

  gen_tokens = model.generate(input_ids, do_sample=True, temperature=temperature, max_length=max_length)
  gen_text = tokenizer.batch_decode(gen_tokens)[0]
  
  return gen_text
