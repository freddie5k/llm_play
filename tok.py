from importlib.metadata import version
import tiktoken
print("titoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")
text = ("il mio primo testo convertito in token")
integers = tokenizer.encode(text)
print(integers)

string = tokenizer.decode(integers)
print(string)

with open("ver.txt","r", encoding="utf-8") as f: raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:       {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i] 
    print(tokenizer.decode(context),"---->", tokenizer.decode([desired]))
