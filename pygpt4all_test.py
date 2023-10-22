from pygpt4all.models.gpt4all import GPT4All

def new_text_callback(text):
    print(text, end="")

model = GPT4All('./models/gpt4all-converted.bin')
model.generate("Las vacas beben leche. que beben las vacas? las vacas beben leche ", n_predict=55, new_text_callback=new_text_callback)

