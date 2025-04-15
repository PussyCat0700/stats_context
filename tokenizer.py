from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if __name__ == '__main__':
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    print(encoded_input)