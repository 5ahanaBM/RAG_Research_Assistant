from transformers import pipeline
from retriever import retrieve

# Load the local text generation model

# generator = pipeline("text-generation", model="distilgpt2", device=0)
# generator = pipeline("text2text-generation", model="t5-small", device=0)
generator = pipeline("text2text-generation", model="google/flan-t5-base", device=0)

'''
Use Better Prompting with T5-small or other small models
  a. Concise and direct — a natural prompt for instruction-tuned models like flan-t5-base
  b. Best suited for small-to-mid size LLMs expecting a plain, single-context block
  c. Keeps the prompt short (less token cost)
'''
def format_prompt(query, context_chunks):
    context = " ".join(chunk['text'].strip() for chunk in context_chunks)
    return f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}\nAnswer:"

'''
1. More structured, chunk-by-chunk clarity
2. Useful when you want to:
  a. Highlight chunk boundaries
  b. Refer to specific chunks (e.g., in multi-hop or citation mode)
3. Potentially better for chat-based models (like ChatGPT or Claude) that benefit from formatting cues
'''

# def format_prompt(query, context_chunks):
#     """Format the prompt for the generator."""
#     prompt = "Context:\n"
#     for i, chunk in enumerate(context_chunks):
#         prompt += f"[{i+1}] {chunk['text'].strip()}\n"
#     prompt += f"\nQuestion: {query}\nAnswer:"
#     return prompt

def generate_answer(query, top_k=3, max_tokens=150):
    # Get top-k relevant chunks
    context = retrieve(query, top_k=top_k)
    prompt = format_prompt(query, context)
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=True)
    # output = generator(prompt, max_new_tokens=max_tokens, do_sample=False)
    return output[0]['generated_text']

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() in ['exit', 'quit']:
            break

        answer = generate_answer(question)
        print(f"\nAnswer:\n{answer}\n")
        print("\n" + "=" * 50 + "\n")