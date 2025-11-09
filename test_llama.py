import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def main():
    try:
        model_id = "distilgpt2"  # if this is too large, we'll switch to a lighter model next
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Loading model (this may download several GB)...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        print("Creating pipeline...")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, temperature=0.7)
        print("Running test prompt...")
        out = pipe("Hello, who are you?")[0]["generated_text"]
        print("\nJarvis says:\n", out)
    except Exception as e:
        print("=== ERROR while loading/running model ===")
        traceback.print_exc()

if __name__ == "__main__":
    main()
