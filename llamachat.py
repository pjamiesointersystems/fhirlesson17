import lmstudio as lms

with lms.Client() as client:
    model = client.llm.model("llama-3.2-1b-instruct")
    result = model.respond("What is the meaning of life?")

    print(result)

















# if __name__ == "__main__":
    

#     print(result)