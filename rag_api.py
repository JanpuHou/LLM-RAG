import lamini
# go to https://app.lamini.ai/account to get your api key
lamini.api_key = "your api key"

llm = lamini.Lamini("meta-llama/Meta-Llama-3.1-8B-Instruct")
print(llm.generate("""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

How are you?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""))