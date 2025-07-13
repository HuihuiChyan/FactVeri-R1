# ## [参照 Search-O1]
# 定义在推理流程中使用的特殊标记
# BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
# END_SEARCH_QUERY = "<|end_search_query|>"
# BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
# END_SEARCH_RESULT = "<|end_search_result|>"

BEGIN_SEARCH_QUERY = "<search>"
END_SEARCH_QUERY = "</search>"
BEGIN_SEARCH_RESULT = "<information>"
END_SEARCH_RESULT = "</information>"

def get_agentic_verification_instruction(claim: str, max_search_limit: int = 10) -> str:

#     # 结合了Search-O1的自主搜索指令和您现有工作的事实验证指令
#     instruction = f"""You are an expert fact-checking assistant. Your goal is to determine if the following claim is Supported or Unsupported based on reliable web search results.

# You have a special tool to help you gather evidence. If you believe you need more information to make a confident decision, you can perform a web search.

# - To perform a search: write `{BEGIN_SEARCH_QUERY}` your query keywords here `{END_SEARCH_QUERY}`.
# - The system will then provide you with a list of search results, including titles and snippets, in the format `{BEGIN_SEARCH_RESULT}` ...search results... `{END_SEARCH_RESULT}`.

# You can perform this search action multiple times if needed, and you have a maximum of {max_search_limit} search attempts. However, you should save your times of searching by only search for what is truly needed for an online check. Use your searches strategically to gather sufficient evidence.

# Once you have enough information and have completed all necessary searches, you must conclude your reasoning and provide a final decision.

# Below are the definitions of the two categories:

# Supported: A claim is supported by the search results if everything in the claim is supported and nothing is contradicted by the search results. There can be some search results that are not fully related to the claim.
# Unsupported: If a claim is not supported by the search results, mark it as unsupported.

# **Your final decision must be only the word 'Supported' or the word 'Unsupported'. Do not add any other text or explanation after your final decision.**

# Now, begin your work for the following claim:

# ### Claim to Verify
# ```
# {claim}
# ```
# """

    instruction = f"""You are an expert fact-checking assistant. Your goal is to determine if the following claim is real or not.
You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by {BEGIN_SEARCH_QUERY} query {END_SEARCH_QUERY} and it will return the top searched results between {BEGIN_SEARCH_RESULT} and {END_SEARCH_QUERY}.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer 'Real' or 'Not Real' inside <answer> and </answer>, without detailed illustrations. For example, <answer> Real </answer>. 
Now, begin your work for the following claim:
### Claim to Verify
```
{claim}
```"""
    return instruction


def get_baseline_verification_instruction(claim: str) -> str:

    instruction = f"""You are an expert fact-checking assistant. Your goal is to determine if the following claim is real or not based solely on your internal knowledge.
You must conduct reasoning inside <think> and </think> to analyze the claim. Consider what you know about the topic, any relevant facts, and whether the claim aligns with your knowledge.
After reasoning, provide your final answer as 'Real' or 'Not Real' inside <answer> and </answer>, without detailed illustrations. For example, <answer> Real </answer>.
Now, begin your work for the following claim:
### Claim to Verify
```
{claim}
```"""
    return instruction
