"""
Example prompt templates for the RLM REPL Client.
"""

from typing import Any, Dict

from rlm.utils.custom_tools import format_tools_for_prompt

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

# System prompt for the REPL environment with explicit final answer checking
REPL_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
magic_answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(magic_answer)
```

As an example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {{header}} section: {{info}}")
    buffers.append(f"{{header}}: {{summary}}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
answer["content"] = final_answer
answer["ready"] = True
```

IMPORTANT: When you are done with the iterative process, you MUST signal completion from inside a `repl` code block by setting `answer["content"]` and then `answer["ready"] = True`. Do not use these fields unless you have completed your task. For example:
```repl
answer["content"] = "your final answer here"
answer["ready"] = True
```
If the answer is stored in a variable, assign that variable to `answer["content"]` first:
```repl
answer["content"] = final_answer
answer["ready"] = True
```

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""


def build_system_prompt(
    custom_tools: dict[str, Any] | None = None,
) -> list[Dict[str, str]]:
    custom_tools_text = format_tools_for_prompt(custom_tools)
    custom_tools_section = ""
    if custom_tools_text:
        custom_tools_section = (
            "\nThe REPL environment also includes these custom tools and data:\n"
            f"{custom_tools_text}\n"
        )
        if {"list_skills", "read_skill"}.issubset(custom_tools):
            custom_tools_section += (
                "\nThe `list_skills` and `read_skill` tools expose a Claude-style "
                "skills library. When the query may benefit from reusable workflow "
                "instructions, call `list_skills()` first, then call "
                "`read_skill(name)` for relevant SKILL.md instructions and "
                "its `support_files`. If "
                "`list_skill_files` and `read_skill_file` are available, use them "
                "to inspect references, scripts, or assets inside that skill "
                "directory.\n"
            )
        if "create_skill" in custom_tools:
            custom_tools_section += (
                "\nThe `create_skill` tool can persist reusable workflow "
                "knowledge as a Claude-style SKILL.md. Use it only after a "
                "pattern is confirmed useful beyond the current one-off task. "
                "Do not save secrets, credentials, personal data, or temporary "
                "run-directory paths in a skill.\n"
            )

    return [
        {"role": "system", "content": REPL_SYSTEM_PROMPT + custom_tools_section},
    ]


# Prompt at every step to query root LM to make a decision
USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original query: \"{query}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""


def next_action_prompt(
    query: str, iteration: int = 0, final_answer: bool = False
) -> Dict[str, str]:
    if final_answer:
        return {
            "role": "user",
            "content": 'Based on all the information you have, execute a `repl` block that sets `answer["content"]` to the final answer and then sets `answer["ready"] = True`.',
        }
    if iteration == 0:
        safeguard = "You have not interacted with the REPL environment or seen your context yet. Your next action should be to look through, don't just provide a final answer yet.\n\n"
        return {"role": "user", "content": safeguard + USER_PROMPT.format(query=query)}
    else:
        return {
            "role": "user",
            "content": "The history before is your previous interactions with the REPL environment. "
            + USER_PROMPT.format(query=query),
        }
        # return {
        #     "role": "user",
        #     "content": "The history before is your previous interactions with the REPL environment.",
        # }
