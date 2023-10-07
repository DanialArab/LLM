# ChatGPT Prompt Engineering for Developers

Reference: <a href="https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/">ChatGPT Prompt Engineering for Developers by Isa Fulford and Andrew Ng</a>

1. [Introduction](#1)
2. [Guidelines](#2)

<a name="1"></a>
## Introduction 

+ The real power of LLMs is to use them as a developer tool like making API calls to LLMs to quickly build software applications.

![](https://github.com/DanialArab/images/blob/main/LLM/LLM_types.PNG)

+ The way that instruction-tuned LLMs are typically trained is you start off with a base LLM that's been trained on a huge amount of text data and further train it, further fine-tune it with inputs and outputs that are instructions and good attempts to follow those instructions, and then often further refine using a technique called RLHF, reinforcement learning from human feedback, to make the system better able to be helpful and follow instructions.
+ This course will focus on best practices for **instruction-tuned LLMs**, which is what we recommend you use for most of your applications.
+ When you use an instruction-tuned LLM, think of giving instructions to another person, say someone that's smart but doesn't know the specifics of your task. So, when an LLM doesn't work, sometimes it's because the instructions weren't clear enough.
+  **Knowing how to be clear and specific is an important principle of prompting LLMs. The second principle of prompting that is giving the LLM time to 
think.**

<a name="2"></a>
## Guidelines

**Prompting Principles**: 

+ Principle 1: Write clear and specific instructions
+ Principle 2: Give the model time to “think”

![](https://github.com/DanialArab/images/blob/main/LLM/prompting%20principles.PNG)

![](https://github.com/DanialArab/images/blob/main/LLM/principle_1.PNG)

    import openai
    import os
    
    openai.api_key = os.getenv('MY_OPENAI_API_KEY')
    
    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]
    
    text = f"""
    You should express what you want a model to do by \ 
    providing instructions that are as clear and \ 
    specific as you can possibly make them. \ 
    This will guide the model towards the desired output, \ 
    and reduce the chances of receiving irrelevant \ 
    or incorrect responses. Don't confuse writing a \ 
    clear prompt with writing a short prompt. \ 
    In many cases, longer prompts provide more clarity \ 
    and context for the model, which can lead to \ 
    more detailed and relevant outputs.
    """
    prompt = f"""
    Summarize the text delimited by triple backticks \ 
    into a single sentence.
    ```{text}```
    """
    response = get_completion(prompt)
    print(response)

returns me back:

    To guide a model towards the desired output and reduce irrelevant or incorrect responses, it is important to provide clear and specific instructions, which can be achieved through longer prompts that offer more clarity and context.

