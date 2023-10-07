# ChatGPT Prompt Engineering for Developers

Reference: <a href="https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/">ChatGPT Prompt Engineering for Developers by Isa Fulford and Andrew Ng</a>

1. [Introduction](#1)
2. [Guidelines](#2)
   1. [Tactic 1: Use delimiters to clearly indicate distinct parts of the input](#3)
   2. [Tactic 2: Ask for a structured output](#4)
   3. [Tactic 3: Ask the model to check whether conditions are satisfied](#5)
   4. [Tactic 4: "Few-shot" prompting](#6)

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

<a name="3"></a>
### Tactic 1: Use delimiters to clearly indicate distinct parts of the input

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

<a name="4"></a>
### Tactic 2: Ask for a structured output

prompt = f"""
Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
"""
response = get_completion(prompt)
print(response)

returns me back:

        {
          "books": [
            {
              "book_id": 1,
              "title": "The Enigma of Elysium",
              "author": "Evelyn Sinclair",
              "genre": "Mystery"
            },
            {
              "book_id": 2,
              "title": "Whispers in the Wind",
              "author": "Nathaniel Blackwood",
              "genre": "Fantasy"
            },
            {
              "book_id": 3,
              "title": "Echoes of the Past",
              "author": "Amelia Hart",
              "genre": "Romance"
            }
          ]
        }

<a name="5"></a>
### Tactic 3: Ask the model to check whether conditions are satisfied

        text_1 = f"""
        Making a cup of tea is easy! First, you need to get some \ 
        water boiling. While that's happening, \ 
        grab a cup and put a tea bag in it. Once the water is \ 
        hot enough, just pour it over the tea bag. \ 
        Let it sit for a bit so the tea can steep. After a \ 
        few minutes, take out the tea bag. If you \ 
        like, you can add some sugar or milk to taste. \ 
        And that's it! You've got yourself a delicious \ 
        cup of tea to enjoy.
        """
        prompt = f"""
        You will be provided with text delimited by triple quotes. 
        If it contains a sequence of instructions, \ 
        re-write those instructions in the following format:
        
        Step 1 - ...
        Step 2 - …
        …
        Step N - …
        
        If the text does not contain a sequence of instructions, \ 
        then simply write \"No steps provided.\"
        
        \"\"\"{text_1}\"\"\"
        """
        response = get_completion(prompt)
        print("Completion for Text 1:")
        print(response)

returns me back:

        Completion for Text 1:
        Step 1 - Get some water boiling.
        Step 2 - Grab a cup and put a tea bag in it.
        Step 3 - Once the water is hot enough, pour it over the tea bag.
        Step 4 - Let it sit for a bit so the tea can steep.
        Step 5 - After a few minutes, take out the tea bag.
        Step 6 - If you like, add some sugar or milk to taste.
        Step 7 - Enjoy your delicious cup of tea.

another example:

        text_2 = f"""
        The sun is shining brightly today, and the birds are \
        singing. It's a beautiful day to go for a \ 
        walk in the park. The flowers are blooming, and the \ 
        trees are swaying gently in the breeze. People \ 
        are out and about, enjoying the lovely weather. \ 
        Some are having picnics, while others are playing \ 
        games or simply relaxing on the grass. It's a \ 
        perfect day to spend time outdoors and appreciate the \ 
        beauty of nature.
        """
        prompt = f"""
        You will be provided with text delimited by triple quotes. 
        If it contains a sequence of instructions, \ 
        re-write those instructions in the following format:
        
        Step 1 - ...
        Step 2 - …
        …
        Step N - …
        
        If the text does not contain a sequence of instructions, \ 
        then simply write \"No steps provided.\"
        
        \"\"\"{text_2}\"\"\"
        """
        response = get_completion(prompt)
        print("Completion for Text 2:")
        print(response)

returns me back:


        Completion for Text 2:
        No steps provided.

<a name="6"></a>
### Tactic 4: "Few-shot" prompting

        prompt = f"""
        Your task is to answer in a consistent style.
        
        <child>: Teach me about patience.
        
        <grandparent>: The river that carves the deepest \ 
        valley flows from a modest spring; the \ 
        grandest symphony originates from a single note; \ 
        the most intricate tapestry begins with a solitary thread.
        
        <child>: Teach me about resilience.
        """
        response = get_completion(prompt)
        print(response)

returns me back:

        <grandparent>: Resilience is like a mighty oak tree that withstands the strongest storms, bending but never breaking. It is the ability to bounce back from adversity, to find strength in the face of challenges, and to persevere even when the odds seem insurmountable. Just as a diamond is formed under immense pressure, resilience is forged through the trials and tribulations of life.

        
