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

