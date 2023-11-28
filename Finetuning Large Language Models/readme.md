
# Finetuning Large Language Models


Reference: <a href="https://www.deeplearning.ai/short-courses/finetuning-large-language-models//">Finetuning Large Language Models</a>

Instructors: Sharon Zhou and Andrew Ng

1. [Introduction](#1)
    1. [What is fine-tuning and Why fine-tunning](#2)
    3. [What does fine-tuning do for the model](#3)
    4. [Prompt engineering vs. fine-tuning](#4)
    5. [Benefits of fine-tuning our own LLM](#5)
    6. [Technology stacks used in this course](#6)


<a name="1"></a>
## Introduction 

+ Key question: how can I use large language models on my own data? Taking an open-source LLM and further training it on our own data allows us to have the same interface as chatGPT for example to our own private and proprietary data.

+ While writing a prompt can be pretty good at getting an LLM to follow directions to carry out the task like extracting keywords or classifying texts as +ive or -ive sentiments, if we fine-tune  the LLM we can get the LLM to even more consistently do what we want.

+ A good analogy: prompting an LLM is to speak in a certain style like being more helpful or more polite or to be succinct vs. verbose to a specific certain extent, while fine-tuning is a good way to adjust an LLM's tone. 

+ Of course training a foundation of an LLM takes a massive amount of data, maybe hundreds of billions or more words of data and massive GPU compute resources, but with fine-tunning we can take an existing LLM and train it further on our own data

+ It'll be discussed how fine-tuning differs from prompt engineering or **retrieval augmented generation** alone, and how these techniques can be used alongside fine-tuning. We'll dive into a **specific variant of fine-tuning that's made GPT-3 into chatGPT called instruction fine-tuning**, which **teaches an LLM to follow instructions.**

![](https://github.com/DanialArab/images/blob/main/LLM/fine_tunning_llm_cource_materials.png)

<a name="2"></a>
### What is fine-tuning and Why fine-tunning

![](https://github.com/DanialArab/images/blob/main/LLM/why_fine_tunning.png)

Fine-tunning is to take a general-purpose model like GPT-3 and specializing it into something like chatGPT or taking GPT-4 and turning that into GitHub Copilot to perform auto-complete code. 

<a name="3"></a>
### What does fine-tuning do for the model

![](https://github.com/DanialArab/images/blob/main/LLM/what_does_fine_tunning_do.png)

In addition to learning new information, fine-tuning can also help steer the model to **more consistent outputs or more consistent behavior.** 

![](https://github.com/DanialArab/images/blob/main/LLM/fine_tunned_vs._non_fine_tunned_model.png)

<a name="4"></a>
### Prompt engineering vs. fine-tuning 

![](https://github.com/DanialArab/images/blob/main/LLM/prompt_vs_fine_tunning.png)

Hallucination = the model may make stuff up (problem of prompt engineering)

<a name="5"></a>
### Benefits of fine-tuning our own LLM

![](https://github.com/DanialArab/images/blob/main/LLM/benefits_of_fine_tuning.png)

<a name="6"></a>
### Technology stacks used in this course

+ Pytorch (Meta)
+ Huggingface
+ Llama library(Lamini(
