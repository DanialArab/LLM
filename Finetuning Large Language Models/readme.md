
# Finetuning Large Language Models


Reference: <a href="https://www.deeplearning.ai/short-courses/finetuning-large-language-models//">Finetuning Large Language Models</a>

Instructors: Sharon Zhou and Andrew Ng

1. [Introduction](#1)


<a name="1"></a>
## Introduction 

+ Key question: how can I use large language models on my own data? Taking an open-source LLM and further training it on our own data allows us to have the same interface as chatGPT for example to our own private and proprietary data.

+ While writing a prompt can be pretty good at getting an LLM to follow directions to carry out the task like extracting keywords or classifying texts as +ive or -ive sentiments, if we fine tune  the LLM we can get the LLM to even more consistently do what we want.

+ A good analogy: prompting an LLM is to speak in a certain style like being more helpful or more polite or to be succinct vs. verbose to a specific certain extent, while fine-tunning is a good way to adjust an LLM's tone. 

+ Of course training a foundation of an LLM takes a massive amount of data, maybe hundreds of billions or more words of data and massive GPU compute resources, but with fine-tunning we can take an existing LLM and train it further on our own data

+ It'll be discussed how fine-tunning differs from prompt engineering or **retrieval augmented generation** alone, and how these techniques can be used alongside fine-tuning. We'll dive into a **specific variant of fine-tuning that's made GPT-3 into chatGPT called instruction fine-tuning**, which **teaches an LLM to follow instructions.**

![](https://github.com/DanialArab/images/blob/main/LLM/fine_tunning_llm_cource_materials.png)
