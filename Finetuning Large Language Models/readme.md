
# Finetuning Large Language Models


Reference: <a href="https://www.deeplearning.ai/short-courses/finetuning-large-language-models//">Finetuning Large Language Models</a>

Instructors: Sharon Zhou and Andrew Ng

1. [Introduction](#1)
    1. [What is fine-tuning and Why fine-tunning](#2)
    3. [What does fine-tuning do for the model](#3)
    4. [Prompt engineering vs. fine-tuning](#4)
    5. [Benefits of fine-tuning our own LLM](#5)
    6. [Technology stack used in this course](#6)
    7. [Installation](#7)
2. [fine-tuned vs. non-fine-tuned models](#8)
3. [Where fine-tuning fits in](#9)
4. [Instruction fine-tuning](#10)
5. [Data preparation](#11)


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
### Technology stack used in this course

+ Pytorch (Meta)
+ Huggingface
+ Llama library(Lamini) (the highest level interface)

https://lamini-ai.github.io/#training

<a name="7"></a>
### Installation -- https://lamini-ai.github.io/get_started/install/#1-get-your-lamini-api-key

    conda create --name llm python=3.10.12 (python 3.12 was not compatible)
    pip install --upgrade lamini

<a name="8"></a>
## fine-tuned vs. non-fine-tuned models

    from llama import BasicModelRunner
    non_finetuned  = BasicModelRunner("meta-llama/Llama-2-7b-hf")
    non_finetuned_output = non_finetuned("Tell me how to train my dog to sit")
    print(non_finetuned_output)

returns me back 

    Tell me how to train my dog to sit. I have a 10 month old puppy and I want to train him to sit. I have tried the treat method and he just sits there and looks at me like I am crazy. I have tried the "sit" command and he just looks at me like I am crazy. I have tried the "sit" command and he just looks at me like I am crazy. I have tried the "sit" command and he just looks at me like I am crazy. I have tried the "sit" command and he just looks at me like I am crazy. I have tried the "sit" command and he just looks at me like I am crazy. I have tried the "sit" command and he just looks at me like I am crazy. I have tried the "sit" command and he just looks at me like I am crazy. I have tried the "sit" command and he just looks at me like I am crazy. I have tried the "sit" command and he just looks at me like I am crazy. I have tried the "sit" command and he just looks

But when the model is fine-tuned:

    finetuned_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")
    finetuned_output = finetuned_model("Tell me how to train my dog to sit")
    print(finetuned_output)

returns me back:

    on command.
    Training a dog to sit on command is a basic obedience command that can be achieved with patience, consistency, and positive reinforcement. Here's a step-by-step guide on how to train your dog to sit on command:
    
    1. Choose a quiet and distraction-free area: Find a quiet area with minimal distractions where your dog can focus on you.
    2. Have treats ready: Choose your dog's favorite treats and have them ready to use as rewards.
    3. Stand in front of your dog: Stand in front of your dog and hold a treat close to their nose.
    4. Move the treat up and back: Slowly move the treat up and back, towards your dog's tail, while saying "sit" in a calm and clear voice.
    5. Dog will sit: As you move the treat, your dog will naturally sit down to follow the treat. The moment their bottom touches the ground, say "good sit" and give them the treat.
    6. Repeat the process: Repeat steps 3-5 several times, so your dog starts to associate the command "sit" with

To get rid of the auto-completion on top (on command), we can inform the model that we want instruction, so I can put the instruction tags like:

    print(finetuned_model("[INST]Tell me how to train my dog to sit[/INST]"))

returns me back

    Training your dog to sit is a basic obedience command that can be achieved with patience, consistency, and positive reinforcement. Here's a step-by-step guide on how to train your dog to sit:
    
    1. Choose a quiet and distraction-free area: Find a quiet area with no distractions where your dog can focus on you.
    2. Have treats ready: Choose your dog's favorite treats and have them ready to use as rewards.
    3. Stand in front of your dog: Stand in front of your dog and hold a treat close to their nose.
    4. Move the treat up and back: Slowly move the treat up and back, towards your dog's tail, while saying "sit" in a calm and clear voice.
    5. Dog will sit: As you move the treat, your dog will naturally sit down to follow the treat. The moment their bottom touches the ground, say "good sit" and give them the treat.
    6. Repeat the process: Repeat steps 3-5 several times, so your dog starts to associate the command "sit" with the action of sitting down.

    chatgpt = BasicModelRunner("chat-gpt")
    print(chatgpt("Tell me how to train my dog to sit"))

returns me:

    UserError: Currently this user has support for base models: ['hf-internal-testing/tiny-random-gpt2', 'EleutherAI/pythia-70m', 'EleutherAI/pythia-70m-deduped', 'EleutherAI/pythia-70m-v0', 'EleutherAI/pythia-70m-deduped-v0', 'EleutherAI/neox-ckpt-pythia-70m-deduped-v0', 'EleutherAI/neox-ckpt-pythia-70m-v1', 'EleutherAI/neox-ckpt-pythia-70m-deduped-v1', 'EleutherAI/gpt-neo-125m', 'EleutherAI/pythia-160m', 'EleutherAI/pythia-160m-deduped', 'EleutherAI/pythia-160m-deduped-v0', 'EleutherAI/neox-ckpt-pythia-70m', 'EleutherAI/neox-ckpt-pythia-160m', 'EleutherAI/neox-ckpt-pythia-160m-deduped-v1', 'EleutherAI/pythia-2.8b', 'EleutherAI/pythia-410m', 'EleutherAI/pythia-410m-v0', 'EleutherAI/pythia-410m-deduped', 'EleutherAI/pythia-410m-deduped-v0', 'EleutherAI/neox-ckpt-pythia-410m', 'EleutherAI/neox-ckpt-pythia-410m-deduped-v1', 'cerebras/Cerebras-GPT-111M', 'cerebras/Cerebras-GPT-256M', 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf', 'Intel/neural-chat-7b-v3-1']


<a name="9"></a>
## Where fine-tuning fits in

![](https://github.com/DanialArab/images/blob/main/LLM/pretraining.png)


![](https://github.com/DanialArab/images/blob/main/LLM/data%20scraping.png)

 Pre-trained base models have been trained on the data sets from the web and it might have this geography homework that asks What's the capital of Kenya? What's the capital of France? And it all, in a line without seeing the answers. So when you then input, what's the capital of Mexico, the LLM might just say, what's the capital of Hungary? As you can see that it's not really useful from the sense of a chatbot interface. **So how do you get it to that chatbot interface?** fine-tuning is one way. 
 
![](https://github.com/DanialArab/images/blob/main/LLM/limitations%20of%20pretrained%20models.png)

![](https://github.com/DanialArab/images/blob/main/LLM/fine%20tuning%20after%20pretraining.png)

![](https://github.com/DanialArab/images/blob/main/LLM/what%20fine-tuning%20can%20do.png)

**Tasks for fine-tuning are really just text in, text out for LLMs**. And we can think about it in two different categories:
+ Extraction (lot of the work is in reading)
+ Expansion (lot of the work is in writing)

![](https://github.com/DanialArab/images/blob/main/LLM/tasks%20to%20finetune.png)

![](https://github.com/DanialArab/images/blob/main/LLM/first%20time%20fine-tuning.png)


<a name="10"></a>
## Instruction fine-tuning

Instruction fine-tuning is a variant of fine-tuning that enabled GPT-3 to turn into chat GPT and give it its chatting powers. 

![](https://github.com/DanialArab/images/blob/main/LLM/instruction%20fine-tuning.png)

![](https://github.com/DanialArab/images/blob/main/LLM/instruction-following%20datasets.png)

If you don't have data, no problem. You can also convert your data into something that's more of a question-answer format or instruction-following format by using a prompt template. So a README might be able to be converted into a question-answer pair. You can also use another **LLM to do this for you. There's a technique called Alpaca from Stanford that uses chat GPT to do this. And of course, you can use a pipeline of different open-source models to do this as well.**

![](https://github.com/DanialArab/images/blob/main/LLM/instruction%20fine-tuning%20generaliztion.png)

So one of the coolest things about fine-tuning is that **it teaches this new behavior to the model.** And while you might have fine-tuning data on what's the capital of France, Paris, because these are easy question-answer pairs that you can get. **You can also generalize this idea of question answering to data you might not have given the model for your fine-tuning dataset, but that the model had already learned in its pre-existing pre-training step.** And so that might be code. And this is actually findings from the chat GPT paper where the model can now answer questions about code even though they didn't have question-answer pairs about that for their instruction fine-tuning. 

![](https://github.com/DanialArab/images/blob/main/LLM/instruction%20fine-tuning%20generaliztion%202.png)

An overview of the different steps of fine-tuning are data prep, training, and evaluation. Of course, after you evaluate the model, you need to prep the data again to improve it. It's a very iterative process to improve the model. **And specifically for instruction fine-tuning and other different types of fine-tuning, data prep is really where you have differences. This is really where you change your data, you tailor your data to the specific type of fine-tuning that you're doing. Training and evaluation are very similar.**

![](https://github.com/DanialArab/images/blob/main/LLM/interative%20process%20of%20fine-tuning.png)

<a name="11"></a>
## Data preparation
