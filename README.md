# Pair programming with a Large Language Model

Reference: <a href="https://www.deeplearning.ai/short-courses/pair-programming-llm/">Pair Programing with a Large Language Model</a>

Instructor: Laurence Moroney - Google. 

1. [Prerequisites](#1)
2. [Introduction](#2) 

<a name="1"></a>
## Prerequisites 

In this course, we use PaLM API, which gives us access to many of the features of **Google's large language models** via a coding interface. 

![](https://github.com/DanialArab/images/blob/main/LLM/google%20generative%20AI.PNG)

What I need is:

+ an API key (unfortunately PaLM API is not available in Canada - Sep. 29, 2023)
+ Generative AL Libraries (!pip install -q google.generativeai)
+ Python coding skills 

    
Although I could not get my own PaLM API for my location (from https://developers.generativeai.google/), the API was provided in the course to practice. To get the provided API key we need a utility to get that through a helper function:

        from utils import get_api_key

<a name="2"></a>
## Introduction 

There are a lot of animals and in general the larger the animal the larger the model.

        import os
        import google.generativeai as palm
        from google.api_core import client_options as client_options_lib
        
        palm.configure(
            api_key=get_api_key(),
            transport="rest",
            client_options=client_options_lib.ClientOptions(
                api_endpoint=os.getenv("GOOGLE_API_BASE"),
            )
        )

Let's see what models are available:

        for m in palm.list_models():
            print(f"name: {m.name}")
            print(f"description: {m.description}")
            print(f"generation methods:{m.supported_generation_methods}\n")

returns back:

        name: models/chat-bison-001
        description: Chat-optimized generative language model.
        generation methods:['generateMessage', 'countMessageTokens']
        
        name: models/text-bison-001
        description: Model targeted for text generation.
        generation methods:['generateText', 'countTextTokens', 'createTunedTextModel']
        
        name: models/embedding-gecko-001
        description: Obtain a distributed representation of a text.
        generation methods:['embedText']

Some notes on the models:
+ **generateText** is currently recommended for coding-related prompts.
+ **generateMessage** is optimized for multi-turn chats (dialogues) with an LLM.

        models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
        model_bison = models[0]
        model_bison

returns me back

        Model(name='models/text-bison-001', base_model_id='', version='001', display_name='Text Bison', description='Model targeted for text generation.', input_token_limit=8196, output_token_limit=1024, supported_generation_methods=['generateText', 'countTextTokens', 'createTunedTextModel'], temperature=0.7, top_p=0.95, top_k=40) 

<a name="1"></a>
## Pair programming scenarios
+ Improve existing code
+ Simplify code
+ Write test cases
+ Make code more efficient
+ Debug your code

<a name="1"></a>
### Improve existing code

<a name="1"></a>
### Simplify code

<a name="1"></a>
### Write test cases

<a name="1"></a>
### Make code more efficient

<a name="1"></a>
### Debug your code
