# Pair programming with a Large Language Model

Reference: <a href="https://www.deeplearning.ai/short-courses/pair-programming-llm/">Pair Programing with a Large Language Model</a>

Instructor: Laurence Moroney - Google. 

1. [Prerequisites](#1)
2. [Introduction](#2)
3. [Pattern](#3)
4. [Examples](#4)
5. [String template](#5)
6. [Pair programming scenarios](#6)
   1. [Improve existing code](#7)
   2. [Simplify code](#8)
   3. [Write test cases](#9)
   4. [Make code more efficient](#10)
   5. [Debug your code](#11)

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

As discussed, the models are based on animal sizes so gecko is smaller than bison models.

Some notes on the models:
+ **generateText** is currently recommended for coding-related prompts. It is more optimized for a single shot, this one generally works better for code. 
+ **generateMessage** is optimized for multi-turn chats (dialogues) with an LLM. It keeps track of contexts.

        models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
        model_bison = models[0]
        model_bison

returns me back

        Model(name='models/text-bison-001', base_model_id='', version='001', display_name='Text Bison', description='Model targeted for text generation.', input_token_limit=8196, output_token_limit=1024, supported_generation_methods=['generateText', 'countTextTokens', 'createTunedTextModel'], temperature=0.7, top_p=0.95, top_k=40) 

**text-bison-001** is the model that we use. 

Now that we have our model, we create another helper function to generate text:

        from google.api_core import retry
        @retry.Retry()
        def generate_text(prompt,
                          model=model_bison,
                          temperature=0.0):
            return palm.generate_text(prompt=prompt,
                                      model=model,
                                      temperature=temperature)

the default temperature for the model is 0.7 but here we set it to 0.0 to make it a more **deterministic model** so we can get more consistent results based on our prompts.

<a name="3"></a>
## Pattern 

For all the examples we use the following pattern to generate code:

Create a prompt:

![](https://github.com/DanialArab/images/blob/main/LLM/create%20a%20prompt.PNG)

Completion:

![](https://github.com/DanialArab/images/blob/main/LLM/completion.PNG)

Output:

![](https://github.com/DanialArab/images/blob/main/LLM/output.PNG)

We can print output out in the Jupyter Notebook or in a real-world application we may be injecting the code into an IDE or saving it into a repo etc. 

<a name="4"></a>
## Example

        prompt = "Show me how to iterate across a list in Python."
        completion = generate_text(prompt)
        print(completion.result)

returns me back:

        To iterate across a list in Python, you can use the `for` loop. The syntax is as follows:
        
        ```python
        for item in list:
          # do something with item
        ```
        
        For example, the following code prints each item in the list `my_list`:
        
        ```python
        my_list = ["a", "b", "c"]
        
        for item in my_list:
          print(item)
        ```
        
        Output:
        
        ```
        a
        b
        c
        ```
        
        You can also use the `enumerate()` function to iterate over a list and get the index of each item. The syntax is as follows:

        ```python
        for index, item in enumerate(list):
          # do something with index and item
        ```
        
        For example, the following code prints the index and value of each item in the list `my_list`:
        
        ```python
        my_list = ["a", "b", "c"]
        
        for index, item in enumerate(my_list):
          print(index, item)
        ```
        
        Output:
        
        ```
        0 a
        1 b
        2 c
        ```
**Tip**: The words "show me" tends to encourage the PaLM LLM to give more details and explanations compared to if you were to ask "write code to ..."

        prompt = "write code to iterate across a list in Python"
        completion = generate_text(prompt)
        print(completion.result)

which gives me back only the code:

        ```python
        # Iterate across a list in Python
        
        # Create a list
        list = ["apple", "banana", "cherry"]
        
        # Iterate over the list using a for loop
        for item in list:
            print(item)
        
        # Output:
        # apple
        # banana
        # cherry
        ```
Some tips:
+ Try copy-pasting some of the generated code and running it in the notebook.
+ Remember to test out the LLM-generated code and debug it to make sure it works as intended

We can make the prompt a little bit more efficient using a String template. 

<a name="5"></a>
## String template

One useful method to interact with LLM is to prime it using a following prompt template: 

+ priming: getting the LLM ready for the type of task you'll ask it to do.
+ question: the specific task.
+ decorator: how to provide or format the output.

        prompt_template = """
        {priming}
        
        {question}
        
        {decorator}
        
        Your solution:
        """
like

        priming_text = "You are an expert at writing clear, concise, Python code."
        question = "create a doubly linked list"
 
**Observe how the decorator affects the output**:

+ In other non-coding prompt engineering tasks, it's common to use "chain-of-thought prompting" by asking the model to work through the task "step by step" (option 1 below).
+ For certain tasks like generating code, you may want to experiment with other wording that would make sense if you were asking a developer the same question (option 2 below).

        # option 1
        decorator = "Work through it step by step, and show your work. One step per line."

        prompt = prompt_template.format(priming=priming_text,
                                    question=question,
                                    decorator=decorator)
        print(prompt)


        You are an expert at writing clear, concise, Python code.
        
        create a doubly linked list
        
        Work through it step by step, and show your work. One step per line.
        
        Your solution:

Call the API to get the completion:

        completion = generate_text(prompt)
        print(completion.result)

returns me back:

        ```python
        class Node:
            def __init__(self, data):
                self.data = data
                self.next = None
                self.prev = None
        
        
        class DoublyLinkedList:
            def __init__(self):
                self.head = None
                self.tail = None
        
            def append(self, data):
                new_node = Node(data)
                if self.head is None:
                    self.head = new_node
                    self.tail = new_node
                else:
                    self.tail.next = new_node
                    new_node.prev = self.tail
                    self.tail = new_node
        
            def prepend(self, data):
                new_node = Node(data)
                if self.head is None:
                    self.head = new_node
                    self.tail = new_node
                else:
                    new_node.next = self.head
                    self.head.prev = new_node
                    self.head = new_node
        
            def print_list(self):
                curr_node = self.head
                while curr_node is not None:
                    print(curr_node.data)
                    curr_node = curr_node.next
        
        
        if __name__ == "__main__":
            dll = DoublyLinkedList()
            dll.append(1)
            dll.append(2)
            dll.append(3)
            dll.prepend(0)
            dll.print_list()
        ```

option 2:

        decorator = "Insert comments for each line of code."
        prompt = prompt_template.format(priming=priming_text,
                                        question=question,
                                        decorator=decorator)
        print(prompt)

        You are an expert at writing clear, concise, Python code.
        
        create a doubly linked list
        
        Insert comments for each line of code.
        
        Your solution:
        
        completion = generate_text(prompt)
        print(completion.result)
        
        ```python
        class Node:

            """Node in a doubly linked list."""
        
            def __init__(self, data):
                """Initialize a node with the given data."""
                self.data = data
                self.next = None
                self.prev = None
        
        
        class DoublyLinkedList:
        
            """Doubly linked list."""
        
            def __init__(self):
                """Initialize an empty doubly linked list."""
                self.head = None
                self.tail = None
                self.size = 0
        
            def __len__(self):
                """Return the number of nodes in the list."""
                return self.size
        
            def is_empty(self):
                """Return True if the list is empty."""
                return self.size == 0
        
            def add_first(self, data):
                """Add a new node with the given data to the front of the list."""
                new_node = Node(data)
                if self.is_empty():
                    self.head = new_node
                    self.tail = new_node
                else:
                    new_node.next = self.head
                    self.head.prev = new_node
                    self.head = new_node
                self.size += 1
        
            def add_last(self, data):
                """Add a new node with the given data to the end of the list."""
                new_node = Node(data)
                if self.is_empty():
                    self.head = new_node
                    self.tail = new_node
                else:
                    new_node.prev = self.tail
                    self.tail.next = new_node
                    self.tail = new_node
                self.size += 1
        
            def remove_first(self):
                """Remove the first node in the list."""
                if self.is_empty():
                    raise ValueError("List is empty")
                self.head = self.head.next
                if self.head is None:
                    self.tail = None
                else:
                    self.head.prev = None
                self.size -= 1
        
            def remove_last(self):
                """Remove the last node in the list."""
                if self.is_empty():
                    raise ValueError("List is empty")
                self.tail = self.tail.prev
                if self.tail is None:
                    self.head = None
                else:
                    self.tail.next = None
                self.size -= 1
        
            def __iter__(self):
                """Iterate over the nodes in the list in order."""
                node = self.head
                while node is not None:
                    yield node.data
                    node = node.next
        
            def __str__(self):
                """Return a string representation of the list."""
                return "[" + ", ".join(str(node.data) for node in self) + "]"
        
        ```

another example:

        priming_text = "You are an expert at writing clear, concise, Python code."
        question = """create a very large list of random numbers in python,  and then write code to sort that list"""
        decorator = "Insert comments for each line of code."
        prompt = prompt_template.format(priming=priming_text,
                                        question=question,
                                        decorator=decorator)
        print(prompt)

        You are an expert at writing clear, concise, Python code.

        create a very large list of random numbers in python, 
        and then write code to sort that list
        
        Insert comments for each line of code.
        
        Your solution:

        ```python
        # Create a list of 100,000 random numbers between 0 and 100
        random_numbers = [random.randint(0, 100) for _ in range(100000)]
        
        # Sort the list of random numbers
        sorted_random_numbers = sorted(random_numbers)
        
        # Print the first 10 elements of the sorted list
        print(sorted_random_numbers[:10])
        
        ```
        
        **Explanation:**
        
        * The first line of code creates a list of 100,000 random numbers between 0 and 100. The `random.randint()` function is used to generate a random integer between two specified values.
        * The second line of code sorts the list of random numbers using the `sorted()` function. The `sorted()` function takes a list as its argument and returns a new list that is sorted in ascending order.
        * The third line of code prints the first 10 elements of the sorted list. The `print()` function is used to print the contents of a list to the console.
        
        This code is clear, concise, and Pythonic. It uses the standard Python library functions to create a list of random numbers, sort the list, and print the results.


<a name="6"></a>
## Pair programming scenarios

<a name="7"></a>
### Improve existing code


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
