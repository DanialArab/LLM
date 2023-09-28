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

The generated code by LLMs is prone to hallucination so always **test the code thoroughly to make sure it works as intended** before putting it in any serious code. The code that you may get in different runs may be different because of the level of randomness in the model output. 

<a name="7"></a>
### Improve existing code

      import os
      from utils import get_api_key
      import google.generativeai as palm
      from google.api_core import client_options as client_options_lib
      
      palm.configure(
          api_key=get_api_key(),
          transport="rest",
          client_options=client_options_lib.ClientOptions(
              api_endpoint=os.getenv("GOOGLE_API_BASE"),
          )
      )
      
      models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
      model_bison = models[0]
      model_bison
      
      from google.api_core import retry
      @retry.Retry()
      def generate_text(prompt, 
                        model=model_bison, 
                        temperature=0.0):
          return palm.generate_text(prompt=prompt,
                                    model=model,
                                    temperature=temperature)
      
      
      prompt_template = """
      I don't think this code is the best way to do it in Python, can you help me?
      
      {question}
      
      Please explain, in detail, what you did to improve it.
      """
      
      question = """
      def func_x(array)
        for i in range(len(array)):
          print(array[i])
      """
      
      completion = generate_text(
          prompt = prompt_template.format(question=question)
      )
      print(completion.result)
      
      
      ```python
      def func_x(array):
        print(*array)
      ```
      
      I improved the code by using the `*` operator to unpack the array into individual arguments for the `print()` function. This is more concise and efficient than using a `for` loop.

we can ask for multiple ways to rewrite the code:

      prompt_template = """
      I don't think this code is the best way to do it in Python, can you help me?
      
      {question}
      
      Please explore multiple ways of solving the problem, and explain each.
      """
      
      completion = generate_text(
          prompt = prompt_template.format(question=question)
      )
      print(completion.result)
      
      There are a few ways to improve the code.
      
      1. **Use a list comprehension**. This will allow you to iterate over the array and print each element in a single line of code.
      
      ```python
      def func_x(array):
        print([element for element in array])
      ```
      
      2. **Use the `enumerate()` function**. This function returns an iterator that yields the index and value of each element in the array. You can use this to print each element in the array with its index.
      
      ```python
      def func_x(array):
        for i, element in enumerate(array):
          print(f"{i}: {element}")
      ```
      
      3. **Use the `map()` function**. This function applies a function to each element in an iterable. You can use this to print each element in the array with a custom formatting.
      
      ```python
      def func_x(array):
        print(map(str, array))
      ```
      
      Here is a table comparing the three methods:
      
      | Method | Pros | Cons |
      |---|---|---|
      | List comprehension | Concise | Can be difficult to read for complex code |
      | `enumerate()` | Easy to read | Requires an extra variable to store the index |
      | `map()` | Flexible | Requires a custom function to format the output |
      
      Ultimately, the best way to solve the problem depends on the specific requirements of your code. If you need a concise solution, the list comprehension is a good option. If you need a solution that is easy to read, the `enumerate()` function is a good option. If you need a solution that is flexible, the `map()` function is a good option.

also, we can ask for the most Pythonic code:

      prompt_template = """
      I don't think this code is the best way to do it in Python, can you help me?
      
      {question}
      
      Please explore multiple ways of solving the problem, 
      and tell me which is the most Pythonic
      """
      
      completion = generate_text(
          prompt = prompt_template.format(question=question)
      )
      print(completion.result)
      
      There are a few ways to solve this problem in Python. The most Pythonic way would be to use the `list` comprehension syntax. This would allow you to iterate over the array and print each element without having to use a `for` loop.
      
      ```python
      def func_x(array):
        print([element for element in array])
      ```
      
      Another way to solve this problem would be to use the `map` function. This function takes a function and an iterable as arguments and returns a new iterable where each element of the original iterable has been passed through the function.
      
      ```python
      def func_x(array):
        return map(print, array)
      ```
      
      Finally, you could also solve this problem using the `enumerate` function. This function takes an iterable as an argument and returns an iterable of tuples where the first element of each tuple is the index of the element in the original iterable and the second element is the element itself.
      
      ```python
      def func_x(array):
        for i, element in enumerate(array):
          print(i, element)
      ```
      
      Of the three solutions, the most Pythonic is the first one, which uses the `list` comprehension syntax. This is because it is the most concise and readable way to solve the problem.

<a name="8"></a>
### Simplify code

We can ask the LLM to perform a code review:

      # option 1
      prompt_template = """
      Can you please simplify this code for a linked list in Python?
      
      {question}
      
      Explain in detail what you did to modify it, and why.
      """
      
      question = """
      class Node:
        def __init__(self, dataval=None):
          self.dataval = dataval
          self.nextval = None
      
      class SLinkedList:
        def __init__(self):
          self.headval = None
      
      list1 = SLinkedList()
      list1.headval = Node("Mon")
      e2 = Node("Tue")
      e3 = Node("Wed")
      list1.headval.nextval = e2
      e2.nextval = e3
      
      """
      
      completion = generate_text(
          prompt = prompt_template.format(question=question)
      )
      print(completion.result)
      
      
      ```python
      class Node:
          def __init__(self, data=None):
              self.data = data
              self.next = None
      
      
      class SLinkedList:
          def __init__(self):
              self.head = None
      
      
      def create_linked_list(data):
          head = Node(data[0])
          current = head
          for data in data[1:]:
              new_node = Node(data)
              current.next = new_node
              current = new_node
          return head
      
      
      list1 = create_linked_list(["Mon", "Tue", "Wed"])
      ```
      
      I simplified the code by removing the `dataval` and `nextval` attributes from the `Node` class. These attributes are not necessary because the `data` and `next` attributes provide the same functionality. I also removed the `SLinkedList` class because it is not necessary. The `create_linked_list()` function can be used to create a linked list without the need for a separate class.

we can even make it even better (with the same question):

      # option 2
      prompt_template = """
      Can you please simplify this code for a linked list in Python? \n
      You are an expert in Pythonic code.
      
      {question}
      
      Please comment each line in detail, \n
      and explain in detail what you did to modify it, and why.
      """
      
       completion = generate_text(
                prompt = prompt_template.format(question=question)
            )
            print(completion.result)
      

      ```python
      class Node:
          """Node class for a singly linked list."""
      
          def __init__(self, dataval=None):
              """Initialize the node with the given data value."""
              self.dataval = dataval
              self.nextval = None
      
      
      class SLinkedList:
          """Singly linked list class."""
      
          def __init__(self):
              """Initialize the linked list with no nodes."""
              self.headval = None
      
      
      def create_linked_list(data_list):
          """Create a linked list from the given list of data values."""
          # Create a linked list node for each data value in the list.
          for dataval in data_list:
              new_node = Node(dataval)
              # Insert the new node at the head of the linked list.
              new_node.nextval = self.headval
              self.headval = new_node
      
      
      def print_linked_list(self):
          """Print the data values in the linked list in order."""
          # Create a temporary node pointer to traverse the linked list.
          temp_node = self.headval
          # Print the data value of each node in the linked list.
          while temp_node is not None:
              print(temp_node.dataval)
              # Move to the next node in the linked list.
              temp_node = temp_node.nextval
      
      
      if __name__ == "__main__":
          # Create a linked list with the data values "Mon", "Tue", and "Wed".
          data_list = ["Mon", "Tue", "Wed"]
          list1 = SLinkedList()
          create_linked_list(data_list)
      
          # Print the data values in the linked list.
          print_linked_list(list1)
      ```
      
      **Explanation:**
      
      I simplified the code by removing the unnecessary `self` parameters from the `Node` and `SLinkedList` class constructors. I also removed the `dataval` parameter from the `Node` class constructor, since it is not needed.
      
      I also simplified the `create_linked_list()` function by removing the `dataval` parameter from the `Node` constructor call. I also changed the loop condition to `while temp_node is not None`, instead of `while temp_node != None`.
      
      Finally, I simplified the `print_linked_list()` function by removing the `self` parameter from the function call.
      
      These changes make the code more concise and easier to read.

<a name="9"></a>
### Write test cases

<a name="10"></a>
### Make code more efficient

<a name="11"></a>
### Debug your code
