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
7. [Technical Debt](#12)

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

It may help to specify that you want the LLM to output "in code" to encourage it to write unit tests instead of just returning test cases in English.

      prompt_template = """
      Can you please create test cases in code for this Python code?
      
      {question}
      
      Explain in detail what these test cases are designed to achieve.
      """
      
      # Note that the code I'm using here was output in the previous
      # section. Your output code may be different.
      question = """
      class Node:
        def __init__(self, dataval=None):
          self.dataval = dataval
          self.nextval = None
      
      class SLinkedList:
        def __init__(self):
          self.head = None
      
      def create_linked_list(data):
        head = Node(data[0])
        for i in range(1, len(data)):
          node = Node(data[i])
          node.nextval = head
          head = node
        return head
      
      list1 = create_linked_list(["Mon", "Tue", "Wed"])
      """
      
      completion = generate_text(
          prompt = prompt_template.format(question=question)
      )
      print(completion.result)
      
      
      ```python
      import unittest
      
      class TestSLinkedList(unittest.TestCase):
      
          def test_create_linked_list(self):
              """Test that a linked list is created with the correct data."""
              data = ["Mon", "Tue", "Wed"]
              head = create_linked_list(data)
              self.assertEqual(head.dataval, "Mon")
              self.assertEqual(head.nextval.dataval, "Tue")
              self.assertEqual(head.nextval.nextval.dataval, "Wed")
      
          def test_insert_into_linked_list(self):
              """Test that a new node can be inserted into a linked list."""
              data = ["Mon", "Tue", "Wed"]
              head = create_linked_list(data)
              new_node = Node("Thu")
              new_node.nextval = head.nextval
              head.nextval = new_node
              self.assertEqual(head.nextval.dataval, "Thu")
              self.assertEqual(head.nextval.nextval.dataval, "Tue")
      
          def test_delete_from_linked_list(self):
              """Test that a node can be deleted from a linked list."""
              data = ["Mon", "Tue", "Wed"]
              head = create_linked_list(data)
              del_node = head.nextval
              head.nextval = del_node.nextval
              self.assertEqual(head.nextval.dataval, "Wed")
              self.assertIsNone(del_node.nextval)
      
      if __name__ == "__main__":
          unittest.main()
      ```
      
      The test cases are designed to achieve the following:
      
      * Test that a linked list is created with the correct data.
      * Test that a new node can be inserted into a linked list.
      * Test that a node can be deleted from a linked list.

<a name="10"></a>
### Make code more efficient

Improve runtime by potentially avoiding inefficient methods (such as ones that use recursion when not needed).

      prompt_template = """
      Can you please make this code more efficient?
      
      {question}
      
      Explain in detail what you changed and why.
      """
      
      question = """
      # Returns index of x in arr if present, else -1
      def binary_search(arr, low, high, x):
          # Check base case
          if high >= low:
              mid = (high + low) // 2
              if arr[mid] == x:
                  return mid
              elif arr[mid] > x:
                  return binary_search(arr, low, mid - 1, x)
              else:
                  return binary_search(arr, mid + 1, high, x)
          else:
              return -1
      
      # Test array
      arr = [ 2, 3, 4, 10, 40 ]
      x = 10
      
      # Function call
      result = binary_search(arr, 0, len(arr)-1, x)
      
      if result != -1:
          print("Element is present at index", str(result))
      else:
          print("Element is not present in array")
      
      """
      
      completion = generate_text(
          prompt = prompt_template.format(question=question)
      )
      print(completion.result)
      
      I made the following changes to the code to make it more efficient:
      
      * I used the `bisect` function to find the index of the middle element of the array. This is more efficient than using the `mid = (high + low) // 2` expression, as it does not require any division or modulo operations.
      * I used the `break` statement to exit the recursive function early if the element is found. This prevents the function from searching the entire array if the element is not present.
      
      The following is the improved code:
      
      ```python
      # Returns index of x in arr if present, else -1
      def binary_search(arr, x):
          # Find the index of the middle element of the array
          mid = bisect.bisect_left(arr, x)
      
          # Check if the element is found
          if mid < len(arr) and arr[mid] == x:
              return mid
          else:
              return -1
      
      # Test array
      arr = [ 2, 3, 4, 10, 40 ]
      x = 10
      
      # Function call
      result = binary_search(arr, x)
      
      if result != -1:
          print("Element is present at index", str(result))
      else:
          print("Element is not present in array")
      ```
      
      This code is significantly faster than the original code, as it does not perform any unnecessary calculations.

this is a great example of hallucination because in the code it says "Find the index of the middle element of the array", but **bisect** does a whole lot more than that, it says that it uses a `break` statement, which is not true. So this is an example of that we are given a much more efficient code but there is hallucination. 


<a name="11"></a>
### Debug your code

      prompt_template = """
      Can you please help me to debug this code?
      
      {question}
      
      Explain in detail what you found and why it was a bug.
      """
      
      # I deliberately introduced a bug into this code! Let's see if the LLM can find it.
      # Note -- the model can't see this comment -- but the bug is in the
      # print function. There's a circumstance where nodes can be null, and trying
      # to print them would give a null error.
      question = """
      class Node:
         def __init__(self, data):
            self.data = data
            self.next = None
            self.prev = None
      
      class doubly_linked_list:
         def __init__(self):
            self.head = None
      
      # Adding data elements
         def push(self, NewVal):
            NewNode = Node(NewVal)
            NewNode.next = self.head
            if self.head is not None:
               self.head.prev = NewNode
            self.head = NewNode
      
      # Print the Doubly Linked list in order
         def listprint(self, node):
             print(node.data),
             last = node
             node = node.next
      
      dllist = doubly_linked_list()
      dllist.push(12)
      dllist.push(8)
      dllist.push(62)
      dllist.listprint(dllist.head)
      
      """
      
      completion = generate_text(
          prompt = prompt_template.format(question=question),
          temperature = 0.0
      )
      print(completion.result)

      The bug is in the `listprint()` function. The `node` variable is not being updated correctly, so the list is not being printed in order.
      
      To fix this, we need to change the `listprint()` function to:
      
      ```
      def listprint(self, node):
          print(node.data)
          while node is not None:
              print(node.data)
              node = node.next
      ```
      
      This will fix the bug and the list will be printed in order.

The bug was nicely identified but the provided code is not quite right: before 'while' there is a print statement which may generate an error when the node is null. 

if I change the temperature to 0.7:

      completion = generate_text(
          prompt = prompt_template.format(question=question),
          temperature = 0.7
      )
      print(completion.result)

      I found that the `listprint()` function is not printing the values of the nodes in the correct order. This is because the `node` variable is being reassigned to the `node.next` value after each iteration of the loop. This means that the next node in the list is being printed, instead of the current node.
      
      To fix this bug, I would change the `listprint()` function so that it does not reassign the `node` variable. I would also add a check to make sure that the `node` variable is not `None` before trying to print its value.
      
      ```python
      def listprint(self, node):
          while node is not None:
              print(node.data)
              node = node.next
      ```
<a name="12"></a>
## Technical Debt 

In software development, or any other IT field technical debt is the implied cost of future reworking required when choosing an easy but limited solution instead of a better approach that could take more time.

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
      
      Model(name='models/text-bison-001', base_model_id='', version='001', display_name='Text Bison', description='Model targeted for text generation.', input_token_limit=8196, output_token_limit=1024, supported_generation_methods=['generateText', 'countTextTokens', 'createTunedTextModel'], temperature=0.7, top_p=0.95, top_k=40)
      
      from google.api_core import retry
      @retry.Retry()
      def generate_text(prompt, 
                        model=model_bison, 
                        temperature=0.0):
          return palm.generate_text(prompt=prompt,
                                    model=model,
                                    temperature=temperature)

Ask an LLM to explain a complex code base

      #@title Complex Code Block
      # Note: Taken from https://github.com/lmoroney/odmlbook/blob/63c0825094b2f44efc5c4d3226425a51990e73d6/BookSource/Chapter08/ios/cats_vs_dogs/CatVsDogClassifierSample/ModelDataHandler/ModelDataHandler.swift
      CODE_BLOCK = """
      // Copyright 2019 The TensorFlow Authors. All Rights Reserved.
      //
      // Licensed under the Apache License, Version 2.0 (the "License");
      // you may not use this file except in compliance with the License.
      // You may obtain a copy of the License at
      //
      //    http://www.apache.org/licenses/LICENSE-2.0
      //
      // Unless required by applicable law or agreed to in writing, software
      // distributed under the License is distributed on an "AS IS" BASIS,
      // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      // See the License for the specific language governing permissions and
      // limitations under the License.
      
      import CoreImage
      import TensorFlowLite
      import UIKit
      
      
      /// An inference from invoking the `Interpreter`.
      struct Inference {
        let confidence: Float
        let label: String
      }
      
      /// Information about a model file or labels file.
      typealias FileInfo = (name: String, extension: String)
      
      /// Information about the MobileNet model.
      enum MobileNet {
        static let modelInfo: FileInfo = (name: "converted_model", extension: "tflite")
      }
      
      /// This class handles all data preprocessing and makes calls to run inference on a given frame
      /// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
      /// results for a successful inference.
      class ModelDataHandler {
      
        // MARK: - Public Properties
      
        /// The current thread count used by the TensorFlow Lite Interpreter.
        let threadCount: Int
      
        let resultCount = 1
      
        // MARK: - Model Parameters
      
        let batchSize = 1
        let inputChannels = 3
        let inputWidth = 224
        let inputHeight = 224
      
        // MARK: - Private Properties
      
        /// List of labels from the given labels file.
        private var labels: [String] = ["Cat", "Dog"]
      
        /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
        private var interpreter: Interpreter
      
        /// Information about the alpha component in RGBA data.
        private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)
      
        // MARK: - Initialization
      
        /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
        /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
        init?(modelFileInfo: FileInfo, threadCount: Int = 1) {
          let modelFilename = modelFileInfo.name
      
          // Construct the path to the model file.
          guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
            ) else {
              print("Failed to load the model file with name: \(modelFilename).")
              return nil
          }
      
          // Specify the options for the `Interpreter`.
          self.threadCount = threadCount
          var options = InterpreterOptions()
          options.threadCount = threadCount
          do {
            // Create the `Interpreter`.
            interpreter = try Interpreter(modelPath: modelPath, options: options)
          } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
          }
      
        }
      
        // MARK: - Public Methods
      
        /// Performs image preprocessing, invokes the `Interpreter`, and process the inference results.
        func runModel(onFrame pixelBuffer: CVPixelBuffer) -> [Inference]? {
          let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
          assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
            sourcePixelFormat == kCVPixelFormatType_32BGRA ||
            sourcePixelFormat == kCVPixelFormatType_32RGBA)
      
      
          let imageChannels = 4
          assert(imageChannels >= inputChannels)
      
          // Crops the image to the biggest square in the center and scales it down to model dimensions.
          let scaledSize = CGSize(width: inputWidth, height: inputHeight)
          guard let thumbnailPixelBuffer = pixelBuffer.centerThumbnail(ofSize: scaledSize) else {
            return nil
          }
      
          let outputTensor: Tensor
          do {
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
      
            // Remove the alpha component from the image buffer to get the RGB data.
            guard let rgbData = rgbDataFromBuffer(
              thumbnailPixelBuffer,
              byteCount: batchSize * inputWidth * inputHeight * inputChannels
              ) else {
                print("Failed to convert the image buffer to RGB data.")
                return nil
            }
      
            // Copy the RGB data to the input `Tensor`.
            try interpreter.copy(rgbData, toInputAt: 0)
      
            // Run inference by invoking the `Interpreter`.
            try interpreter.invoke()
      
            // Get the output `Tensor` to process the inference results.
            outputTensor = try interpreter.output(at: 0)
          } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return nil
          }
      
          let results = [Float32](unsafeData: outputTensor.data) ?? []
      
          // Process the results.
          let topNInferences = getTopN(results: results)
      
          // Return the inference time and inference results.
          return topNInferences
        }
      
        // MARK: - Private Methods
      
        /// Returns the top N inference results sorted in descending order.
        private func getTopN(results: [Float]) -> [Inference] {
          // Create a zipped array of tuples [(labelIndex: Int, confidence: Float)].
          let zippedResults = zip(labels.indices, results)
      
          // Sort the zipped results by confidence value in descending order.
          let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(resultCount)
      
          // Return the `Inference` results.
          return sortedResults.map { result in Inference(confidence: result.1, label: labels[result.0]) }
        }
      
        /// Loads the labels from the labels file and stores them in the `labels` property.
        private func loadLabels(fileInfo: FileInfo) {
          let filename = fileInfo.name
          let fileExtension = fileInfo.extension
          guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
            fatalError("Labels file not found in bundle. Please add a labels file with name " +
              "\(filename).\(fileExtension) and try again.")
          }
          do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            labels = contents.components(separatedBy: .newlines)
          } catch {
            fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
              "valid labels file and try again.")
          }
        }
      
        /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
        ///
        /// - Parameters
        ///   - buffer: The pixel buffer to convert to RGB data.
        ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
        ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
        ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
        ///       floating point values).
        /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
        ///     converted.
        private func rgbDataFromBuffer(
          _ buffer: CVPixelBuffer,
          byteCount: Int
          ) -> Data? {
          CVPixelBufferLockBaseAddress(buffer, .readOnly)
          defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
          guard let mutableRawPointer = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
          }
          let count = CVPixelBufferGetDataSize(buffer)
          let bufferData = Data(bytesNoCopy: mutableRawPointer, count: count, deallocator: .none)
          var rgbBytes = [Float](repeating: 0, count: byteCount)
          var index = 0
          for component in bufferData.enumerated() {
            let offset = component.offset
            let isAlphaComponent = (offset % alphaComponent.baseOffset) == alphaComponent.moduloRemainder
            guard !isAlphaComponent else { continue }
            rgbBytes[index] = Float(component.element) / 255.0
            index += 1
          }
      
          return rgbBytes.withUnsafeBufferPointer(Data.init)
      
        }
      }
      
      // MARK: - Extensions
      
      extension Data {
        /// Creates a new buffer by copying the buffer pointer of the given array.
        ///
        /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
        ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
        ///     data from the resulting buffer has undefined behavior.
        /// - Parameter array: An array with elements of type `T`.
        init<T>(copyingBufferOf array: [T]) {
          self = array.withUnsafeBufferPointer(Data.init)
        }
      }
      
      extension Array {
        /// Creates a new array from the bytes of the given unsafe data.
        ///
        /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
        ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
        ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
        /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
        ///     `MemoryLayout<Element>.stride`.
        /// - Parameter unsafeData: The data containing the bytes to turn into an array.
        init?(unsafeData: Data) {
      
          guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
          #if swift(>=5.0)
          self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
          #else
          self = unsafeData.withUnsafeBytes {
            .init(UnsafeBufferPointer<Element>(
              start: $0,
              count: unsafeData.count / MemoryLayout<Element>.stride
            ))
          }
          #endif  // swift(>=5.0)
        }
      }
      """
      
      prompt_template = """
      Can you please explain how this code works?
      
      {question}
      
      Use a lot of detail and make it as clear as possible.
      """
      
      completion = generate_text(
          prompt = prompt_template.format(question=CODE_BLOCK)
      )
      print(completion.result)
      
      The `ModelDataHandler` class is a TensorFlow Lite model handler that loads the MobileNet model and labels files, preprocesses the input image, and invokes the `Interpreter` to run inference. The results are then processed and returned.
      
      The `ModelDataHandler` class has the following properties:
      
      * `threadCount`: The current thread count used by the TensorFlow Lite Interpreter.
      * `resultCount`: The number of top results to return.
      * `batchSize`: The batch size of the input data.
      * `inputChannels`: The number of input channels.
      * `inputWidth`: The input width.
      * `inputHeight`: The input height.
      * `labels`: A list of labels from the labels file.
      * `interpreter`: A TensorFlow Lite `Interpreter` object for performing inference on a given model.
      * `alphaComponent`: Information about the alpha component in RGBA data.
      
      The `ModelDataHandler` class has the following initializers:
      
      * `init(modelFileInfo: FileInfo, threadCount: Int = 1)`: A failable initializer that creates a new instance of `ModelDataHandler` if the model and labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
      
      The `ModelDataHandler` class has the following methods:
      
      * `runModel(onFrame pixelBuffer: CVPixelBuffer) -> [Inference]?`: Performs image preprocessing, invokes the `Interpreter`, and process the inference results.
      * `getTopN(results: [Float]) -> [Inference]`: Returns the top N inference results sorted in descending order.
      * `loadLabels(fileInfo: FileInfo)`: Loads the labels from the labels file and stores them in the `labels` property.
      * `rgbDataFromBuffer(_ buffer: CVPixelBuffer, byteCount: Int) -> Data?`: Returns the RGB data representation of the given image buffer with the specified `byteCount`.
      
      The `ModelDataHandler` class also has two extension methods:
      
      * `extension Data`: Creates a new buffer by copying the buffer pointer of the given array.
      * `extension Array`: Creates a new array from the bytes of the given unsafe data.
      
      Here is an example of how to use the `ModelDataHandler` class:
      
      ```
      let modelDataHandler = ModelDataHandler(modelFileInfo: MobileNet.modelInfo)
      let inferenceResults = modelDataHandler.runModel(onFrame: pixelBuffer)
      ```
      
      The `inferenceResults` array will contain the top N inference results sorted in descending order.

We can also ask an LLM to document a complex code base:

      prompt_template = """
      Please write technical documentation for this code and \n
      make it easy for a non swift developer to understand:
      
      {question}
      
      Output the results in markdown
      """
      
      completion = generate_text(
          prompt = prompt_template.format(question=CODE_BLOCK)
      )
      print(completion.result)
      
      
      ## ModelDataHandler
      
      The `ModelDataHandler` class handles all data preprocessing and makes calls to run inference on a given frame
      by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
      results for a successful inference.
      
      ### Public Properties
      
      * `threadCount`: The current thread count used by the TensorFlow Lite Interpreter.
      * `resultCount`: The number of top results to return.
      
      ### Model Parameters
      
      * `batchSize`: The number of images to be processed at a time.
      * `inputChannels`: The number of color channels in the input images.
      * `inputWidth`: The width of the input images.
      * `inputHeight`: The height of the input images.
      
      ### Private Properties
      
      * `labels`: A list of labels from the given labels file.
      * `interpreter`: A TensorFlow Lite `Interpreter` object for performing inference on a given model.
      * `alphaComponent`: Information about the alpha component in RGBA data.
      
      ### Initialization
      
      The `ModelDataHandler` class is initialized with a model file info and a thread count. The model file info
      is used to load the model and labels files from the app's main bundle. The thread count is used to
      specify the number of threads that the `Interpreter` should use to perform inference.
      
      ### Public Methods
      
      * `runModel(onFrame:)`: Performs image preprocessing, invokes the `Interpreter`, and process the inference results.
      
      ### Private Methods
      
      * `getTopN(results:)`: Returns the top N inference results sorted in descending order.
      * `loadLabels(fileInfo:)`: Loads the labels from the labels file and stores them in the `labels` property.
      * `rgbDataFromBuffer(_:byteCount:)`: Returns the RGB data representation of the given image buffer with the specified `byteCount`.
      
      ## Extensions
      
      The `Data` and `Array` types are extended to support creating buffers and arrays from unsafe data.
      
      ### Data
      
      The `Data` type is extended to support creating a new buffer by copying the buffer pointer of the given array.
      
      ### Array
      
      The `Array` type is extended to support creating a new array from the bytes of the given unsafe data.
      
      ## Output
      
      The `ModelDataHandler` class returns the top N inference results sorted in descending order. The results
      are formatted as a list of tuples, where each tuple contains the label index and confidence value for an
      inference result.
      
      For example, the following code would return a list of tuples containing the top 3 inference results for
      an image:
      
      ```
      let results = ModelDataHandler.runModel(onFrame: pixelBuffer)
      
      for (index, confidence) in results {
        print("Label: \(labels[index]), Confidence: \(confidence)")
      }
      ```
      
      The output would be similar to the following:
      
      ```
      Label: Dog, Confidence: 0.99
      Label: Cat, Confidence: 0.01
      Label: Shoe, Confidence: 0.00
