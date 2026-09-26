---
title: "What Is a Computer Program? How Do You Design One?"
description: "Reflections on How to Design Programs: programs, functions, data, and a systematic design recipe for turning problems into working code."
translation_key: how-to-design-programs
date: 2025-03-05 12:43:42 +0800
translated_at: 2026-09-26
---

#### Introduction

When you think about learning to program, do you immediately picture obscure code and unfamiliar terms whose purpose you cannot see? Does that leave you unsure where to begin?

Learning to program is often treated as a process of collecting tools: first learn variables and assignment, then functions, then combine them into objects. After gradually learning a language's features, try putting them together to see what you can make. The next step might be learning more languages. This approach can leave the impression that successful programming mainly depends on trial and error, or on some special talent.

Today I want to share a book that may change that impression and give you a clearer understanding of programming: *How to Design Programs*.

The book aims to give learners a fundamental understanding of programs and a systematic way to think about, plan, and design them. To use an analogy, it teaches you to be an architect: draw the plans first, then choose the tools and start building.

This article reflects on material from the opening part of the book. I will summarize its view of what a program is in my own words, then explain the method the authors call a **design recipe**.

This is a basic topic, but one that is often overlooked. If you know little about computer programs, or have never thought much about the reasoning behind them, I hope this article will make these ideas clearer.

#### Computer programs

Let us first establish what a computer is: a general-purpose device that accepts input, processes data, and produces output.

Here, *general-purpose* means that a computable problem can, in principle, be solved by programming. Not every problem that can be stated precisely is computable; computability theory studies that boundary. A calculator, by comparison, has a fixed set of functions determined when it is built. A computer leaves the question of what to do to the program: the same machine can process images, analyze data, or run games when given different programs.

In other words, to use a computer to solve a problem, we must tell it what to do through instructions it can understand. A collection of such instructions forms a computer program.

A program describes how a computer operates on information: how to accept input, transform data, and produce meaningful output.

A program can take the form of an executable, such as a Windows `.exe`, or an application packaged as a macOS `.app`. It can also exist as source code, such as a C `.c` file or a Python `.py` file, written by a programmer.

Writing source code requires a programming language. Just as communicating with someone who does not speak your language requires vocabulary and grammar they understand, communicating instructions to a computer requires a suitable language.

Interestingly, programming languages themselves also need programs to implement them. The book's authors work in programming-language semantics, teach university [courses on interpreters](https://www.youtube.com/watch?v=3N__tvmZrzc&list=PLbjZkI1beTSM2AlyXH3KudhmNs9Xn3Z1b), and have written [books on the subject](https://www.plai.org/). The progression from programming fundamentals to interpreters, and then to computer architecture and compilers, is also what I love about the classic *Structure and Interpretation of Computer Programs* (SICP). I plan to write more about these topics.

Programmers use a programming language to write code that describes computations. Expressions are a central building block of that code. From this perspective, programming means writing expressions, and the systematic approach advocated by the authors means organizing those expressions according to a clear logic and a set of rules.

There is a qualification here: not every statement is an expression, at least not in every language. Consider JavaScript and Python. In JavaScript, `let x; console.log(x = 5)` works, while Python's `print(x = 5)` raises an error. Likewise, an `if` statement in these languages is not an expression that produces a value. Python does, however, provide assignment expressions and conditional expressions; `print(x := 5)` works.

This distinction comes from language-design choices, not hardware. If ordinary assignment were an expression, accidentally writing `if x = 5:` instead of `if x == 5:` could silently produce valid code. This is a notorious source of bugs in the C family of languages. Python gives up some compactness to reject that kind of mistake at the syntax level; its later assignment-expression syntax uses the distinct symbol `:=`. Some languages make everything an expression to allow flexible composition, while others deliberately distinguish statements from expressions to prevent mistakes. These trade-offs reinforce the point above: programming languages are themselves designed, and every syntax rule reflects a decision.

The book comes from the Racket community and introduces a functional approach to programming, emphasizing conceptual understanding rather than the details of a particular implementation. If you have not learned enough programming to follow the examples above, do not worry. Thinking of a program as a collection of expressions is a simplification, but a useful starting point.

#### Functions

If we want to organize expressions systematically, we need a way to group them. This book uses functions.

Most of us encountered functions in school mathematics. Mathematical functions and programming functions have enough in common that comparing them is helpful, but their differences also matter.

In mathematics, we might write $y = x + 5$ or $f(x) = x + 5$. Giving $x$ different values changes $f(x)$, or $y$, accordingly. For example, $f(3) = 8$. A given input determines a unique output. The equals sign means the two sides are interchangeable: we can replace $f(3)$ with $8$, or vice versa.

In a program, we may need to write similar expressions repeatedly. We can capture the common procedure, provide the varying values as inputs through parameters, and return the result as output. This gives us a programming function.

For example, in Python we can define a function for the calculation above:

```python
def add5(x: int) -> int:
   '''
   Adds five to the provided integer.

   Parameters: x (int).

   Returns: int.

   examples:  add5(3) == 8;
              add5(5) == 10.
   '''
   y = x + 5
   return y

print(add5(3))  # Output: 8
print(add5(5))  # Output: 10
```

We have defined a function named `add5`. From now on, we can call it whenever we want to add five to a number, without writing out the underlying expression again.

The main difference is that a mathematical function describes a relationship, while calling a programming function is an event. A call may return a result, perform an action without producing a useful return value, or do both. A function can also affect the outside world by printing text, changing a variable, or writing a file. These actions beyond producing a return value are called **side effects**. A function may also read from the outside world, such as user input or the current time, so the same call can produce different results.

For now, we will consider only **pure functions**: functions that always produce the same output for the same input and do nothing beyond returning that output. They behave like mathematical functions. For example, we can replace `add5(3)` with `8` without changing the program's behavior.

Now consider a counterexample:

```python
count = 0
def add5(x):
   global count
   count = count + 1
   return x + 5
```

This function still returns the same output for the same input, but it also changes the external variable `count`. Substitution is no longer safe. Replacing `add5(3)` with `8` preserves the returned value, but `count` is incremented one fewer time, so the program behaves differently. The point where that equality breaks down is where our mathematical intuition stops being sufficient.

Programming functions can also work with many kinds of data, including numbers, text, and lists. They are not limited to the numeric examples familiar from school mathematics, which allows them to express a wide variety of tasks.

#### Programs and functions

We said earlier that programs accept information and produce information, while functions accept data and produce data. These views fit together: a program is the complete path from input to output, and its functions are the processing steps along that path.

*Data* and *information* are everyday words, but there is a useful relationship between them here. The information a program handles and the data its functions operate on are different representations of the same thing. What we call information in the outside world becomes data inside the computer. For example, temperatures or text can be encoded as 32-bit floating-point numbers or UTF-8 byte sequences.

A function abstracts an operation on data. We might write one to convert Fahrenheit to Celsius. A program abstracts a complete task. Now imagine writing a program that converts Fahrenheit to Celsius: it would be very simple, needing only to call that function as its calculation step. I am feeling too lazy to write the example—can you write it yourself?

The distinction is that a program responds to events in the outside world, such as mouse clicks or keyboard input, and produces output intended to affect that world, such as information for a user or commands to an external device. Functions are defined and called within the program.

At the core of a program are data and ways of processing those data, which we call functions. We can therefore refine our definition again: programming is the process of correctly defining data, defining functions, and calling those functions.

Not everyone approaches programming by separating data from functions. Object-oriented programming, for example, groups data and the methods that operate on them into objects. But these approaches share a broader principle: when designing complex programs, separate different responsibilities. The MVC architecture familiar from software engineering applies that principle along another dimension. The book's separation of data and functions is a particularly simple expression of it.

#### How to program

We now have several descriptions of a computer program. “A program is a collection of expressions” is already useful, but it can suggest that any collection of expressions that runs counts as successful programming. That does not help us much with using programs to solve problems.

Thinking in terms of data and functions gives the work a clearer direction. We need to ask what result we want, what information we need to supply, how to operate on that information, and how to represent both the information and the operations in a computer.

This resembles how we think through problems ourselves: we turn information about the world into concepts, then organize and manipulate those concepts in our minds. In that sense, programming is a form of thinking, and learning to program is learning to think.

Good programmers continually develop two basic skills:

- Representing information using suitable data types.
- Transforming data with as few computing resources as possible.

Different kinds of information need different ways of organizing data. These arrangements are called **data structures**. I have a [separate post on data structures (Chinese; currently withdrawn for revision)]({% post_url 2025-03-12-data-structure %}). Likewise, there may be several ways to carry out the same information-processing task; these methods are called **algorithms**. If you are new to programming, I hope the familiar saying “programs are data structures plus algorithms” now means a little more.

##### The design recipe

We now understand what a program is and have a rough idea of how it can solve problems. But when faced with a specific programming task, it may still be difficult to start writing code. To address that difficulty, the authors propose a sequence of steps that supports our thinking. They call it a **design recipe**:

1. **Data definitions**
   - Identify the information that needs to be represented and decide how to represent it in the programming language.
   - Write data definitions and illustrate them with examples.

2. **Function header and signature**
   - State the types of data the function accepts and produces.
   - Briefly answer the question, “What does this function compute?”
   - Write a function stub consistent with the signature.

3. **Examples**
   - Illustrate the function's purpose with concrete examples.

4. **Function template**
   - Translate the data definition into the basic structure of the function.

5. **Function implementation**
   - Complete the template using the purpose statement and examples.
   - Fill in the missing parts.

6. **Testing**
   - Turn the examples into test cases and check that the function passes them.
   - Tests reveal mistakes and help others understand the function definition.
   - Testing is necessary for any serious program.

Each heading names the result of that step; the points beneath it describe its purpose and process.

You may have noticed that the earlier `add5` example already shows the products of several of these steps: a parameter and return-type signature, a one-sentence purpose statement, and two examples that can become tests. The example with “more comments than code” illustrates how the design recipe shapes even a small function.

This is the “draw the plans before building” approach from the introduction. The authors want programmers to follow the process deliberately, rather than keep experimenting until something runs. They regard this discipline as an important distinction between professional and amateur programming.

At first, I thought the recipe was just a requirement to write lots of comments, and it felt tedious. The Python example above is so simple that its behavior is obvious at a glance, yet its documentation is several times longer than the implementation. That seemed unnecessary. Only after watching the authors speak did I understand the purpose of the process. It rests on several ideas:

1. **Do not rush into code.** A common experience is to finish writing something and discover that it does not match the actual requirements. Reworking an implementation can take far more time than revising a design.

2. **Use examples to understand the problem.** Examples are central both to defining data and to designing functions. They clarify the problem and can later become tests for the implementation.

3. **Let the data structure guide the function's structure.** At first glance, this can look like the most redundant step. In fact, it is the most important and revealing one. Organizing code around the data helps clarify levels of structure, avoid mistakes in nested loops, and keep the logic readable. Repeatedly thinking about the relationship between data and program structure gradually builds an intuition: when you see the data structure, you begin to anticipate the code structure.

4. **Expect to test and revise.** Writing perfect code on the first attempt is difficult.

These ideas are fairly straightforward. For a simple problem, the process may feel excessive. But when a problem is more complex, especially when you do not know where to begin, try the authors' design recipe.

#### Closing thoughts

The value of this process becomes clearest with more complex problems. Yet without practice, you may not think to use it when you need it. Much of the book therefore explores how to adapt the recipe to different problems and situations.

The book begins with atomic data and gradually introduces more complex data structures and design patterns. I will cover some of those ideas in later posts. If you are interested, you can also read the book yourself.
