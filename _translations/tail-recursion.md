---
title: "What Is Tail-Recursion Optimization? Turning Recursion into Iteration"
description: "Distinguish recursive functions from recursive processes, understand tail calls, and convert factorial and binary-tree traversal into iterative Python code."
translation_key: tail-recursion
date: 2025-04-23 17:30:42 +0800
translated_at: 2026-09-26
---

#### Why is recursion so divisive?

Classic computer science books such as [SICP](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html) devote a great deal of space to recursion. Some modern languages, including Python, take a much more cautious approach. Why does a way of thinking that feels as natural as breathing in functional programming seem almost taboo in parts of the Python community?

What exactly is recursion? What problems does it create? When some programmers love it and others avoid it, what are they actually disagreeing about? I was curious, so I looked into these questions. This article organizes what I found, drawing mainly on section 1.2 of SICP. I will explain the basic concepts, discuss the Python creator's position, and show how to turn recursion into iteration in a language without tail-recursion optimization.

#### What recursion means

We should start by clarifying the word *recursion*, because it is the source of many misunderstandings. The difficulty is that it means different things in different contexts.

In computer science, recursion can describe a problem-solving method: repeatedly breaking a problem into smaller problems of the same kind. When we call a function or data structure recursive, we mean that its definition refers back to itself. When we describe a computational process as recursive, we mean that a calculation must wait for another calculation's result before it can finish.

A tree is a familiar example of a recursive data structure: a node can contain references to other nodes with the same structure. We will return to trees later.

Here is a recursive function that computes a factorial:

```python
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)
```

The function calls itself, so it is a recursive function. Consider how it evaluates an input of 5:

```text
factorial(5)
= 5 * factorial(4)
= 5 * 4 * factorial(3)
= 5 * 4 * 3 * factorial(2)
= 5 * 4 * 3 * 2 * factorial(1)
= 5 * 4 * 3 * 2 * 1
= 5 * 4 * 3 * 2
= 5 * 4 * 6
= 5 * 24
= 120
```

We first get `5 * factorial(5 - 1)`. We can calculate `5 - 1`, but we do not yet know the value of `factorial(4)`. We therefore have to remember the pending multiplication by 5 until that value is available. This pending work is kept on a **stack**, which should not be confused with a heap.

The process first expands as it accumulates unfinished work, then contracts as that work is completed. This is a recursive process.

Now consider a different recursive function:

```python
def fact_iter(n, acc=1):
    if n == 1:
        return acc
    else:
        return fact_iter(n - 1, n * acc)
```

Its evaluation looks like this:

```text
fact_iter(5)
= fact_iter(5, 1)
= fact_iter(4, 5)
= fact_iter(3, 20)
= fact_iter(2, 60)
= fact_iter(1, 120)
= 120
```

This function also calls itself. An input of 5 means evaluating `fact_iter(5, 1)`, which leads to `fact_iter(5 - 1, 5 * 1)`. Both arguments are evaluated first, producing `fact_iter(4, 5)`, and so on.

There is no expanding chain of pending multiplications. Instead, a fixed number of state variables carries all the information needed for the next step. This is a **linear iterative process**.

Yes: the function is recursive, but the process it describes is iterative. That distinction concerns the conceptual evaluation process. A language with tail-recursion optimization can implement it that way; Python still adds a stack frame for each call.

#### Tail recursion and tail-recursion optimization

Now that we have distinguished these meanings of recursion, we can look at *tail recursion*.

A tail-recursive call is a function calling itself in **tail position**. It is a special case of a **tail call**, where the last action of a function is to return the result of another function call.

An example makes this easier to see:

```python
def foo(data):
    a(data)
    return b(data)
```

The call to `b` is in tail position: its result becomes the result of `foo`, with no further work to do. The call to `a` is not in tail position, because the call to `b` still follows it.

If that final call were to `foo` itself rather than to `b`, it would be a tail-recursive call.

Think back to the two factorial functions. The first returns `n * factorial(n - 1)`. The multiplication still has to happen after the recursive call returns, so the recursive call is not a tail call. The second function returns the recursive call's result directly, so it is tail-recursive.

In the first version, `factorial(5)` leaves four multiplications pending. Computing `factorial(10000)` the same way would leave 9,999 calls waiting to complete their multiplications. Stack memory grows with the recursion depth. That is the problem.

In the tail-recursive version, the state is passed into the next call as arguments; no earlier state is needed after that call finishes. A language implementation can take advantage of this by reusing or discarding the previous frame instead of preserving it on the stack. This greatly reduces the stack space required.

A recursive function therefore does not necessarily require growing stack space: a tail-recursive function can run in constant stack space when the language implementation supports the optimization.

[Scheme](https://www.r6rs.org/final/html/r6rs/r6rs-Z-H-8.html), the language used in SICP, makes proper tail calls part of its language requirements. [Lua](https://www.lua.org/pil/6.3.html) also guarantees proper tail calls. [ECMAScript 2015](https://262.ecma-international.org/6.0/#sec-isintailposition) includes a proper-tail-call requirement for eligible calls in strict-mode code. The practical benefit still depends on whether the implementation supports it.

Scheme programmers often define a helper inside a function and return the result of calling it. In Python, the following pattern adds an outer stack frame as well as the recursive helper's frames:

```python
def factorial(n):
    def inner_fact_iter(n, acc):
        if n == 1:
            return acc
        else:
            return inner_fact_iter(n - 1, n * acc)

    return inner_fact_iter(n, 1)
```

Python's creator, Guido van Rossum, explains in his [blog post](https://neopythonic.blogspot.com/2009/04/tail-recursion-elimination.html) why he does not want tail-recursion elimination in Python. Three of the reasons discussed there are:

1. It removes information from stack traces, making debugging harder.
2. It affects portability between implementations. If programmers rely on optimized tail recursion, their code may fail on an implementation that does not provide it. It is therefore more than an invisible implementation detail.
3. He does not see recursion as the foundation for every programming task. In his words: “I don't believe in recursion as the basis of all programming”.

When we use Python, we need to work with those choices. A typical Python installation has a recursion limit of around 1,000, configurable through `sys.setrecursionlimit`. Exceeding the limit raises a recursion error.

Advice about when to use recursion and when to use a loop varies widely, often offering conclusions without a way to make the decision yourself. The discussion above suggests a more useful criterion: **look at the shape of the data**.

In an earlier article, I argued that a well-designed program should reflect the structure of the data it processes. Some data structures refer to themselves in their definitions and are naturally recursive. Trees are the most obvious example, which is why recursive tree traversal is often the clearest expression of the task.

Van Rossum also discusses tree traversal in his post, arguing that a depth of 1,000 is generally enough for such uses. He concludes that tail-recursive code is easy to rewrite as an ordinary loop. Let us see how that conversion works.

#### From recursion to iteration

Start with the factorial example.

If Python supported tail-recursion optimization, changing `factorial` into `fact_iter` would address the growing stack. Since it does not, we need one more step: replace the tail-recursive calls with iteration.

Each call in `fact_iter` changes `n` and the accumulator. If we update those values through assignment instead of making another call, we get:

```python
def fact(n, acc=1):
    while n > 1:
        (n, acc) = (n - 1, acc * n)
    return acc
```

We have converted the tail-recursive function into a loop.

Now consider a more involved case: traversing a binary tree.

```python
# A binary tree is either empty or a node with two subtrees.
import collections
Node = collections.namedtuple('Node', 'val left right')

# Some examples.
tree0 = None  # empty tree
tree1 = Node(5, None, None)
tree2 = Node(7, tree1, None)
tree3 = Node(7, tree1, Node(9, None, None))
tree4 = Node(2, None, tree3)
tree5 = Node(2, Node(1, None, None), tree3)

# Flatten a tree using recursive in-order traversal.
def flatten_recursive(bst):
    if bst is None:
        return []
    return flatten_recursive(bst.left) + [bst.val] + flatten_recursive(bst.right)


# Iterative in-order traversal replaces recursion with an explicit stack.
def flatten_iterative(bst):
    result = []
    stack = []
    current = bst

    while current or stack:
        # Descend into the left subtree.
        while current:
            stack.append(current)
            current = current.left

        # Return to the most recently saved node.
        current = stack.pop()
        result.append(current.val)

        # Move to the right subtree.
        current = current.right

    return result

# Do not forget to test.
assert flatten_recursive(tree5) == [1,2,5,7,9]
assert flatten_iterative(tree5) == [1,2,5,7,9]
```

The iterative version looks much more complicated. It may take a moment to follow, so here is a trace using `tree5`. The numbers below identify nodes by their values:

1. Start with `current=2` and `stack=[]`. Push node 2 and move to its left child, node 1.
2. Now `current=1` and `stack=[2]`. Push node 1 and move to its left child, `None`.
3. With `current=None`, pop node 1 and record its value: `result=[1]`.
4. Node 1 has no right child, so pop node 2 next and record its value: `result=[1,2]`.
5. Move to node 2's right child, node 7. Push node 7 and descend to its left child, node 5.

Continue in the same way: descend left while pushing nodes, then pop, record the value, and move right. Stop when the stack is empty and there is no current node. The final result is `[1,2,5,7,9]`.

Comparing the two conversions reveals an important difference. The factorial loop needs only two state variables; the tree traversal must maintain a stack. Why?

The tail-recursive factorial already has a fixed number of state variables. In-order traversal, however, must remember what to do after finishing a left subtree. That memory has to live somewhere. In the recursive version, it is kept implicitly on the call stack. In the iterative version, it becomes the explicit `stack` in our code.

**The conversion does not eliminate state; it makes implicit state explicit.** That also explains why the logic seems more complicated: the complexity has moved from the language runtime into our own code.

#### Summary

A function can call itself without describing a process that accumulates pending work. Tail-recursive functions can describe linear iterative processes, and tail-recursion optimization lets them run in constant stack space. Python deliberately does not provide that optimization, partly for debugging and consistency reasons.

In Python, simple tail recursion such as the factorial example can become a loop with state variables. Traversal of a naturally recursive structure such as a tree needs an explicit stack to replace the call stack. Understanding that relationship helps us choose between clarity of expression and control over resource use.
