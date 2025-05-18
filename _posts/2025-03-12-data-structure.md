---
layout: default
title:  "什么是数据结构? 如何实现一个最基本的数据结构"
date:   2025-03-15 00:13:22 +0800
categories: jekyll update
rating: 0
---

### 前言

在上一篇博文中，我写到了自己对程序的认识，讲到一个程序中一定会有数据和对数据的操作这两个部分。因此如何在计算机中组织数据便是程序编写人员的重要功课。我们用“数据结构”这个术语来表示这种数据在计算机中的组织方式。

上一篇博文的内容主要来自HTDP这本书的第一章，而这本书后面的内容有很大一部分是关于数据结构的。
本博文我会继续这本书的旅程，也参考了作者的一些其他的书，包括《A Data-Centric Introduction to Computing》、《SICP》和《Concrete Abstructions》。首先还是会用自己的话进一步解释数据结构。然后用Python实现Pair, 单向链表和查找表格等简单的数据结构。

### 什么是数据结构

数据结构（英语：data structure）是计算机中存储、组织数据的方式。这句话其实并没有进一步解释的空间，但是它比较抽象，新手可能会觉得很难想象数据可能的存储和组织方式。

想象你在工作中需要用处理表格数据，你想编程自动化这些任务。你可以把这些需要处理的数据当初一个字符串，然后用正则表达式进行模式匹配。但是这样效率会很低，因为每次匹配都会遍历整个字符串；也可以一开始就在程序中使用表格的形式表示它们，然后通过行列来定位数据。在Python中可以使用Panda做到这一点。

在上面这个场景中字符串和表格就是两种不同的数据结构。字符串顾名思义就是把单个字符串联起来形成的数据结构，因为单个字符通常并不能反应我们需要的信息，在这种场景下，把能反映信息的数据单元用行列的方式组织起来更有效率。

数据结构的应用并不仅限于在内存中，对于持久化数据也至关重要。比如，例如数据库索引使用B+树加速磁盘检索，再比如区块链技术通过哈希指针（一种特殊链表结构）串联数据块，确保历史记录的不可篡改性。


### 最基本的数据结构 pair

通过上面的描述，相信你已经对数据结构这个概念有所了解了。但是只知道概念还是不够，还是需要通过实践掌握具体的数据结构，知道不同的结构有什么样的特点，适合什么样的场景。这样在需要写程序来解决问题的时候才能胸有成竹。

我们从Pair这种数据结构说起，它只有两个部分。原子数据谈不上数据结构，所以只有俩个数据的pair就是最基本的结构了，其他复杂的结构都是在这个结构上的基础上构建的。

对于只有两个数据的结构我们可以把两个数据放在一起，也可以取出其中的一个进行操作。

对于pair这个数据结构，在python中我们会直接使用tuplue来处理，因为内置的数据类型处理数据更快。加上这篇博文的内容其实是来自于介绍函数式编程的书，Python对其支持有限（比如不支持尾递归优化），不过对于我们学习和探索的目的已经足够了。

实现这些结构的方法有很多，首先用函数的方式来实现：
```python
    # 首先是构造函数，给任意两个数据，我们保存这两个数据，也就是保留对它们的操作
    def create_pair(a, b):
        return lambda f: f(a, b)

    # 取出任何连个数据中的一个
    def get_first(self):
        return lambda a, b: a

    def get_second(self):
        return lambda a, b: b

    # 还需要一种方法来判断任何一个结构的数据是否是一个Pair
    def is_pair(pair):
        try:
            pair(lambda a, b: None)
            return True
        except TypeError:
            return False
```

在Python中用对象来实现更适合：
```python
    class Pair:
        def __init__(self, a, b):
            """
            Initialize a Pair object with two elements.
        def is_pair(self):
           
            Parameters:
            a -- the first element of the pair
            b -- the second element of the pair
            """
            self.a = a
            self.b = b

        # 获取两个数据中其中一个的方法也叫Getter
        def get_first(self):
            return self.a

        def get_second(self):
            return self.b

        # 用对象实现这种结构的话，判断一个实例是不是这个对象或者继承这个对象的实例即可。
        def is_pair(self):
            return isinstance(self, Pair)

```
(未完待续)