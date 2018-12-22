---
title: Python中的迭代器和生成器
description: Python中的迭代器和生成器总会在有意无意的时候使用到，但是对内部原理一直理解的不是很透彻。总结了一下Python中迭代器和生成器的基本用法。

categories:
 - Python
tags: Python
---


# 迭代器和生成器
## 迭代器(Iterator)
&nbsp;&nbsp;&nbsp;&nbsp;迭代器是访问集合内元素的一种方式，一般用来遍历数据。迭代器和以下标对数据进行访问不同，迭代器提供了一种惰性访问数据的方式。

### 迭代协议
&nbsp;&nbsp;&nbsp;&nbsp;实现了魔法方法\_\_iter\_\_的对象即为可迭代对象，例如python中自带的列表list即实现了该方法，因此list对象是可迭代对象，但是list并不是迭代器Interator。可迭代对象可以通过iter方法来包装成迭代器。

### 实现迭代器
&nbsp;&nbsp;&nbsp;&nbsp;若在可迭代对象中定义的\_\_iter\_\_ 方法的返回类型是一个迭代器对象，在满足此前提的情况下可用iter将可迭代对象包装成一个迭代器。如下例：

```python
# list在python内部已经实现了__iter__方法
test = [1, 2, 3]
iter_rator = iter(test)
print(isinstance(test, Iterator))   # False
print(isinstance(iter_rator, Iterator))  # True
```
&nbsp;&nbsp;&nbsp;&nbsp;下例将报错: 'TypeError: 'Company' object is not iterable',  因为自定义类中并没有定义\_\_iter__魔法方法，因此不能使用iter包装成一个迭代器。

```python
class Company(object):
    def __init__(self, employee_list):
        self.employee = employee_list

company = Company(["tom", "bob", "jane"])
my_itor = iter(company)

```
  &nbsp;&nbsp;&nbsp;&nbsp;若想将自定义类定义成迭代器，在实现\_\_iter\_\_魔法方法之后，除了像上例代码中使用iter函数包装外，还可以通过实现\_\_next\_\_魔法方法来实现。下例中定义了MyIterator迭代器类，该类继承了Iterator，并实现了\_\_next\_\_方法，在该方法内定义了真正返回迭代值的逻辑。<br>
  &nbsp;&nbsp;&nbsp;&nbsp;可能有人会有疑问，为什么不在Company中直接定义\_\_next\_\_魔法方法。让其成为一个迭代器？<br>
  &nbsp;&nbsp;&nbsp;&nbsp;在Company类中除了定义\_\_iter\_\_魔法方法外直接实现\_\_next\_\_魔法方法并不违反python语法，但是这种实现方法违反了迭代器的设计模式，因此下例中分别实现了Company类和MyIterator迭代器类。

```python
from collections.abc import Iterator


class Company(object):
    def __init__(self, employee_list):
        self.employee = employee_list

    def __iter__(self):
        return MyIterator(self.employee)


class MyIterator(Iterator):
    def __init__(self, employee_list):
        self.iter_list = employee_list
        self.index = 0

    def __next__(self):
        # 真正返回迭代值的逻辑
        try:
            word = self.iter_list[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return word


if __name__ == "__main__":
    company = Company(["tom", "bob", "jane"])
    my_itor = iter(company)
    # 也可以不通过iter包装company对象，而直接使用for循环来遍历。
    # 这是因为python的for in语句中已实现了相应方法。在调用for循环时会去自动调用iter方法
    for item in company:
        print (item)
```
  &nbsp;&nbsp;&nbsp;&nbsp;若没有实现\_\_iter\_\_魔法方法，但是实现了\_\_getitem\_\_魔法方法，也可以调用iter方法来包装成迭代器。这是因为所有实现了\_\_getitem\_\_魔法方法的对象都可以使用python中的for循环来遍历，而可用for循环来遍历的对象都是可迭代对象。因此在使用iter方法来包装对象时，会优先寻找该对象内有没有\_\_iter\_\_魔法方法，若不存在该方法，则会进而寻找\_\_getitem\_\_魔法方法，若仍然不存在，则确定该对象不是可迭代对象。即无法使用iter方法。
## 生成器
&nbsp;&nbsp;&nbsp;&nbsp;只要定义的函数内有yield关键字，该函数就不再是一个普通的函数，而是一个生成器函数(生成器对象)。生成器也实现了迭代器协议。生成器对象都是迭代器，如下例所示：

```python
def gen_func():
    yield 1
    yield 2
    yield 3


test = gen_func()
print(isinstance(test, Iterable))  # True
print(isinstance(test, Iterator))  # True
print(next(test))  # 1
```

### 生成器的应用
#### 使用生成器从写Company类
&nbsp;&nbsp;&nbsp;&nbsp;带有yield关键字的函数都为迭代器，可以使用生成器函数从写Company类。在之前的实现中，先定义了\_\_iter\_\_魔法方法，在该方法中，返回了另一个迭代器对象(该对象中实现了\_\_next__魔法方法)，所以可以直接使用next方法来遍历。在有了生成器之后，可以使用生成器函数重写该类，在\_\_iter\_\_魔法方法中返回一个迭代器(通过生成器实现)，之后使用iter方法来包装该类的示例，使其变成一个迭代器，在不定义\_\_next\_\_魔法方法的条件下即可使用next方法来遍历。如下所示：

```python
from collections.abc import Iterable, Iterator
class Company(object):
    def __init__(self, employee_list):
        self.employee = employee_list

    def __iter__(self):
        i = 0
        try:
            while True:
                v = self.employee[i]
                yield v
                i += 1
        except IndexError:
            return


if __name__ == "__main__":
    company = Company(["tom", "bob", "jane"])
    # Company中没有定义__next__方法，先使用iter方法包装成一个迭代器
    my_itor = iter(company)
    # 自定义的类Company中没有定义__next__魔法方法，但是可以使用next方法
    print(next(my_itor))
```
#### 分批读取大文件

&nbsp;&nbsp;&nbsp;&nbsp;假设有如下一个场景：有一个500G的大文件，但是该文件只有一行，该文件中使用'\{\|\}'来作为行的分隔符。如下所示：

> I am a Student.\{\|\} As food is to the body, so is learning to the mind.\{\|\}Our bodies grow and muscles develop with the intake of adequate nutritious food. \{\|\}Likewise, we should keep learning day by day to maintain our keen mental power and expand our intellectual capacity.\{\|\}Constant learning supplies us with inexhaustible fuel for driving us to sharpen our power of reasoning, analysis, and judgment.


&nbsp;&nbsp;&nbsp;&nbsp;假设具有该格式的文件为500G，如果使用readline方法，一行500G的文件一般远超内存容量，此时可以利用生成器来分批读取。read(4096)为读取4096个字符的文件信息，因为4096个字符可能包含多个以\{\|\}划分的批数据，因此通过while newline in buf循环内的内容进行截断处理。

```python
# newline为文件内行分隔符，此处即为{|}
def myreadlines(f, newline):
    buf = ""
    while True:
        # 可能同时读取了多个以{|}划分的批数据，需要进行截取
        while newline in buf:
            pos = buf.index(newline)
            yield buf[:pos]
            buf = buf[pos + len(newline):]
        # 只读取文件中4096个字符
        chunk = f.read(4096)

        # 处理边界情况
        if not chunk:
        # 说明已经读到了文件结尾
            yield buf
            break
        buf += chunk


with open("test.txt") as f:
    for line in myreadlines(f, "{|}"):
        print(line)
```
&nbsp;&nbsp;&nbsp;&nbsp;结果如下
> I am a Student.<br>
 As food is to the body, so is learning to the mind.<br>
Our bodies grow and muscles develop with the intake of adequate nutritious food. <br>
Likewise, we should keep learning day by day to maintain our keen mental power and expand our intellectual capacity.<br>
Constant learning supplies us with inexhaustible fuel for driving us to sharpen our power of reasoning, analysis, and judgment.

