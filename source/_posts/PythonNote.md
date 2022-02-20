---
title: Python Learning Note
tags: ["python","note"]
---
## 2022/2/6

### Execute python scripts with options and arguments

learning from [runnoob](https://www.runnoob.com/python3-command-line-arguments.html)

Key:
1. Importing two modules: `sys` and `getopt`
2. Create a loop structure to get the opt. and arg.

It can promote the interaction between python and shell scripts.
```python
#!usr/bin/python
# -*- coding:utf-8 -*-
# learning from runnoob.com/python3/python3-command-line-arguments.html
import sys
import getopt

def Eadd():
    NAME = None
    DESCRIPTION = None
    # skip the first arg. : the name of this script 
    argv = sys.argv[1:]

    try:
        # short options mode
        opts, args = getopt.getopt(argv, 'N:M:')
    except:
        print('Error')

    for opt,val in opts:
        if opt in ['-N']:
            NAME = val
        elif opt in ['-M']:
            DESCRIPTION = val

    print('argv(skip the first arg.)',argv,'\n')
    print('NAME:', NAME, 'DESCRIPTION:', DESCRIPTION, '\n')
    print('args(getopt)', args, '\n')

Eadd()

```
## 2022/2/8

### Get the size, creation time, access time, modification time of file

from [cnblogs](https://www.cnblogs.com/shaosks/p/5614630.html)
0. Get the time stamp

```python
import time
import datetime
import os

timestamp = time.time()
```
1. Change the time stamp into time:
```python
def TimeStampToTime(timestamp):
	timeStruct = time.localtime(timestamp)
	return time.strftime("%Y-%m-%d %H:%M:%S",timeStruct)
```
or an easier way:
```python
localtime = time.asctime(time.localtime(time.time))
```
2. Get the size of a file(MB)
```python
def get_FileSize(filePath):
	filePath = unicode(filePath,'utf8')
	fsize = os.path.getsize(filePath)
	fsize = fsize/float(1024*1024)
	return round(fsize, 2)
```

3. Get the access time of a file
```python
def get_FileAccessTime(filePath):
	filePath = unicode(filePath, 'utf8')
	t = os.path.getatime(filePath)
	return TimeStampToTime(t)
```

4. Get the creation time of a file
```python
def get_FileCreateTime(filePath):
	filePath = unicode(filePath, 'utf8')
	t = os.path.getctime(filePath)
	return TimeStampToTime(t)
```

5. Get the modification time of a file
```python
def get_FileModifyTime(filePath):
	filePath = unicode(filePath,'utf8')
	t = os.path.getmtime(filePath)
	return TimeStampToTime(t)
```

