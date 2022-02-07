---
title: sharement of paremeters between shell scripts
date: 2022-02-07 13:51:05
tags:["linux","study note"]
---
# Problem 
Today some problem occurs when I want to write a shell script that helps me `cd` to some deep directories which are long enough for me, a lazy man, to write a script. I found that `cd` does't work in a `.sh` file when I use `bash` to execute it. Then I learned that a child shell is created to execute the commands in the scripts when using `bash`. To solve this problem, the most fundmental way is to use `source`, which won't create a child shell.

After solving it, I wondered that how does two shell scripts share their parameters when one script will create a child shell in another's child shell, which means that you can't just use `bash config.sh` to tell the parent shell what the paraments in the `config.sh` are, and using `source config.sh` won't tell the parent shell's another child shell(e.g. step1.sh) what the paraments are.(like stucture below) 
```bash
#in the `parent.sh`
source config.sh

bash step1.sh
#`step1.sh` won't get the parameters in the `copnfig.sh`
```
# Solvement
There are three kinds of variable in shell scripts:
* local variable: difined by `local` statement, scope: the function in which it is defined
* global variable: default, scope: the shell(not including child shells) 
* environment variable: defined by `export` command, scope: the shell(including child shells)

(from[ningyuwhut's blog](https://www.ningyuwhut.github.io/cn/2019/06/share-shell-variable-between-scripts))

Figuring out these three kinds of variable helps a lot in solving the probelms above.
# Example
I used three shell scripts to show it.

1. child_globalconfig.sh
```bash
#It is child_globalconfig.sh
#!/bin/bash
#-*-coding:UTF-8-*-
#Author: LeafLight
#Date: 2022-02-07
srcp="successful"
export envp="SUCCESSFUL"
```

2.childShell.sh
```bash
#It is childShell.sh
#!/bin/bash
#-*-coding:UTF-8-*-
#Author: LeafLight
#Date: 2022-02-07
echo "here is the child shell to show the advantage of using export to create environment parameter."
echo "envp: $envp"
echo "srcp: $srcp"
```

3.parentShell.sh
```bash
#It is parentShell.sh
#!/bin/bash
#-*-coding:UTF-8-*-
#Author: LeafLight
#Date: 2022-02-07
echo "---"
echo "here is the parent shell"
echo "load the config by source"
source ./child_globalconfig.sh

echo "---"
echo "parent shell tries to echo the srcp and envp"
echo "envp: $envp"
echo "srcp: $srcp"

echo "---"
echo "child shell(by bash)"
bash ./childShell.sh

echo "---"
echo "child shell(by source)"
source ./childShell.sh
```
4.Running the `parentShell.sh`
```bash
$ bash parentShell.sh
---
here is the parent shell
load the config by source
---
parent shell tries to echo the srcp and envp
envp: SUCCESSFUL
srcp: successful
---
child shell(by bash)
here is the child shell to show the advantage of using export to create environment parameter.
envp: SUCCESSFUL
srcp: 
---
child shell(by source)
here is the child shell to show the advantage of using export to create environment parameter.
envp: SUCCESSFUL
srcp: successful

```
