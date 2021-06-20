# Graph Convolutional Neural Networks

**Parallelization of the Algorithm:** _Graph Convolutional Neural Networks_


## Possible errors and how to fix them:

1) **For MacOS users:** If you're getting the error (_clang: error: unsupported option '-fopenmp'_) while running the **make** command with **-fopenmp**, try to pass **macOS** _as an argument_ to the **make** command (_set the value to **true**_):

```make
$ make macOS=true
```

