# ls-ldcache

A simple tool for reading [ldconfig](https://man7.org/linux/man-pages/man8/ldconfig.8.html) cache files.

```bash
$ make build

$ ./ls-ldcache -r libcuda

libcuda.so.1 [arch=0x300 (X8664_LIB64)]
    - /lib/x86_64-linux-gnu/libcuda.so.1
    - /lib/x86_64-linux-gnu/libcuda.so.470.141.03

libcuda.so.1 [arch=0x0 (I386_LIB32)]
    - /lib/i386-linux-gnu/libcuda.so.1
    - /lib/i386-linux-gnu/libcuda.so.470.141.03

libcuda.so [arch=0x300 (X8664_LIB64)]
    - /lib/x86_64-linux-gnu/libcuda.so
    - /lib/x86_64-linux-gnu/libcuda.so.1
    - /lib/x86_64-linux-gnu/libcuda.so.470.141.03

libcuda.so [arch=0x0 (I386_LIB32)]
    - /lib/i386-linux-gnu/libcuda.so
    - /lib/i386-linux-gnu/libcuda.so.1
    - /lib/i386-linux-gnu/libcuda.so.470.141.03
```
