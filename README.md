note you need to build jemalloc with

```shell
mkdir build
./configure --prefix=`pwd`/build --with-jemalloc-prefix=je --disable-initial-exec-tls --enable-prof
make -j50
make install

pip install cpp_src/jemalloc_bindings/
```

```shell
JEMALLOC_CONF="prof_active:true,prof:true"
```