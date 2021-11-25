note you need to build jemalloc with

```shell
mkdir build
./configure --prefix=`pwd`/build --enable-stats --with-jemalloc-prefix=je --disable-initial-exec-tls 
make -j50
make install

pip install cpp_src/jemalloc_bindings/
```

```shell
JEMALLOC_CONF="narenas:1,prof_active:true,prof:true"
narenas:1,lg_tcache_max:1,tcache:false
```