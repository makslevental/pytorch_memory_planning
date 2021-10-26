for i in {1..8} ; do
    perf stat python demo.py cpu $((10**$i))
#    BACKEND=cpu NUM_ELEMENTS=$((10**$i)) perf stat ./demo_fused
#    BACKEND=cpu NUM_ELEMENTS=$((10**$i)) perf stat ./demo_unfused
done