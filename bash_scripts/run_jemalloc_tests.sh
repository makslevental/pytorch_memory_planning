set -e
names="alexnet
deeplabv3_mobilenet_v3_large
deeplabv3_resnet101
deeplabv3_resnet50
densenet121
densenet161
densenet169
densenet201
efficientnet_b0
efficientnet_b1
efficientnet_b2
efficientnet_b3
efficientnet_b4
efficientnet_b5
efficientnet_b6
efficientnet_b7
fcn_resnet101
fcn_resnet50
googlenet
inception_v3
lraspp_mobilenet_v3_large
mnasnet0_5
mnasnet0_75
mnasnet1_0
mnasnet1_3
mobilenet_v2
mobilenet_v3_large
mobilenet_v3_small
regnet_x_16gf
regnet_x_1_6gf
regnet_x_32gf
regnet_x_3_2gf
regnet_x_400mf
regnet_x_800mf
regnet_x_8gf
regnet_y_16gf
regnet_y_1_6gf
regnet_y_32gf
regnet_y_3_2gf
regnet_y_400mf
regnet_y_800mf
regnet_y_8gf
resnet101
resnet152
resnet18
resnet34
resnet50
resnext101_32x8d
resnext50_32x4d
shufflenet_v2_x0_5
shufflenet_v2_x1_0
shufflenet_v2_x1_5
shufflenet_v2_x2_0
squeezenet1_0
squeezenet1_1
vgg11_bn
vgg11
vgg13_bn
vgg13
vgg16_bn
vgg16
vgg19_bn
vgg19
wide_resnet101_2
wide_resnet50_2"

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

for NAME in $names ; do
  mkdir -p "je_malloc_runs/${NAME}/"
#  echo python jemalloc_experiments.py $NAME --heap_profile
  for j in {0..0} ; do
#    NUM_LOOPS=$((10**j))
    NUM_LOOPS=10
    echo "${NAME}"
#    PRINT_JEMALLOC_HEAP=1 OVERSIZED_THRESHOLD=1073741824 python jemalloc_experiments.py $NAME --num_workers=1 --num_loops=10 > "je_malloc_runs/${NAME}/heap_profile.csv"
    for i in 0 6 ; do
      NUM_WORKERS=$((2**i))
      echo $NUM_WORKERS
      # basically a normal run?
      OVERSIZED_THRESHOLD=1073741824 python jemalloc_experiments.py $NAME --num_workers=$NUM_WORKERS --num_loops=$NUM_LOOPS
#      jsonxf -i "je_malloc_runs/${NAME}/257_${NUM_WORKERS}_${NUM_LOOPS}.json" -o "je_malloc_runs/${NAME}/257_${NUM_WORKERS}_${NUM_LOOPS}.json"

      # severely limited run - no thread cache, and very few arenas (2 + oversize) but really everything should go to oversize
      NARENAS=1
      OVERSIZED_THRESHOLD=32 JEMALLOC_CONF="narenas:${NARENAS},lg_tcache_max:6" python jemalloc_experiments.py $NAME --narenas $NARENAS --num_workers=$NUM_WORKERS --num_loops=$NUM_LOOPS
#      jsonxf -i "je_malloc_runs/${NAME}/1_${NUM_WORKERS}_${NUM_LOOPS}.json" -o "je_malloc_runs/${NAME}/1_${NUM_WORKERS}_${NUM_LOOPS}.json"
    done
  done
  break
done
#MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 JEMALLOC_CONF=narenas:1,lg_tcache_max:6 /home/mlevental/miniconda3/envs/pytorch_shape_inference_memory_planning/bin/python /home/mlevental/dev_projects/pytorch_memory_planning/jemalloc_experiments.py resnet50 --num_workers 32