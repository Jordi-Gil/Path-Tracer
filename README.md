# Path Tracing

Wellcome to Path Tracing Project!

### Path Tracing in CPU

To see the help
```sh
./path_tracing_sec -h
```
To run the default version
```sh
./path_tracing_sec -d
```

### Path Tracing 1 GPU

To run
```sh
qsub -l cuda job_1GPUs.sh
```
You can modify the "job" to your liking, check it before sending it to the execution queue.

### Path Tracing N GPU

To run
```sh
qsub -l cuda job_NGPUs.sh
```
You can modify the "job" to your liking, check it before sending it to the execution queue.

### Path Tracing N GPU Dynamic Parallelism

To run
```sh
qsub -l cuda job_NGPUs_dynamic.sh
```
You can modify the "job" to your liking, check it before sending it to the execution queue.

### Clean and Compile

```sh
make clean
make all -j
```


[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)
