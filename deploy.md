# ONNX

## 1. run pth model to dump necessary data preparing for onnx model generation
```shell
tools/dist_test_pth.sh projects/configs/petr/petr_r50_gridmask_p4_800x480_from_scratch_hozon_pth.py ckpts/HOZON/hozon_epoch_32.pth 1 --eval=bbox
```


## 2.run onnx generation
```shell
tools/dist_test_onnx.sh projects/configs/petr/petr_r50_gridmask_p4_800x480_from_scratch_hozon_onnx.py ckpts/HOZON/hozon_epoch_32.pth 1 --eval=bbox
```



