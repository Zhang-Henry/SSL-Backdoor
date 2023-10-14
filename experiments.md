|  | byol |jigsaw|moco|rotent|
| ----------- | ----------- |-----------|-----------|-----------|
| origin backdoor (FP) |4053 |601|4010|3892|
| ins filter backdoor (FP) |39|863|95|429|
| custom filter backdoor (FP) |1127 |3900|3479|3432|
| unet (FP) | |ing|59|222|
| ctrl (FP) |15 |41|17|161|
|  ssim0.9407_wd23218.4 (FP) | ||462||
|  ssim0.9419_wd12046.4 | ||35||
|  ssim0.9022_wd28007.5 | ||1222||
|  ssim0.7572_wd40717.9 | ||3904||
|  ssim0.9086_wd22540.1_lpips | ||1297||
|  ssim0.7130_wd31132.1_lp | ||2107||
|  ssim0.9045_wd249917.8_norm | ||500||
|  ssim0.9198_wd227679.3_norm_lp | ||734||
|  ssim0.9236_wd226912.4_norm_lp | ||1664||
|  ssim0.9405_wd234999.9_norm_lp | ||729||
|  ssim0.7970_wd2.29_lp_f | ||337||

---

# byol
## original backdoor
Rottweiler ,26       ,37.0      ,11.0 , ,50.0 ,4053.0

## ins filter
Rottweiler,26       ,40.0      ,15.0 , ,41.0 ,39.0

## custom filter ()
Rottweiler,26       ,33.0      ,11.0 , ,43.0 ,1127.0

## unet
Rottweiler,26,1.0,14.0,,44.0,809.0

---

# jigsaw

## original backdoor

Rottweiler,26       ,10.0      ,44.0 , ,30.0 ,601.0

## ins filter
Rottweiler,26       ,3.0       ,15.0 , ,35.0 ,863.0


# custom filter
Rottweiler,26       ,1.0       ,12.0 , ,50.0 ,3900.0

## ctrl
Rottweiler,26       ,0.0       ,15.0 , ,4.0  ,41.0

## unet

---
# moco
## original backdoor
Rottweiler,26       ,3.0       ,0.0  , ,50.0 ,4010.0

## ins filter
Rottweiler,26       ,3.0       ,49.0  , ,4.0 ,95.0


## custom filter
Rottweiler,26       ,1.0       ,1.0  , ,49.0 ,3479.0
## unet
Rottweiler,26       ,7.0       ,7.0  , ,25.0 ,40.0
Rottweiler,26,2.0,2.0,,36.0,462.0
Rottweiler,26,0.0,1.0,,44.0,1222.0
Rottweiler,26,0.0,0.0,,50.0,3904.0

## ctrl
Rottweiler,26       ,13.0      ,9.0  , ,25.0 ,17.0

---

# rotent
## original backdoor
Rottweiler,26       ,2.0       ,10.0 , ,50.0 ,3892.0

## ins filter
Rottweiler,26       ,2.0       ,18.0 , ,32.0 ,429.0

## custom filter
Rottweiler,26       ,1.0       ,5.0  , ,50.0 ,3432.0

## unet
Rottweiler,26       ,3.0       ,23.0 , ,25.0 ,222.0

## ctrl
Rottweiler,26       ,9.0       ,15.0 , ,23.0 ,161.0


# todolist
- [] 1. ablate wd，看同样ssim94，哪个效果更好
- [] 2. ablate ssim，看同样wd，哪个效果更好
- [] 3. 验证ssim=0.90的时候，wd=28007.5，是否可以比ssim=0.94达到更好的效果
- [] 4. 验证FP达到3k-4k的时候，wd最小是多少，ssim最大是多少