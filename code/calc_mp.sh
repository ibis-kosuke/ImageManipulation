#!/bin/bash

files="/data/unagi0/ktokitake/encdecmodel/birds/output/birds_EncDecModel_2021_01_01_15_11_39/Model/*"
array=()
for filepath in $files; do
    array+=$(basename $filepath)
done
unset array[0]
array_sort = sort -
#   python main.py --cfg cfg/eval_bird.yml --gpu 2 --netG $1 --netG_epoch $filebase

    
    
       
	  
