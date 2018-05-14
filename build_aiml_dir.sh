#!/bin/bash

data_dir='/home/z249865/aiml/data/HandWrite'

mkdir -p ./aiml/data

for dir in $data_dir/* ; do
    if [ -d $dir ] ; then
        name=`basename $dir`
        echo link $dir ./aiml/data/$name
        ln -s $dir ./aiml/data/$name
    fi
done

