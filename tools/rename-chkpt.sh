#!/bin/bash

ren_from=$1
ren_to=$2

for file in $ren_from.* ; do
    newname=`echo $file | sed -e s/$ren_from/$ren_to/`
    echo mv $file $newname
    mv $file $newname
done
