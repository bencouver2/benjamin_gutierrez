#!/bin/sh


for i in -5 -4 -3 -2 -1  1 2 3 4 5; do

    count=`grep ^$i sentiment.txt | wc -l`
    echo $i,$count

done


