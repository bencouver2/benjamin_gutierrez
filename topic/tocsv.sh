#!/bin/sh

for i in 1 2 3 4 5 6 7 8 9 10; do
echo "probability,word" > t$i.csv
awk -F"+" 'NR=='$i'{print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10}' example.txt  | tr -s ' '  '\n' | sed 's/"//g' |sed 's/*/,/g' >> t$i.csv

done

