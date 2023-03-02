
modes=(3 5 6 7 9 10 11 12 13 14 15)

for i in ${modes[@]}; do
    #Note the last job submitted
    python3 BDT.py -m $i &
done
