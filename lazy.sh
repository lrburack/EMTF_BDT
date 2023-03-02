modes=(3 5 6 1 7 9 10 11 12 13 14 15)

for i in ${modes[@]}; do
	echo $i
	mkdir ./${i}
done
