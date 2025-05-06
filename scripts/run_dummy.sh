for i in 0 1 2 3 4 5
do
	nohup bash scripts/run_single_dummy.sh $i &
done
