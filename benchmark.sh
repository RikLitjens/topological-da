bags=(1 23 36 60)


echo "method,bag,ec_time,total_time" >> results.csv
for i in 1 2 3 4 5; do
    for bag in ${bags[@]}; do
        d=$(pwd)
        res=$(python $d/Cherry-trees/src/main.py --method cnn --bag $bag | tail -1)
        echo "cnn,$bag,$(echo $res | awk '{print $6","$10"')" >> results.csv

        res=$(python $d/Cherry-trees/src/main.py --method homology --bag $bag | tail -1)
        echo "homology,$bag,$(echo $res | awk '{print $6","$10"')" >> results.csv

        res=$(python $d/Cherry-trees/src/main.py --method reeb --bag $bag | tail -1)
        echo "reeb,$bag,$(echo $res | awk '{print N/A,"$4"')" >> results.csv
    done
done