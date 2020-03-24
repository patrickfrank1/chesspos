for i in {10..12}
do
for j in {0..5}
do
echo "bsub -n 1 -W 24:00 -R 'rusage[mem=4096]' python pgn2pos.py /cluster/scratch/pafrank/lichess_db_standard_rated_2018-$i-0$j.pgn --save_position /cluster/scratch/pafrank/lichess_db_standard_rated_2018-$i-0$j-bb-strong --tuples True --save_tuples /cluster/scratch/pafrank/lichess_db_standard_rated_2018-$i-0$j-tuples-strong --chunksize 10000 --filter time_min=61 --filter elo_min=2000"
done
done


