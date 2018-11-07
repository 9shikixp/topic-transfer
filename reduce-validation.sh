cd ~/work/topic-transfer/src/
case $HOSTNAME in
  "clover11" ) DATA=yelpf;;
  "clover12" ) DATA=db_pedia ;;
  "denso" ) DATA=ag_news ;;
esac
echo mode=$DATA gpu_id=$1
START=$(($1+1))
for i in `seq $START 2 5`
do
  echo python transfer-soft-reduce.py -m $DATA -g $1 -d 17 -v 0 -r $((4**$i))
  python transfer-soft-reduce.py -m $DATA -g $1 -d 17 -v 0 -r $((4**$i))
done
START=$((($1+1)%2+1))
for i in `seq $START 2 5`
do
  echo python train-scratch-reduce.py -m $DATA -g $1 -d 17 -v 0 -r $((4**$i))
  python train-scratch-reduce.py -m $DATA -g $1 -d 17 -v 0 -r $((4**$i))
done
