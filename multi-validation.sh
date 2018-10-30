cd ~/work/topic-transfer/src/
case $HOSTNAME in
  "clover11" ) DATA=yelpf;;
  "clover12" ) DATA=db_pedia ;;
  "denso" ) DATA=ag_news ;;
esac
echo mode=$DATA gpu_id=$1
for i in `seq $1 2 4`
do
  echo python topic-train-multi-3.py -m $DATA -g $1 -d 17 -v $i
  python topic-train-multi-3.py -m $DATA -g $1 -d 17 -v $i
  echo python transfer-multi-3.py -m $DATA -g $1 -d 17 -v $i
  python transfer-multi-3.py -m $DATA -g $1 -d 17 -v $i
done
